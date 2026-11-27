// Client-side autoregressive generation over the ONNX-exported HermitLM,
// mirroring HermitLM.generate() / HermitInference.chat() in
// hermitlm/training/model.py and hermitlm/runtime/inference.py.

// Cache keys are derived from the asset manifest hash, so a new deploy
// automatically invalidates the old model without a manual version bump.
const DEFAULT_ASSET_VERSION = "v2";

export async function fetchAssetVersion() {
  try {
    const res = await fetch("./asset-manifest.json", { cache: "no-store" });
    if (!res.ok) return DEFAULT_ASSET_VERSION;
    const manifest = await res.json();
    return (manifest["model.quant.onnx"] || {}).sha256 || DEFAULT_ASSET_VERSION;
  } catch (e) {
    return DEFAULT_ASSET_VERSION;
  }
}

const PAD_ID = 0;
const BOS_ID = 1;
const EOS_ID = 2;
const MAX_SEQ_LEN = 512;

function sampleNext(logits, temperature, topK, topP = 0.95) {
  const n = logits.length;
  const scaled = new Float32Array(n);
  let maxLogit = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i] / temperature;
    scaled[i] = v;
    if (v > maxLogit) maxLogit = v;
  }

  // top-K threshold using a bounded insertion table (topK is small).
  // Sorting the full ~3k vocabulary twice per token was pure overhead.
  let threshold = -Infinity;
  const k = topK > 0 ? Math.min(topK, n) : 0;
  if (k > 0) {
    const best = new Float32Array(k);
    best.fill(-Infinity);
    let count = 0;
    for (let i = 0; i < n; i++) {
      const v = scaled[i];
      if (count === k && v <= threshold) continue;
      let pos = count < k ? count++ : k - 1;
      while (pos > 0 && best[pos - 1] < v) {
        best[pos] = best[pos - 1];
        pos--;
      }
      best[pos] = v;
      threshold = best[k - 1];
    }
  }

  // softmax over survivors only; keep their indices for the nucleus filter
  const probs = new Float32Array(n);
  const candidates = [];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (scaled[i] < threshold) continue;
    const p = Math.exp(scaled[i] - maxLogit);
    probs[i] = p;
    candidates.push(i);
    sum += p;
  }
  for (const i of candidates) probs[i] /= sum;

  // top-P (nucleus) filter on normalized probs; sort only the survivors
  if (topP < 1.0 && candidates.length > 1) {
    candidates.sort((a, b) => probs[b] - probs[a]);
    let cum = 0;
    let cutoff = candidates.length;
    for (let j = 0; j < candidates.length; j++) {
      cum += probs[candidates[j]];
      if (cum >= topP) {
        cutoff = Math.max(1, j + 1);
        break;
      }
    }
    const allowed = new Uint8Array(n);
    for (let j = 0; j < cutoff; j++) allowed[candidates[j]] = 1;
    let renorm = 0;
    for (const i of candidates) {
      if (!allowed[i]) probs[i] = 0;
      else renorm += probs[i];
    }
    for (const i of candidates) probs[i] /= renorm;
    sum = 1;
  }

  let r = Math.random() * sum;
  for (let i = 0; i < n; i++) {
    r -= probs[i];
    if (r <= 0 && probs[i] > 0) return i;
  }
  return n - 1;
}

export class HermitModel {
  constructor(session) {
    this.session = session;
  }

  // Cache-first fetch. Repeat visits skip the ~9MB download entirely.
  // Reports byte progress so the status pill shows where loading stalls.
  static async fetchBuffer(url, { onProgress, version = DEFAULT_ASSET_VERSION } = {}) {
    const cacheName = `hermit-assets-${version}`;
    const report = (done, total) => {
      if (onProgress) onProgress(done, total);
    };
    const readWithProgress = async (res) => {
      const total = Number(res.headers.get("content-length")) || 0;
      if (!res.body || !res.body.getReader) {
        const buf = await res.arrayBuffer();
        report(buf.byteLength, total || buf.byteLength);
        return buf;
      }
      const reader = res.body.getReader();
      const chunks = [];
      let done = 0;
      for (;;) {
        const { done: finished, value } = await reader.read();
        if (finished) break;
        chunks.push(value);
        done += value.length;
        report(done, total);
      }
      const buf = new Uint8Array(done);
      let offset = 0;
      for (const chunk of chunks) {
        buf.set(chunk, offset);
        offset += chunk.length;
      }
      return buf.buffer;
    };
    try {
      const cache = await caches.open(cacheName);
      const hit = await cache.match(url);
      if (hit) {
        const buf = await hit.arrayBuffer();
        report(buf.byteLength, buf.byteLength);
        return buf;
      }
      const net = await fetch(url, { signal: AbortSignal.timeout(120000) });
      if (!net.ok) throw new Error(`fetch failed: ${url}`);
      await cache.put(url, net.clone());
      return await readWithProgress(net);
    } catch (e) {
      // No Cache API (private mode, file://) or a bad entry: plain fetch.
      const net = await fetch(url, { signal: AbortSignal.timeout(120000) });
      if (!net.ok) throw new Error(`fetch failed: ${url}`);
      return await readWithProgress(net);
    }
  }

  static async load(modelUrl, { onProgress, version } = {}) {
    const data = await HermitModel.fetchBuffer(modelUrl, {
      version,
      onProgress: onProgress
        ? (done, total) =>
            onProgress(
              total > 0 ? `model ${(done / 1e6).toFixed(1)}/${(total / 1e6).toFixed(1)}MB` : `model ${(done / 1e6).toFixed(1)}MB`
            )
        : undefined,
    });
    if (onProgress) onProgress("starting model");
    const session = await ort.InferenceSession.create(data, {
      executionProviders: ["wasm"],
    });
    if (onProgress) onProgress("model ready");
    return new HermitModel(session);
  }

  formatPrompt(userInput) {
    // Escape added-token markers so user text cannot open a new turn.
    const safe = String(userInput)
      .replaceAll("<|im_start|>", "<| im_start |>")
      .replaceAll("<|im_end|>", "<| im_end |>");
    return `<|im_start|>user\n${safe}<|im_end|>\n<|im_start|>assistant\n`;
  }

  // Single forward pass over the prompt. Returns the last-token hidden
  // state for the probe decision model. Returns null if the loaded artifact
  // predates the `last_hidden` output. Shares the chat session: no extra
  // download, one extra inference step.
  async classify(promptIds) {
    const context =
      promptIds.length > MAX_SEQ_LEN
        ? promptIds.slice(promptIds.length - MAX_SEQ_LEN)
        : promptIds;
    const inputTensor = new ort.Tensor(
      "int64",
      BigInt64Array.from(context.map((id) => BigInt(id))),
      [1, context.length]
    );
    const results = await this.session.run({ input_ids: inputTensor });
    if (!results.last_hidden) return null;
    return results.last_hidden.data;
  }

  async *generate(
    promptIds,
    { maxNewTokens = 80, temperature = 0.6, topK = 20, topP = 0.95 } = {}
  ) {
    let ids = promptIds.slice();

    for (let step = 0; step < maxNewTokens; step++) {
      const context =
        ids.length > MAX_SEQ_LEN ? ids.slice(ids.length - MAX_SEQ_LEN) : ids;

      const inputTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(context.map((id) => BigInt(id))),
        [1, context.length]
      );

      const results = await this.session.run({ input_ids: inputTensor });
      const logits = results.logits;
      const T = logits.dims[1];
      const vocabSize = logits.dims[2];
      const lastLogits = logits.data.subarray(
        (T - 1) * vocabSize,
        T * vocabSize
      );

      const nextId = sampleNext(lastLogits, temperature, topK, topP);
      ids.push(nextId);

      if (nextId === EOS_ID || nextId === BOS_ID) return;

      yield nextId;
    }
  }
}

export { PAD_ID, BOS_ID, EOS_ID, MAX_SEQ_LEN };
