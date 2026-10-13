// Client-side autoregressive generation over the ONNX-exported HermitLM,
// mirroring HermitLM.generate() / HermitInference.chat() in
// hermitlm/training/model.py and hermitlm/runtime/inference.py.

// Bump this when model artifacts change, or browsers keep serving the old one.
const CACHE_NAME = "hermit-assets-v2";
const PAD_ID = 0;
const BOS_ID = 1;
const EOS_ID = 2;
const MAX_SEQ_LEN = 512;

function sampleNext(logits, temperature, topK, topP = 0.95) {
  const n = logits.length;
  const scaled = new Float32Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;

  // top-K pre-filter
  let threshold = -Infinity;
  if (topK > 0 && topK < n) {
    const sorted = Array.from(scaled).sort((a, b) => b - a);
    threshold = sorted[topK - 1];
  }

  // softmax over survivors
  let maxLogit = -Infinity;
  for (let i = 0; i < n; i++) {
    if (scaled[i] >= threshold && scaled[i] > maxLogit) maxLogit = scaled[i];
  }

  const probs = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (scaled[i] < threshold) continue;
    const p = Math.exp(scaled[i] - maxLogit);
    probs[i] = p;
    sum += p;
  }
  for (let i = 0; i < n; i++) probs[i] /= sum;

  // top-P (nucleus) filter on normalized probs
  if (topP < 1.0) {
    const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    let cum = 0;
    let cutoff = n;
    for (let k = 0; k < order.length; k++) {
      cum += probs[order[k]];
      if (cum >= topP) {
        cutoff = k + 1;
        break;
      }
    }
    const allowed = new Set(order.slice(0, Math.max(1, cutoff)));
    let renorm = 0;
    for (let i = 0; i < n; i++) {
      if (!allowed.has(i)) probs[i] = 0;
      renorm += probs[i];
    }
    for (let i = 0; i < n; i++) probs[i] /= renorm;
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
  static async fetchBuffer(url, { onProgress } = {}) {
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
      const cache = await caches.open(CACHE_NAME);
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

  static async load(modelUrl, { onProgress } = {}) {
    const data = await HermitModel.fetchBuffer(modelUrl, {
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
    return `<|im_start|>user\n${userInput}<|im_end|>\n<|im_start|>assistant\n`;
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
