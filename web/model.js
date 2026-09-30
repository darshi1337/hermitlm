// Client-side autoregressive generation over the ONNX-exported HermitLM,
// mirroring HermitLM.generate() / HermitInference.chat() in
// hermitlm/training/model.py and hermitlm/runtime/inference.py.

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

  static async load(modelUrl, { onProgress } = {}) {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    });
    if (onProgress) onProgress("model ready");
    return new HermitModel(session);
  }

  formatPrompt(userInput) {
    return `<|im_start|>user\n${userInput}<|im_end|>\n<|im_start|>assistant\n`;
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
