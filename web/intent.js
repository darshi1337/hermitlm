// In-browser probe decision model. Mirrors ProbeDecisionModel.predict() from
// hermitlm/tools/decision_model.py. One forward pass gives `last_hidden`
// (see HermitModel.classify in model.js). This module applies the exported
// linear heads (web/probe.json) with temperature-scaled softmax.

export const CONFUSED_THRESHOLD = 0.9;
export const CLARIFY_THRESHOLD = 0.75;

export function linear(hidden, weight, bias) {
  // hidden: Float32Array/number[n], weight: number[out][n], bias: number[out]
  const out = new Array(bias.length);
  for (let o = 0; o < bias.length; o++) {
    let s = bias[o];
    const row = weight[o];
    for (let i = 0; i < hidden.length; i++) s += row[i] * hidden[i];
    out[o] = s;
  }
  return out;
}

export function softmax(logits, temperature = 1.0) {
  let maxLogit = -Infinity;
  const scaled = logits.map((v) => v / temperature);
  for (const v of scaled) if (v > maxLogit) maxLogit = v;
  const exps = scaled.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function classifyIntent(lastHidden, probe) {
  const logits7 = linear(lastHidden, probe.head7.weight, probe.head7.bias);
  const proba7 = softmax(logits7, probe.temperature7);
  let best = 0;
  for (let i = 1; i < proba7.length; i++) {
    if (proba7[i] > proba7[best]) best = i;
  }
  const logitsBin = linear(lastHidden, probe.head_bin.weight, probe.head_bin.bias);
  const probaBin = softmax(logitsBin, probe.temperature_bin);
  const label = probe.options[best];
  const proba = {};
  probe.options.forEach((letter, i) => (proba[letter] = proba7[i]));
  const ranked = probe.options
    .map((letter, i) => [letter, proba7[i]])
    .sort((x, y) => y[1] - x[1]);
  return {
    label,
    name: probe.names[label],
    confidence: proba7[best],
    p_confused: probaBin[1],
    proba,
    top2: ranked.slice(0, 2),
  };
}

export function classifyFine(lastHidden, probe) {
  if (!probe.head21) return null;
  const logits = linear(lastHidden, probe.head21.weight, probe.head21.bias);
  const probs = softmax(logits, probe.temperature21 || 1.0);
  let best = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[best]) best = i;
  }
  return { fine: probe.fine_options[best], confidence: probs[best] };
}
