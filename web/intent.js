// In-browser probe decision model. Mirrors ProbeDecisionModel.predict() from
// hermitlm/tools/decision_model.py. One forward pass gives `last_hidden`
// (see HermitModel.classify in model.js). This module applies the exported
// linear heads (web/probe.json) with temperature-scaled softmax.

// Fallbacks for probe JSON files fitted before the conformal gates existed.
// A calibrated probe carries clarify_threshold / escalation_threshold, and
// both gates read those stored values rather than any hard-coded boundary.
const LEGACY_CLARIFY_THRESHOLD = 0.75;
const LEGACY_ESCALATION_THRESHOLD = 0.5;

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

export function softmax(logits, temperature = 1.0, prior = null, lambda = 0) {
  let scaled = logits.map((v) => v / temperature);
  if (prior && lambda > 0) {
    const logPrior = prior.map((p) => Math.log(Math.max(p, 1e-6)));
    scaled = scaled.map((v, i) => v - lambda * (logPrior[i] || 0));
  }
  let maxLogit = -Infinity;
  for (const v of scaled) if (v > maxLogit) maxLogit = v;
  const exps = scaled.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function classifyIntent(lastHidden, probe) {
  const lambda = probe.prior_lambda ?? 1.0;
  const logits7 = linear(lastHidden, probe.head7.weight, probe.head7.bias);
  const proba7 = softmax(logits7, probe.temperature7, probe.priors?.coarse, lambda);
  let best = 0;
  for (let i = 1; i < proba7.length; i++) {
    if (proba7[i] > proba7[best]) best = i;
  }
  const logitsBin = linear(lastHidden, probe.head_bin.weight, probe.head_bin.bias);
  const probaBin = softmax(logitsBin, probe.temperature_bin, probe.priors?.confused, lambda);
  // Calibrated out-of-scope probability. Older probes without head_yes fall
  // back to the legacy confused head, matching router.classify_intent().
  let pYes = probaBin[1];
  if (probe.head_yes) {
    const logitsYes = linear(lastHidden, probe.head_yes.weight, probe.head_yes.bias);
    const probaYes = softmax(logitsYes, probe.temperature_yes, probe.priors?.yes, lambda);
    pYes = probaYes[1];
  }
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
    p_yes: pYes,
    proba,
    top2: ranked.slice(0, 2),
    // Debiasing provenance, matching classify_intent() in router.py.
    level: Array.isArray(probe.priors?.coarse) && lambda > 0 ? "L0" : "raw",
    rotations: 1,
    clarify_threshold: probe.clarify_threshold ?? LEGACY_CLARIFY_THRESHOLD,
    escalation_threshold: probe.escalation_threshold ?? LEGACY_ESCALATION_THRESHOLD,
  };
}

export function classifyFine(lastHidden, probe) {
  if (!probe.head21) return null;
  const lambda = probe.prior_lambda ?? 1.0;
  const logits = linear(lastHidden, probe.head21.weight, probe.head21.bias);
  const probs = softmax(logits, probe.temperature21 || 1.0, probe.priors?.fine, lambda);
  let best = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[best]) best = i;
  }
  return { fine: probe.fine_options[best], confidence: probs[best] };
}
