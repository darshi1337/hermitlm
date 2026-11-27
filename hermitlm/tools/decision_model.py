"""System-1 decision model over HermitLM (single-pass constrained decoding).

Follows https://nishtahir.com/build-your-own-decision-model/ : instead of
autoregressive generation (one forward pass per token), do a single forward
pass, take the last-position logits, mask to the allowed option tokens and
softmax them into a calibrated decision.

The decision task is intent classification over the persona data
(`data/train.jsonl` / `data/eval.jsonl`, `{input, output, type}`). The mapping
collapses the 21 fine-grained `type` labels into 7 coarse intents. The
verbalizers are the single lowercase letters a-g. Lowercase matters. The vocab
of the tokenizer holds single tokens for `a`-`g` (ids 21-27), but uppercase
`A`-`G` are out-of-vocabulary and encode to nothing.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.tools.artifacts import verify_artifact
from hermitlm.tools.heads import FittedHead
from hermitlm.tools.prior import apply_prior
from hermitlm.training.model import HermitLM

# Fallbacks for probe artifacts fitted before the conformal gates existed.
# They are deliberately local to the artifact loader: runtime gates read the
# stored values, not the environment thresholds in hermitlm/settings.py.
LEGACY_CLARIFY_THRESHOLD = 0.75
LEGACY_ESCALATION_THRESHOLD = 0.5

# Coarse intent mapping: 7 balanced buckets covering all 21 fine types.
# Bucket sizes on train.jsonl (95k) stay within a factor of two:
# a~14.9k b~14.8k c~11.9k d~14.8k e~11.9k f~9.0k g~17.9k
COARSE_MEMBERS: dict[str, list[str]] = {
    "a": ["greeting", "bye", "lonely"],  # social
    "b": ["food", "water", "bubbles"],  # needs
    "c": ["glass", "environment", "movement"],  # tank
    "d": ["light", "color", "sound", "noise"],  # sense
    "e": ["state", "breathing"],  # body
    "f": ["temp_cold", "temp_hot", "night"],  # condition
    "g": ["misc", "confused", "reflection"],  # mind / chatter / out-of-scope
}

COARSE_NAMES: dict[str, str] = {
    "a": "social",
    "b": "needs",
    "c": "tank",
    "d": "sense",
    "e": "body",
    "f": "condition",
    "g": "mind",
}

OPTION_LETTERS: list[str] = ["a", "b", "c", "d", "e", "f", "g"]

TOOL_CLASSES: list[str] = ["math", "web", "faq", "chat"]

# Coarse intent to UI persona mode. Empty string means normal (no override).
INTENT_PERSONA: dict[str, str] = {
    "a": "",
    "b": "hungry",
    "c": "",
    "d": "guard",
    "e": "sleepy",
    "f": "sleepy",
    "g": "",
}

FINE_TO_COARSE: dict[str, str] = {
    fine: coarse for coarse, fines in COARSE_MEMBERS.items() for fine in fines
}

ALL_FINE_TYPES: list[str] = sorted(FINE_TO_COARSE)

# Fine-grained 21-way verbalizers: a-u are all single tokens in this vocab.
FINE21_ORDER: list[str] = ALL_FINE_TYPES
FINE21_LETTERS: list[str] = [chr(ord("a") + i) for i in range(len(FINE21_ORDER))]
FINE_TO_LETTER: dict[str, str] = {
    fine: letter for fine, letter in zip(FINE21_ORDER, FINE21_LETTERS)
}


def get_option_ids(tokenizer: Tokenizer) -> list[int]:
    """Resolve each option letter to a single token id.

    Raises ValueError if any letter is not a single token, so a tokenizer
    change fails loudly instead of silently mis-scoring.
    """
    ids: list[int] = []
    for letter in OPTION_LETTERS:
        encoded = tokenizer.encode(letter).ids
        if len(encoded) != 1:
            raise ValueError(f"option {letter!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"option token ids are not distinct: {ids}")
    return ids


def get_fine_option_ids(tokenizer: Tokenizer) -> list[int]:
    """Resolve each fine-grained letter (a-u) to a single token id."""
    ids: list[int] = []
    for letter in FINE21_LETTERS:
        encoded = tokenizer.encode(letter).ids
        if len(encoded) != 1:
            raise ValueError(f"option {letter!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"option token ids are not distinct: {ids}")
    return ids


def get_label_ids(tokenizer: Tokenizer, labels, fallback_labels=None) -> tuple[list[str], list[int]]:
    """Resolve constrained-decoding labels to single tokens.

    Score levels are naturally written as digits (`0`, `1`, ...), but a custom
    tokenizer may encode a digit into several tokens. Falling back to letters
    keeps constrained decoding valid instead of silently scoring a prefix
    token. Returns the labels actually used and their token ids.
    """
    labels = [str(label) for label in labels]
    ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label).ids
        if len(encoded) != 1:
            ids = []
            break
        ids.append(encoded[0])
    if len(ids) == len(labels) and len(set(ids)) == len(ids):
        return labels, ids

    fallback = [str(label) for label in (fallback_labels or [])]
    if len(fallback) != len(labels):
        fallback = [chr(ord("a") + i) for i in range(len(labels))]
    ids = []
    for label in fallback:
        encoded = tokenizer.encode(label).ids
        if len(encoded) != 1:
            raise ValueError(f"fallback label {label!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"fallback label ids are not distinct: {ids}")
    return fallback, ids


def option_assignment(shift: int) -> list[str]:
    """Return the class shown at each display letter ``a..g``.

    ``shift=2`` assigns ``c d e f g a b`` to ``a b c d e f g``. Rotating this
    assignment is the zero-shot L0 debias: option-letter bias is spread over
    classes and averaged out.
    """
    k = len(OPTION_LETTERS)
    shift %= k
    return [OPTION_LETTERS[(i + shift) % k] for i in range(k)]


def rotation_shifts(rotations: int) -> list[int]:
    """Evenly spaced assignments; two rotations cost two passes."""
    k = len(OPTION_LETTERS)
    if not 1 <= rotations <= k:
        raise ValueError(f"rotations must be in [1, {k}]")
    return [round(j * k / rotations) % k for j in range(rotations)]


def unrotate_display(values, shift: int) -> dict[str, float]:
    """Map display-letter values back onto their original class letters."""
    assignment = option_assignment(shift)
    return {cls: float(values[i]) for i, cls in enumerate(assignment)}


def debias_level(priors: dict, prior_lambda: float) -> str:
    """`L0` when a fitted coarse prior is actually applied, else `raw`."""
    return "L0" if ("coarse" in (priors or {}) and prior_lambda > 0) else "raw"


def format_prompt(query: str, assignment: list[str] | None = None, shift: int = 0) -> str:
    """Build the classification prompt. The model scores one letter next.

    ``assignment[i]`` is the class printed next to display letter ``i``.
    ``shift`` is shorthand for ``option_assignment(shift)`` (L0 debiasing).
    """
    classes = list(assignment) if assignment is not None else option_assignment(shift)
    if len(classes) != len(OPTION_LETTERS):
        raise ValueError("assignment must cover every option letter")
    options_text = "\n".join(
        f"{letter}. {COARSE_NAMES[cls]}" for letter, cls in zip(OPTION_LETTERS, classes)
    )
    return (
        "<|im_start|>user\n"
        f"Classify the intent of this message: {query}\n"
        f"Options:\n{options_text}\n"
        "Answer with one letter:\n"
        "<|im_start|>assistant\n"
    )



def constrained_probs(
    last_logits: torch.Tensor, option_ids: list[int] | torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Mask full-vocab logits to the allowed options and softmax with temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not isinstance(option_ids, torch.Tensor):
        option_ids = torch.tensor(option_ids, dtype=torch.long)
    option_ids = option_ids.to(last_logits.device)
    gathered = last_logits[..., option_ids].float() / temperature
    return F.softmax(gathered, dim=-1)


class DecisionModel:
    """Single-pass intent classifier on top of a HermitLM checkpoint."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
        self.config.vocab_size = self.tokenizer.get_vocab_size()
        self.model = HermitLM(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.option_ids = get_option_ids(self.tokenizer)

    @torch.no_grad()
    def predict_proba(self, query: str, temperature: float = 1.0, rotations: int = 2) -> dict[str, float]:
        shifts = rotation_shifts(rotations)
        unrotated = []
        for s in shifts:
            assignment = option_assignment(s)
            ids = self.tokenizer.encode(format_prompt(query, assignment)).ids
            ids = ids[-self.config.max_seq_len :]
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits, _ = self.model(idx)
            display_logp = torch.log_softmax(
                logits[0, -1, self.option_ids].float() / temperature, dim=-1
            )
            orig_logp = torch.empty_like(display_logp)
            orig_logp[torch.tensor([OPTION_LETTERS.index(c) for c in assignment])] = display_logp
            unrotated.append(orig_logp)
        combined = torch.stack(unrotated).mean(dim=0)
        probs = torch.softmax(combined, dim=-1)
        return {letter: float(p) for letter, p in zip(OPTION_LETTERS, probs.tolist())}

    @torch.no_grad()
    def predict(self, query: str, temperature: float = 1.0, rotations: int = 2) -> tuple[str, dict[str, float]]:
        proba = self.predict_proba(query, temperature, rotations)
        label = max(OPTION_LETTERS, key=lambda letter: proba[letter])
        return label, proba

def chat_prompt(query: str) -> str:
    """In-distribution chat prompt used for hidden-state probes."""
    return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


def score_prompt(query: str, response: str = "") -> str:
    """Prompt for the reply-quality score head: query plus a candidate reply."""
    return chat_prompt(query) + response


PROMPT_TEMPLATES = format_prompt("x") + "\n" + chat_prompt("x")


class LegacyHead:
    """Adapter for pre-selection probe artifacts."""

    def __init__(self, state, temperature, device):
        from torch import nn

        self.layer = None
        d_model = state["weight"].size(1)
        self.linear = nn.Linear(d_model, state["weight"].size(0)).to(device)
        self.linear.load_state_dict(state)
        self.temperature = float(temperature)

    def scores(self, hidden):
        return self.linear(hidden) / self.temperature

    def log_probs(self, hidden):
        return torch.log_softmax(self.scores(hidden), dim=-1)


class ProbeDecisionModel:
    """Supervised System-1 head: frozen LM forward pass plus linear probe.

    Same single-pass cost profile as constrained decoding, but trained. It
    predicts the 7 coarse intents plus P(confused) as the out-of-scope signal
    (option E from the article).
    """

    def __init__(self, checkpoint_path: str, tokenizer_path: str, probe_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
        self.config.vocab_size = self.tokenizer.get_vocab_size()
        self.model = HermitLM(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        probe = torch.load(probe_path, map_location=self.device, weights_only=True)
        selection = probe.get("selection")
        self.layer = selection["layer"] if selection else None
        verify_artifact(
            probe, checkpoint_path, tokenizer_path,
            dtype=str(next(self.model.parameters()).dtype),
            layer=self.layer,
        )
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.prior_lambda = float(probe.get("prior_lambda", 1.0))
        self.priors = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in probe.get("priors", {}).items()}
        if selection:
            heads = selection["heads"]
            self.head7 = FittedHead.from_dict(heads["coarse"]).to(self.device)
            self.head_bin = FittedHead.from_dict(heads["confused"]).to(self.device)
            self.head21 = FittedHead.from_dict(heads["fine"]).to(self.device) if "fine" in heads else None
            self.temperature7 = self.head7.temperature
            self.temperature_bin = self.head_bin.temperature
            self.temperature21 = self.head21.temperature if self.head21 else 1.0
        else:

            self.head7 = LegacyHead(probe["head7"], probe["temperature7"], self.device)
            self.head_bin = LegacyHead(probe["head_bin"], probe["temperature_bin"], self.device)
            self.head21 = LegacyHead(probe["head21"], probe.get("temperature21", 1.0), self.device) if "head21" in probe else None
            self.temperature7 = float(probe["temperature7"])
            self.temperature_bin = float(probe["temperature_bin"])
            self.temperature21 = float(probe.get("temperature21", 1.0))
        self.head_yes = self._aux_head(probe, "yes", "head_yes", "temperature_yes")
        self.head_score = self._aux_head(probe, "score", "head_score", "temperature_score")
        labels = probe.get("score_labels") or [str(k) for k in range(len(probe.get("score_levels", [])))]
        self.score_labels, _ = get_label_ids(self.tokenizer, labels)
        self.score_levels = [float(v) for v in probe.get("score_levels", [])]
        self.score_thresholds = [float(v) for v in probe.get("score_thresholds", [])]
        self.clarify_threshold = float(probe.get("clarify_threshold", LEGACY_CLARIFY_THRESHOLD))
        self.escalation_threshold = float(
            probe.get("escalation_threshold", LEGACY_ESCALATION_THRESHOLD)
        )
        self.target_error = float(probe.get("target_error", 0.1))
        self.threshold_coverage = {
            str(k): float(v) for k, v in probe.get("threshold_coverage", {}).items()
        }
        self.threshold_abstention = float(probe.get("threshold_abstention", 0.0))
        self.tool_head = None
        self.tool_layer = None

    def _aux_head(self, probe: dict, name: str, head_key: str, temp_key: str):
        """Load an optional head that may live in `selection` or at top level."""
        selection = probe.get("selection") or {}
        state = selection.get("heads", {}).get(name)
        if state is None:
            state = probe.get(head_key)
        if state is None:
            return None
        if isinstance(state, dict) and "mean" in state and "std" in state:
            return FittedHead.from_dict(state).to(self.device)
        temperature = float(probe.get(temp_key, 1.0))
        if isinstance(state, dict):
            temperature = float(state.get("temperature", temperature))
        return LegacyHead(state, temperature, self.device)

    @staticmethod
    def _head_layer(head) -> int | None:
        """`None` or -1 means the final normed hidden state (legacy probes)."""
        if head is None:
            return None
        layer = getattr(head, "layer", None)
        return None if layer is None or int(layer) == -1 else int(layer)

    @property
    def has_yes_head(self) -> bool:
        return self.head_yes is not None

    @property
    def has_score_head(self) -> bool:
        return self.head_score is not None and bool(self.score_levels)

    @property
    def level(self) -> str:
        """Which debias level the coarse prediction reflects."""
        return debias_level(self.priors, self.prior_lambda)

    @property
    def rotations(self) -> int:
        """Probe heads score one fixed prompt, so one forward pass is enough."""
        return 1

    @torch.no_grad()
    def _hidden(self, query: str, layer=None) -> torch.Tensor:
        """Hidden state at the last real token of a chat prompt."""
        return self._hidden_prompt(chat_prompt(query), layer)

    @torch.no_grad()
    def _hidden_prompt(self, prompt: str, layer=None) -> torch.Tensor:
        cache = getattr(self, "_hidden_cache", None)
        if cache is None:
            cache = {}
            self._hidden_cache = cache
        key = (prompt, layer)
        if key in cache:
            return cache[key]
        ids = self.tokenizer.encode(prompt).ids[-self.config.max_seq_len :]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        if layer is None:
            _, _, hidden = self.model(idx, return_hidden=True)
            value = hidden[0, len(ids) - 1]
        else:
            _, hidden = self.model.forward_with_hidden_layers(idx)
            value = hidden[0, len(ids) - 1, layer, :]
        if len(cache) >= 16:
            cache.clear()
        cache[key] = value
        return value

    def _log_probs(self, head, hidden, prior_key=None):
        lp = head.log_probs(hidden.unsqueeze(0))
        if prior_key in self.priors:
            lp = apply_prior(lp, self.priors[prior_key], self.prior_lambda)
        return lp

    @torch.no_grad()
    def predict(self, query: str) -> tuple[str, dict[str, float], float]:
        h = self._hidden(query, self.layer)
        proba7 = torch.exp(self._log_probs(self.head7, h, "coarse"))[0]
        proba = {letter: float(p) for letter, p in zip(OPTION_LETTERS, proba7.tolist())}
        p_confused = float(torch.exp(self._log_probs(self.head_bin, h, "confused"))[0, 1])
        return max(OPTION_LETTERS, key=lambda letter: proba[letter]), proba, p_confused

    @torch.no_grad()
    def predict_fine(self, query: str) -> tuple[str, dict[str, float]] | None:
        if self.head21 is None:
            return None
        h = self._hidden(query, self.layer)
        proba = torch.exp(self._log_probs(self.head21, h, "fine"))[0]
        by_fine = {fine: float(p) for fine, p in zip(FINE21_ORDER, proba.tolist())}
        return max(FINE21_ORDER, key=lambda fine: by_fine[fine]), by_fine

    @torch.no_grad()
    def predict_yes(self, query: str) -> tuple[float, dict[str, float]] | None:
        """P(out-of-scope) from the calibrated yes/no head.

        Class 0 is "in scope" and class 1 is "yes, out of scope". Returns
        ``None`` when the artifact predates the head.
        """
        if self.head_yes is None:
            return None
        h = self._hidden_prompt(chat_prompt(query), self._head_layer(self.head_yes))
        proba = torch.exp(self._log_probs(self.head_yes, h, "yes"))[0]
        by_class = {name: float(p) for name, p in zip(("in_scope", "out_of_scope"), proba.tolist())}
        return by_class["out_of_scope"], by_class

    @torch.no_grad()
    def predict_score(self, query: str, response: str) -> tuple[float, list[float]] | None:
        """Expected reply quality plus the level distribution.

        ``expected = sum(k * p_k)`` over the calibrated score levels stored in
        the artifact. Returns ``None`` when no score head is available.
        """
        if not self.has_score_head:
            return None
        h = self._hidden_prompt(score_prompt(query, response), self._head_layer(self.head_score))
        proba = torch.exp(self._log_probs(self.head_score, h, "score"))[0]
        levels = torch.tensor(self.score_levels, dtype=proba.dtype)
        expected = float((levels * proba).sum())
        return expected, proba.tolist()

    def load_tool_heads(self, tool_probe_path: str) -> bool:
        try:
            tool = torch.load(tool_probe_path, map_location=self.device, weights_only=True)
            verify_artifact(
                tool, self.checkpoint_path, self.tokenizer_path,
                dtype=str(next(self.model.parameters()).dtype),
            )
            selection = tool.get("selection")
            self.tool_layer = selection["layer"] if selection else None
            if selection:
                self.tool_head = FittedHead.from_dict(selection["heads"]["tool"]).to(self.device)
                self.tool_priors = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in tool.get("priors", {}).items()}
                self.tool_prior_lambda = float(tool.get("prior_lambda", 1.0))
            else:

                self.tool_head = LegacyHead(tool["head"], tool["temperature"], self.device)
                self.tool_priors = {}
                self.tool_prior_lambda = 1.0
            return True
        except Exception:  # noqa: BLE001 - missing file means no tool routing
            self.tool_head = None
            return False

    @torch.no_grad()
    def predict_tool(self, query: str) -> tuple[str, dict[str, float]] | None:
        if getattr(self, "tool_head", None) is None:
            return None
        h = self._hidden(query, self.tool_layer)
        lp = self.tool_head.log_probs(h.unsqueeze(0))
        if "tool" in getattr(self, "tool_priors", {}):
            lp = apply_prior(lp, self.tool_priors["tool"], self.tool_prior_lambda)
        proba = torch.exp(lp)[0]
        by_tool = {label: float(p) for label, p in zip(TOOL_CLASSES, proba.tolist())}
        return max(TOOL_CLASSES, key=lambda label: by_tool[label]), by_tool

    @staticmethod
    def coarse_of(fine_type: str) -> str:
        return FINE_TO_COARSE[fine_type]


def export_probe(probe_path: str, output_path: str) -> dict:
    """Write probe heads to JSON for the browser UI. Return the data."""
    from hermitlm.tools.heads import FittedHead

    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    data = {
        "options": OPTION_LETTERS,
        "names": {letter: COARSE_NAMES[letter] for letter in OPTION_LETTERS},
        "fine_options": FINE21_ORDER,
    }
    selection = probe.get("selection")
    if selection:
        data["head_layer"] = selection["layer"]
        names = {
            "coarse": ("head7", "temperature7"),
            "confused": ("head_bin", "temperature_bin"),
            "fine": ("head21", "temperature21"),
        }
        for key, (head_key, temp_key) in names.items():
            if key in selection["heads"]:
                h = FittedHead.from_dict(selection["heads"][key])
                w, b = h.effective()
                data[head_key] = {"weight": w.tolist(), "bias": b.tolist()}
                data[temp_key] = h.temperature
    else:
        data["temperature7"] = float(probe["temperature7"])
        data["temperature_bin"] = float(probe["temperature_bin"])
        data["head7"] = {
            "weight": probe["head7"]["weight"].tolist(),
            "bias": probe["head7"]["bias"].tolist(),
        }
        data["head_bin"] = {
            "weight": probe["head_bin"]["weight"].tolist(),
            "bias": probe["head_bin"]["bias"].tolist(),
        }
        if "head21" in probe:
            data["head21"] = {
                "weight": probe["head21"]["weight"].tolist(),
                "bias": probe["head21"]["bias"].tolist(),
            }
            data["temperature21"] = float(probe.get("temperature21", 1.0))
    # Optional yes/no and score heads. They may be stored in `selection` or at
    # top level for legacy artifacts; export both shapes as plain linear heads.
    for name, head_key, temp_key in (
        ("yes", "head_yes", "temperature_yes"),
        ("score", "head_score", "temperature_score"),
    ):
        state = (selection or {}).get("heads", {}).get(name)
        if state is None:
            state = probe.get(head_key)
        if state is None:
            continue
        if "mean" in state and "std" in state:
            fitted = FittedHead.from_dict(state)
            w, b = fitted.effective()
            temperature = fitted.temperature
        else:
            w, b = state["weight"], state["bias"]
            temperature = float(state.get("temperature", probe.get(temp_key, 1.0)))
        data[head_key] = {"weight": w.tolist(), "bias": b.tolist()}
        data[temp_key] = float(temperature)
    for key in ("score_levels", "score_thresholds", "score_labels"):
        if key in probe:
            data[key] = probe[key]
    # Split-conformal gates. Only store plain JSON numbers so the browser and
    # the API read exactly the same cutoffs (old artifacts simply omit them).
    for key in ("clarify_threshold", "escalation_threshold", "target_error", "threshold_abstention"):
        if key in probe:
            data[key] = float(probe[key])
    if "threshold_coverage" in probe:
        data["threshold_coverage"] = {
            str(name): float(value) for name, value in probe["threshold_coverage"].items()
        }
    # Priors are label-free and apply to both legacy and selected artifacts.
    # Export them even when empty so the browser sees the same payload the API
    # loads (intent.js checks probe.priors?.coarse).
    if "priors" in probe:
        data["priors"] = probe["priors"]
        data["prior_lambda"] = float(probe.get("prior_lambda", 1.0))
    for key in (
        "schema_version", "checkpoint_sha256", "tokenizer_sha256",
        "prompt_template_sha256", "dtype", "layer",
    ):
        if key in probe:
            data[key] = probe[key]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    print(f"Exported probe to {out} ({out.stat().st_size / 1024:.1f}KB)")
    return data

