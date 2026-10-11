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

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.training.model import HermitLM

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

FINE_TO_COARSE: dict[str, str] = {
    fine: coarse for coarse, fines in COARSE_MEMBERS.items() for fine in fines
}

ALL_FINE_TYPES: list[str] = sorted(FINE_TO_COARSE)


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


def format_prompt(query: str) -> str:
    """Build the classification prompt. The model scores one letter next."""
    options = "\n".join(f"{letter}. {COARSE_NAMES[letter]}" for letter in OPTION_LETTERS)
    return (
        "<|im_start|>user\n"
        f"Classify the intent of this message: {query}\n"
        f"Options:\n{options}\n"
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
    def predict_proba(self, query: str, temperature: float = 1.0) -> dict[str, float]:
        ids = self.tokenizer.encode(format_prompt(query)).ids
        ids = ids[-self.config.max_seq_len :]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(idx)
        probs = constrained_probs(logits[0, -1], self.option_ids, temperature)
        return {letter: float(p) for letter, p in zip(OPTION_LETTERS, probs.tolist())}

    @torch.no_grad()
    def predict(self, query: str, temperature: float = 1.0) -> tuple[str, dict[str, float]]:
        proba = self.predict_proba(query, temperature)
        label = max(OPTION_LETTERS, key=lambda letter: proba[letter])
        return label, proba

def chat_prompt(query: str) -> str:
    """In-distribution chat prompt used for hidden-state probes."""
    return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


class ProbeDecisionModel:
    """Supervised System-1 head: frozen LM forward pass plus linear probe.

    Same single-pass cost profile as constrained decoding, but trained. It
    predicts the 7 coarse intents plus P(confused) as the out-of-scope signal
    (option E from the article).
    """

    def __init__(self, checkpoint_path: str, tokenizer_path: str, probe_path: str, device: str = "cpu"):
        from torch import nn

        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
        self.config.vocab_size = self.tokenizer.get_vocab_size()
        self.model = HermitLM(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        probe = torch.load(probe_path, map_location=self.device, weights_only=True)
        d_model = probe["d_model"]
        self.head7 = nn.Linear(d_model, len(OPTION_LETTERS)).to(self.device)
        self.head7.load_state_dict(probe["head7"])
        self.head7.eval()
        self.temperature7 = float(probe["temperature7"])
        self.head_bin = nn.Linear(d_model, 2).to(self.device)
        self.head_bin.load_state_dict(probe["head_bin"])
        self.head_bin.eval()
        self.temperature_bin = float(probe["temperature_bin"])

    @torch.no_grad()
    def _hidden(self, query: str) -> torch.Tensor:
        ids = self.tokenizer.encode(chat_prompt(query)).ids[-self.config.max_seq_len :]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        _, _, hidden = self.model(idx, return_hidden=True)
        return hidden[0, len(ids) - 1]

    @torch.no_grad()
    def predict(self, query: str) -> tuple[str, dict[str, float], float]:
        """Return (coarse_label, proba_over_7, p_confused)."""
        h = self._hidden(query)
        logits7 = self.head7(h.unsqueeze(0)) / self.temperature7
        proba7 = torch.softmax(logits7, dim=-1)[0]
        proba = {letter: float(p) for letter, p in zip(OPTION_LETTERS, proba7.tolist())}
        logits_bin = self.head_bin(h.unsqueeze(0)) / self.temperature_bin
        p_confused = float(torch.softmax(logits_bin, dim=-1)[0, 1])
        return max(OPTION_LETTERS, key=lambda letter: proba[letter]), proba, p_confused

    @staticmethod
    def coarse_of(fine_type: str) -> str:
        return FINE_TO_COARSE[fine_type]
