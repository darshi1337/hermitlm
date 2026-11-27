import logging
import warnings

import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.tools.decision_model import escape_delimiters
from hermitlm.training.model import HermitLM

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class HermitInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu", probe=None):
        self.device = torch.device(device)
        self.probe = probe  # optional ProbeDecisionModel with a score head
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        if "config" in ckpt:
            self.config = HermitConfig(**ckpt["config"])
        else:
            warnings.warn(
                "checkpoint has no stored config; using HermitConfig defaults",
                UserWarning, stacklevel=2,
            )
            logger.info("checkpoint config missing; using defaults")
            self.config = HermitConfig()

        tokenizer_vocab_size = self.tokenizer.get_vocab_size()
        if self.config.vocab_size != tokenizer_vocab_size:
            warnings.warn(
                f"checkpoint vocab_size={self.config.vocab_size} differs from "
                f"tokenizer vocab_size={tokenizer_vocab_size}; using tokenizer size",
                UserWarning, stacklevel=2,
            )
            logger.info(
                "vocab mismatch: checkpoint=%d tokenizer=%d",
                self.config.vocab_size, tokenizer_vocab_size,
            )
        self.config.vocab_size = tokenizer_vocab_size

        self.model = HermitLM(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        logger.info("HermitLM loaded: %s", self.model.param_summary())

    def chat(self, user_input, temperature=0.8, top_k=40, max_tokens=64):
        prompt = self._format_prompt(user_input)

        full_ids = self.tokenizer.encode(prompt).ids
        input_ids = full_ids[-self.config.max_seq_len :]
        logger.info(
            "chat prompt tokens=%d truncated=%d max_new=%d temperature=%.3f top_k=%d",
            len(full_ids), len(full_ids) - len(input_ids), max_tokens, temperature, top_k,
        )
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_t = self.model.generate(
            input_t,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        generated_ids = output_t[0].tolist()[len(input_ids):]

        # Stop at the first end-of-turn or new-turn marker. decode() strips
        # special tokens from the text, so a text-level split can't catch
        # them if the model emits a fresh turn instead of hitting eos.
        for i, token_id in enumerate(generated_ids):
            if token_id in (self.config.eos_id, self.config.bos_id):
                generated_ids = generated_ids[:i]
                break

        text = self.tokenizer.decode(generated_ids)

        return text.strip()

    def chat_candidates(self, user_input, n, temperature=0.8, top_k=40, max_tokens=64):
        """Generate n independent candidates with the same logic as chat()."""
        return [self.chat(user_input, temperature, top_k, max_tokens) for _ in range(max(1, n))]

    def best_of_n(self, user_input, n=3, temperature=0.8, top_k=40, max_tokens=64, score_query=None):
        """Generate n candidates and keep the best one.

        When the loaded probe carries a calibrated score head, candidates are
        ranked by ``expected = sum(k * p_k)``. Otherwise the heuristic persona
        reward is used. Returns ``(best_text, heuristic_scores, expected)``;
        ``expected`` is None when neither ranker is available.
        """
        from hermitlm.training.reward import score_response

        candidates = self.chat_candidates(user_input, n, temperature, top_k, max_tokens)
        scores = [score_response(text) for text in candidates]
        if len(candidates) == 1:
            return candidates[0], [0.0], None

        ranker = None
        if getattr(self, "probe", None) is not None:
            query = user_input if score_query is None else score_query
            try:
                ranked = [self.probe.predict_score(query, text) for text in candidates]
                if all(value is not None for value in ranked):
                    ranker = [value[0] for value in ranked]
            except Exception:  # noqa: BLE001 - fall back to the heuristic
                ranker = None
        if ranker is None:
            best = max(range(len(candidates)), key=lambda i: scores[i])
            return candidates[best], scores, None
        best = max(range(len(candidates)), key=lambda i: ranker[i])
        return candidates[best], scores, ranker[best]

    def _format_prompt(self, user_input):
        return (
            "<|im_start|>user\n"
            f"{escape_delimiters(user_input)}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

def main():
    import argparse

    p = argparse.ArgumentParser(description="Chat with HermitLM")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    engine = HermitInference(args.checkpoint, args.tokenizer, args.device)

    print("\n🦀 Hermit Chat (type 'quit' to exit)\n")

    while True:
        user = input("You > ").strip()

        if user.lower() in ["quit", "exit", "q"]:
            break

        response = engine.chat(user)

        print(f"Hermit > {response}")

if __name__ == "__main__":
    main()