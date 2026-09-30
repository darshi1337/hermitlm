import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.training.model import HermitLM


class HermitInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        if "config" in ckpt:
            self.config = HermitConfig(**ckpt["config"])
        else:
            print("Warning: using default config")
            self.config = HermitConfig()

        tokenizer_vocab_size = self.tokenizer.get_vocab_size()
        if self.config.vocab_size != tokenizer_vocab_size:
            print(
                f"Warning: checkpoint vocab_size={self.config.vocab_size} differs from "
                f"tokenizer vocab_size={tokenizer_vocab_size}; using tokenizer size "
                "(load_state_dict will fail below if the checkpoint doesn't match)"
            )
        self.config.vocab_size = tokenizer_vocab_size

        self.model = HermitLM(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"HermitLM loaded: {self.model.param_summary()}")

    def chat(self, user_input, temperature=0.8, top_k=40, max_tokens=64):
        prompt = self._format_prompt(user_input)

        input_ids = self.tokenizer.encode(prompt).ids
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

    def _format_prompt(self, user_input):
        return (
            "<|im_start|>user\n"
            f"{user_input}<|im_end|>\n"
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