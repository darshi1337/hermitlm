import torch
from tokenizers import Tokenizer

from .config import HermitConfig
from .model import HermitLM


class HermitInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if "config" in ckpt:
            self.config = HermitConfig(**ckpt["config"])
        else:
            print("Warning: using default config")
            self.config = HermitConfig()

        self.config.vocab_size = self.tokenizer.get_vocab_size()

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

        text = self.tokenizer.decode(generated_ids)

        if "<|im_end|>" in text:
            text = text.split("<|im_end|>")[0]

        if "<|im_start|>" in text:
            text = text.split("<|im_start|>")[0]

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