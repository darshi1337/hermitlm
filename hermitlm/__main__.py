import os
import sys

CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"


def main():
    if len(sys.argv) < 2:
        print("HermitLM — A compact conversational model")
        print()
        print("Usage:")
        print("  python -m hermitlm train        Train the model")
        print("  python -m hermitlm prepare      Generate data & tokenizer")
        print("  python -m hermitlm chat         Chat in terminal")
        print("  python -m hermitlm bot          Run Discord bot")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        from hermitlm.prepare_data import prepare
        prepare()

    elif cmd == "train":
        from hermitlm.train import train
        train()

    elif cmd == "chat":
        if not os.path.exists(CHECKPOINT_PATH):
            print("Model not found. Train first:\n")
            print("  python -m hermitlm prepare")
            print("  python -m hermitlm train")
            return

        from hermitlm.inference import main as inference_main
        inference_main()

    elif cmd == "bot":
        from hermitlm.discord_bot import client
        from dotenv import load_dotenv

        load_dotenv()
        token = os.getenv("DISCORD_TOKEN")

        if not token:
            raise ValueError("DISCORD_TOKEN not found in .env")

        client.run(token)

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m hermitlm' for usage.")


if __name__ == "__main__":
    main()