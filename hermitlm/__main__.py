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
        print("  python -m hermitlm api          Run FastAPI server")
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
        from hermitlm.discord_bot import run_bot

        run_bot()

    elif cmd == "api":
        import uvicorn

        uvicorn.run(
            "hermitlm.api:app",
            host=os.getenv("HERMIT_API_HOST", "0.0.0.0"),
            port=int(os.getenv("HERMIT_API_PORT", "8000")),
            reload=os.getenv("HERMIT_API_RELOAD", "0") == "1",
        )

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m hermitlm' for usage.")


if __name__ == "__main__":
    main()
