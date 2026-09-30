"""Push checkpoint + ONNX + tokenizer to Hugging Face Hub."""
import argparse

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--onnx", default="web/model.onnx")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    a = p.parse_args()
    api = HfApi()
    api.create_repo(a.repo, exist_ok=True)
    for f in [a.checkpoint, a.onnx, a.tokenizer]:
        api.upload_file(path_or_fileobj=f, path_in_repo=f.split("/")[-1], repo_id=a.repo)
    print(f"Pushed to {a.repo}")
if __name__ == "__main__":
    main()
