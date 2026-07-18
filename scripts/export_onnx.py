import argparse
import shutil
from pathlib import Path

import torch

from hermitlm.config import HermitConfig
from hermitlm.training.model import HermitLM


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits, _ = self.model(input_ids)
        return logits


def export_onnx(checkpoint_path, tokenizer_path, output_dir, opset=17):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()

    model = HermitLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = LogitsOnly(model)
    dummy = torch.randint(0, config.vocab_size, (1, 8), dtype=torch.long)

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "logits": {1: "seq_len"}},
        opset_version=opset,
        dynamo=False,
    )

    shutil.copy(tokenizer_path, output_dir / "tokenizer.json")

    print(f"Exported model to {onnx_path}")
    print(f"Copied tokenizer to {output_dir / 'tokenizer.json'}")
    print(
        f"Config: pad_id={config.pad_id} bos_id={config.bos_id} eos_id={config.eos_id} "
        f"max_seq_len={config.max_seq_len} vocab_size={config.vocab_size}"
    )


def main():
    p = argparse.ArgumentParser(description="Export HermitLM checkpoint to ONNX for the web UI")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--output-dir", default="web")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    export_onnx(args.checkpoint, args.tokenizer, args.output_dir, args.opset)


if __name__ == "__main__":
    main()
