"""Export probe decision-head weights to web/probe.json for in-browser intent.

Reads checkpoints/decision_probe.pt (see scripts/train_probe.py). It writes a small
JSON (~61KB) with the 7-way and confused-vs-rest linear heads and their
temperatures. web/intent.js applies them to the last_hidden output of the model.
Run on every push via .github/workflows/deploy.yml.
"""

import argparse
import json
from pathlib import Path

import torch

from hermitlm.tools.decision_model import COARSE_NAMES, OPTION_LETTERS


def export_probe(probe_path: str, output_path: str) -> dict:
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    data = {
        "options": OPTION_LETTERS,
        "names": {letter: COARSE_NAMES[letter] for letter in OPTION_LETTERS},
        "temperature7": float(probe["temperature7"]),
        "temperature_bin": float(probe["temperature_bin"]),
        "head7": {
            "weight": probe["head7"]["weight"].tolist(),
            "bias": probe["head7"]["bias"].tolist(),
        },
        "head_bin": {
            "weight": probe["head_bin"]["weight"].tolist(),
            "bias": probe["head_bin"]["bias"].tolist(),
        },
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    print(f"Exported probe to {out} ({out.stat().st_size / 1024:.1f}KB)")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Export probe head to JSON for the web UI")
    p.add_argument("--probe", default="checkpoints/decision_probe.pt")
    p.add_argument("--output", default="web/probe.json")
    args = p.parse_args()
    export_probe(args.probe, args.output)


if __name__ == "__main__":
    main()
