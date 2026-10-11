"""Export probe decision-head weights to web/probe.json for in-browser intent.

Reads checkpoints/decision_probe.pt (see scripts/train_probe.py). It writes a small
JSON (~61KB) with the 7-way and confused-vs-rest linear heads and their
temperatures. web/intent.js applies them to the last_hidden output of the model.
Run on every push via .github/workflows/deploy.yml.
"""

import argparse

from hermitlm.tools.decision_model import export_probe


def main() -> None:
    p = argparse.ArgumentParser(description="Export probe head to JSON for the web UI")
    p.add_argument("--probe", default="checkpoints/decision_probe.pt")
    p.add_argument("--output", default="web/probe.json")
    args = p.parse_args()
    export_probe(args.probe, args.output)


if __name__ == "__main__":
    main()
