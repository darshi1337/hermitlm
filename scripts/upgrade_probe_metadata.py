"""Add schema-3 fit metadata to existing probe artifacts without refitting.

Only descriptive keys are added: ``chosen_layer``, ``thresholds`` and
``refit``. Head weights, priors and all SHA-256 pins stay untouched.
"""
import argparse

import torch

from hermitlm.settings import PROBE_PATH, TOOL_PROBE_PATH, TOOL_THRESHOLD
from hermitlm.tools.artifacts import SCHEMA_VERSION, refit_metadata
from hermitlm.tools.decision_model import (
    LEGACY_CLARIFY_THRESHOLD,
    LEGACY_ESCALATION_THRESHOLD,
)


def _layer(probe: dict, selection: dict | None) -> int:
    if "chosen_layer" in probe and probe["chosen_layer"] is not None:
        return int(probe["chosen_layer"])
    if selection:
        return int(selection["layer"])
    if "layer" in probe and probe["layer"] is not None:
        return int(probe["layer"])
    return -1  # final hidden state


def upgrade(path: str, kind: str) -> None:
    probe = torch.load(path, map_location="cpu", weights_only=True)
    selection = probe.get("selection")
    layer = _layer(probe, selection)
    thresholds = dict(probe.get("thresholds") or {})

    if kind == "tool":
        thresholds.setdefault("tool", float(TOOL_THRESHOLD))
        if selection:
            states = {"tool": selection["heads"]["tool"]}
        else:
            states = {"tool": {"temperature": probe.get("temperature", 1.0)}}
    else:
        thresholds.setdefault(
            "clarify", float(probe.get("clarify_threshold", LEGACY_CLARIFY_THRESHOLD))
        )
        thresholds.setdefault(
            "escalation", float(probe.get("escalation_threshold", LEGACY_ESCALATION_THRESHOLD))
        )
        thresholds.setdefault("target_error", float(probe.get("target_error", 0.1)))
        if selection:
            states = dict(selection["heads"])
        else:
            states = {
                "coarse": {"temperature": probe.get("temperature7", 1.0)},
                "confused": {"temperature": probe.get("temperature_bin", 1.0)},
                "fine": {"temperature": probe.get("temperature21", 1.0)},
            }
            for name in ("yes", "score"):
                if f"head_{name}" in probe:
                    states[name] = probe[f"head_{name}"]

    probe["chosen_layer"] = layer
    probe["thresholds"] = thresholds
    probe.setdefault("refit", refit_metadata(states, folds=5, seed=42))
    probe["schema_version"] = SCHEMA_VERSION
    torch.save(probe, path)
    print(f"upgraded {path} to schema {SCHEMA_VERSION} (layer={layer})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade probe artifacts to schema 3")
    parser.add_argument("--probe", default=PROBE_PATH)
    parser.add_argument("--tool-probe", default=TOOL_PROBE_PATH)
    parser.add_argument("--skip-tool", action="store_true")
    args = parser.parse_args()
    upgrade(args.probe, "intent")
    if not args.skip_tool:
        upgrade(args.tool_probe, "tool")


if __name__ == "__main__":
    main()
