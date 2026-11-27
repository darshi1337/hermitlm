"""Versioned, hash-checked decision-head artifacts.

Artifacts are plain ``torch.save`` dicts. Every head records a schema version
plus SHA-256 hashes of the base checkpoint and tokenizer. Loading rejects
artifacts without hashes: silently scoring with a head fitted for a different
checkpoint is worse than failing loudly and retraining.
"""

import hashlib
import logging
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SCHEMA_VERSION = 3
# Schema 3 adds the fit context that schema 2 artifacts omit. Older artifacts
# stay readable; a v3 artifact must carry all of these fields.
V3_REQUIRED_FIELDS = ("chosen_layer", "thresholds", "refit")


class ArtifactMismatchError(ValueError):
    """Raised when an artifact does not belong to the loaded model/tokenizer."""


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_metadata(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    *,
    prompt_template=None,
    dtype=None,
    layer=None,
) -> dict:
    """Return version/hash fields to store next to a fitted head."""
    import hashlib

    meta = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
    }
    if prompt_template is not None:
        meta["prompt_template_sha256"] = hashlib.sha256(
            prompt_template.encode("utf-8")
        ).hexdigest()
    if dtype is not None:
        meta["dtype"] = str(dtype)
    if layer is not None:
        meta["layer"] = int(layer)
    return meta


def refit_metadata(heads: dict, *, folds: int, seed: int) -> dict:
    """Per-head solver/alpha/temperature plus the shared fold count and seed."""
    fitted = {}
    for name, state in heads.items():
        if not isinstance(state, dict):
            # Legacy linear heads have no solver choice recorded.
            fitted[name] = {
                "solver": "legacy",
                "alpha": None,
                "temperature": 1.0,
                "oof_nll": 0.0,
            }
            continue
        alpha = state.get("alpha")
        fitted[name] = {
            "solver": str(state.get("solver", "legacy")),
            "alpha": None if alpha is None else float(alpha),
            "temperature": float(state.get("temperature", 1.0)),
            "oof_nll": float(state.get("oof_nll", 0.0)),
        }
    return {"folds": int(folds), "seed": int(seed), "heads": fitted}


def head_metadata(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    *,
    chosen_layer: int,
    thresholds: dict,
    refit: dict,
    prompt_template=None,
    dtype=None,
) -> dict:
    """Version/hash fields plus the fit context required by schema 3."""
    meta = artifact_metadata(
        checkpoint_path, tokenizer_path,
        prompt_template=prompt_template, dtype=dtype, layer=chosen_layer,
    )
    meta["chosen_layer"] = int(chosen_layer)
    meta["thresholds"] = {str(k): float(v) for k, v in thresholds.items()}
    meta["refit"] = refit
    return meta


def verify_artifact(
    artifact: dict,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    *,
    dtype=None,
    layer=None,
) -> None:
    """Raise :class:`ArtifactMismatchError` on stale or unversioned artifacts.

    Both hashes must be present. A head without them cannot be tied to the
    checkpoint/tokenizer it was fitted against, so it is rejected rather than
    silently accepted.
    """
    expected_checkpoint = artifact.get("checkpoint_sha256")
    expected_tokenizer = artifact.get("tokenizer_sha256")
    if not expected_checkpoint or not expected_tokenizer:
        warnings.warn(
            "decision head is missing checkpoint/tokenizer hash pins; "
            "refusing to load it",
            UserWarning, stacklevel=2,
        )
        raise ArtifactMismatchError(
            "decision head is missing schema_version/checkpoint_sha256/"
            "tokenizer_sha256; retrain the probe with the current scripts"
        )
    version = int(artifact.get("schema_version", 1))
    logger.info(
        "verifying artifact schema=%s checkpoint_sha=%s tokenizer_sha=%s",
        version, str(expected_checkpoint)[:12], str(expected_tokenizer)[:12],
    )
    if version >= 3:
        missing = [key for key in V3_REQUIRED_FIELDS if key not in artifact]
        if missing:
            raise ArtifactMismatchError(
                "decision head is schema 3 but is missing " + ", ".join(missing)
            )
        thresholds = artifact["thresholds"]
        refit = artifact["refit"]
        if not isinstance(thresholds, dict) or not thresholds:
            raise ArtifactMismatchError("decision head has no fitted thresholds")
        if not isinstance(refit, dict) or "heads" not in refit:
            raise ArtifactMismatchError("decision head has no refit metadata")
        chosen = artifact.get("chosen_layer")
        if chosen is None:
            raise ArtifactMismatchError("decision head has no chosen_layer")
        if layer is not None and int(chosen) != int(layer):
            raise ArtifactMismatchError("decision head records a different chosen layer")

    if file_sha256(checkpoint_path) != expected_checkpoint:
        raise ArtifactMismatchError(
            "decision head was fitted for a different checkpoint; retrain the probe"
        )
    if file_sha256(tokenizer_path) != expected_tokenizer:
        raise ArtifactMismatchError(
            "decision head was fitted for a different tokenizer; retrain the probe"
        )
    import hashlib

    expected_prompt = artifact.get("prompt_template_sha256")
    if not expected_prompt:
        warnings.warn(
            "decision head has no prompt-template hash pin; prompt changes "
            "will not be detected",
            UserWarning, stacklevel=2,
        )
    if expected_prompt:
        from hermitlm.tools.decision_model import PROMPT_TEMPLATES

        actual = hashlib.sha256(PROMPT_TEMPLATES.encode("utf-8")).hexdigest()
        if actual != expected_prompt:
            raise ArtifactMismatchError(
                "prompt template changed since the head was fitted; retrain the probe"
            )
    if dtype is not None and "dtype" in artifact and str(artifact["dtype"]) != str(dtype):
        raise ArtifactMismatchError("decision head was fitted with a different dtype")
    if layer is not None and "layer" in artifact and int(artifact["layer"]) != int(layer):
        raise ArtifactMismatchError("decision head was fitted for a different hidden layer")
