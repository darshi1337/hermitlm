"""Versioned, hash-checked decision-head artifacts.

Artifacts are plain ``torch.save`` dicts. Every head records a schema version
plus SHA-256 hashes of the base checkpoint and tokenizer. Loading rejects
artifacts without hashes: silently scoring with a head fitted for a different
checkpoint is worse than failing loudly and retraining.
"""

import hashlib
from pathlib import Path

SCHEMA_VERSION = 2


class ArtifactMismatchError(ValueError):
    """Raised when an artifact does not belong to the loaded model/tokenizer."""


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_metadata(checkpoint_path: str | Path, tokenizer_path: str | Path) -> dict:
    """Return the version/hash fields to store next to a fitted head."""
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
    }


def verify_artifact(
    artifact: dict, checkpoint_path: str | Path, tokenizer_path: str | Path
) -> None:
    """Raise :class:`ArtifactMismatchError` on stale or unversioned artifacts.

    Both hashes must be present. A head without them cannot be tied to the
    checkpoint/tokenizer it was fitted against, so it is rejected rather than
    silently accepted.
    """
    expected_checkpoint = artifact.get("checkpoint_sha256")
    expected_tokenizer = artifact.get("tokenizer_sha256")
    if not expected_checkpoint or not expected_tokenizer:
        raise ArtifactMismatchError(
            "decision head is missing schema_version/checkpoint_sha256/"
            "tokenizer_sha256; retrain the probe with the current scripts"
        )
    if file_sha256(checkpoint_path) != expected_checkpoint:
        raise ArtifactMismatchError(
            "decision head was fitted for a different checkpoint; retrain the probe"
        )
    if file_sha256(tokenizer_path) != expected_tokenizer:
        raise ArtifactMismatchError(
            "decision head was fitted for a different tokenizer; retrain the probe"
        )
