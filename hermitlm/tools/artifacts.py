"""Versioned, hash-checked decision-head artifacts.

Artifacts are plain ``torch.save`` dicts. New artifacts record a schema
version plus SHA-256 hashes of the base checkpoint and tokenizer. ``load``
only verifies hashes that are present, so older artifacts keep working.
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
    """Raise :class:`ArtifactMismatchError` on a stale artifact.

    Artifacts without hashes (schema 1) are accepted unchanged.
    """
    expected_checkpoint = artifact.get("checkpoint_sha256")
    if not expected_checkpoint:
        return
    if file_sha256(checkpoint_path) != expected_checkpoint:
        raise ArtifactMismatchError(
            "decision head was fitted for a different checkpoint; retrain the probe"
        )
    expected_tokenizer = artifact.get("tokenizer_sha256")
    if expected_tokenizer and file_sha256(tokenizer_path) != expected_tokenizer:
        raise ArtifactMismatchError(
            "decision head was fitted for a different tokenizer; retrain the probe"
        )
