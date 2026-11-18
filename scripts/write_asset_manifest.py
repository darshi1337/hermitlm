"""Write a small asset manifest so the browser cache can be invalidated.

The client-cache key includes the quantized model hash from this file. A new
deploy with new weights therefore never serves stale cached weights paired
with a fresh probe/tokenizer.
"""
import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_dir: str = "web") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = ["model.quant.onnx", "tokenizer.json", "probe.json"]
    manifest = {
        name: {"sha256": _sha256(out / name), "bytes": (out / name).stat().st_size}
        for name in files
        if (out / name).exists()
    }
    path = out / "asset-manifest.json"
    path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {path} ({len(manifest)} assets)")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="web")
    write_manifest(parser.parse_args().output_dir)
