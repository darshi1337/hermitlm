"""Fetch vendored onnxruntime-web assets into web/vendor/.

The vendor dir is git-ignored (12MB of binaries). Run once for local dev;
deploy CI runs it on every build. Files come from jsDelivr, pinned by
version and SHA-256.

Usage:
    python scripts/fetch_vendor.py [--output-dir web/vendor]
"""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

VERSION = "1.19.2"
BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{VERSION}/dist"
FILES = {
    "ort.min.js": "6c5e8696a7993c8af6bbf7666a8acf7f3179165939d9dba459506a59c178061a",
    "ort-wasm-simd-threaded.mjs": "d870a377322c3053fb97432d548423f165dd15e2af232947592fc07b0d2f3639",
    "ort-wasm-simd-threaded.wasm": "1bf0b9ed7ad025cf9ca88ce6da29e54df3f128a169f8241d71823e81f078d578",
}


def fetch(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        dest = out / name
        with urlopen(f"{BASE}/{name}", timeout=300) as res:
            data = res.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha:
            raise ValueError(f"SHA mismatch for {name}: {digest}")
        dest.write_bytes(data)
        print(f"{name} ({len(data) / 1e6:.1f}MB) SHA ok")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch vendored onnxruntime-web assets")
    p.add_argument("--output-dir", default="web/vendor")
    args = p.parse_args()
    fetch(args.output_dir)


if __name__ == "__main__":
    main()
