#!/usr/bin/env bash
# Mirror CI: pip install -r requirements.txt pytest ruff && pip install -e .
set -euo pipefail

# CPU wheels keep the devcontainer small. The container has no GPU passthrough.
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt pytest ruff
pip install --no-cache-dir -e .
