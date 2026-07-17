#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

git pull

source venv/bin/activate
pip install -r requirements.txt

sudo systemctl restart hermitlm-api
sudo systemctl restart hermitlm-discord
