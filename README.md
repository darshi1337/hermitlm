<h1 align="center">HermitLM</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/darshi1337/hermitlm-100k-generic">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-hermitlm--100k-4b5563">
  </a>
  <a href="https://huggingface.co/darshi1337/hermitlm-8M">
    <img alt="Model" src="https://img.shields.io/badge/Model-hermitlm--8M-b65f36">
  </a>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-3fa34d">
  <a href="https://hub.xerj.org/contributors/darshi1337">
    <img alt="XERJ Corpus Hub contributor" src="https://hub.xerj.org/contributors/darshi1337/badge.svg">
  </a>
</p>

<p align="center">
  <img alt="HermitLM" src="assets/hermit_1.png">
</p>

<p align="center">
  <a href="assets/video.mp4">
    <img alt="Try Demo" src="https://img.shields.io/badge/TRY%20DEMO-VIDEO-56f5c6">
  </a>
</p>

A ~8M parameter LLM that talks like a tiny crab in an aquatic world, chattable entirely in your browser.

## Documentation

The README is an index. Each topic has its own file:

| # | Topic | File |
|---|---|---|
| 01 | Introduction, transcript and how it works | [docs/01-introduction.md](docs/01-introduction.md) |
| 02 | Architecture and parameter count | [docs/02-architecture.md](docs/02-architecture.md) |
| 03 | Features | [docs/03-features.md](docs/03-features.md) |
| 04 | Training (SFT, PPO, DPO) | [docs/04-training.md](docs/04-training.md) |
| 05 | Evaluation | [docs/05-eval.md](docs/05-eval.md) |
| 06 | Tool router | [docs/06-tools.md](docs/06-tools.md) |
| 07 | Decision model and metrics | [docs/07-decision-model.md](docs/07-decision-model.md) |
| 08 | Calibration toolkit | [docs/08-calibration.md](docs/08-calibration.md) |
| 09 | Tool routing details | [docs/09-tool-routing.md](docs/09-tool-routing.md) |
| 10 | Reply quality and analytics | [docs/10-reply-quality.md](docs/10-reply-quality.md) |
| 11 | HTTP API | [docs/11-api.md](docs/11-api.md) |
| 12 | Browser website | [docs/12-website.md](docs/12-website.md) |
| 13 | Hugging Face repos | [docs/13-hugging-face.md](docs/13-hugging-face.md) |
| 14 | Development and tests | [docs/14-development.md](docs/14-development.md) |
| 15 | License | [docs/15-license.md](docs/15-license.md) |

## Quick links

- [15-chapter textbook](chapters/README.md), starting with [the overview](chapters/00_overview.md).
- [Formula and hyperparameter sheet](chapters/14_glossary_math.md).
- [Calibration work status](docs/calibration-improvements.md).
- [Pre-deploy diagnostics](docs/14-development.md) — `python -m hermitlm.tools.check`.
