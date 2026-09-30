"""Eval dashboard: perplexity + persona reward + tool-router accuracy."""
import argparse
import json
import math
import os

import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.training.dataset import get_dataloader
from hermitlm.training.model import HermitLM
from hermitlm.training.reward import score_response
from hermitlm.training.train import compute_loss


def _offline_tool_trigger(query):
    """Keyword-only routing check — no network calls, deterministic."""
    from hermitlm.tools.router import TOOL_TAG_RE

    if TOOL_TAG_RE.search(query or ""):
        return True
    q = query.lower()
    math_hit = any(k in q for k in ("solve", "integrate", "derivative", "equation", "factor"))
    web_hit = any(k in q for k in ("latest", "news", "today", "who is", "search"))
    return math_hit or web_hit


def run_eval(checkpoint="checkpoints/best_model.pt", data_dir="data", out_dir="eval_out"):
    os.makedirs(out_dir, exist_ok=True)
    Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mc = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
    model = HermitLM(mc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loader, _dataset = get_dataloader(os.path.join(data_dir, "eval.jsonl"), os.path.join(data_dir, "tokenizer.json"), mc.max_seq_len, 4, shuffle=False)
    tot, n = 0, 0
    with torch.no_grad():
        for x, y, mask in loader:
            logits, _ = model(x)
            tot += compute_loss(logits, y, mask).item()
            n += 1
            if n >= 50:
                break
    ppl = math.exp(tot / max(1, n))
    # persona reward on MODEL generations (not reference texts, which can't
    # discriminate between checkpoints)
    from hermitlm.training.ppo import format_prompt

    tokenizer = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    prompts = []
    with open(os.path.join(data_dir, "eval.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("input"):
                prompts.append(d["input"])
            if len(prompts) >= 10:
                break
    rewards = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer.encode(format_prompt(prompt)).ids[-mc.max_seq_len + 64 :]
            idx = torch.tensor([ids], dtype=torch.long)
            out = model.generate(idx, max_new_tokens=48, temperature=0.6, top_k=20)
            gen = tokenizer.decode(out[0].tolist()[len(ids):]).strip()
            rewards.append(score_response(gen))
    avg_r = sum(rewards) / max(1, len(rewards))
    # offline tool-router checks (no network)
    checks = ["integrate x^2", "latest AI news", "hello crab"]
    tool_ok = sum(1 for q in checks if _offline_tool_trigger(q))
    report = {"perplexity_proxy": round(ppl, 3), "avg_persona_reward": round(avg_r, 3), "tool_trigger_rate": f"{tool_ok}/{len(checks)}"}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = f"# HermitLM Eval\n\n- Perplexity: {report['perplexity_proxy']}\n- Persona reward: {report['avg_persona_reward']}\n- Tool triggers: {report['tool_trigger_rate']}\n"
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(md)
    print(md)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HermitLM eval dashboard")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="eval_out")
    args = parser.parse_args()
    run_eval(args.checkpoint, args.data_dir, args.out_dir)
