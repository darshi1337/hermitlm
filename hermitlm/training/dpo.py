"""DPO fine-tuning stage (simpler alternative to PPO)."""
import json
import os
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig
from hermitlm.training.model import HermitLM
from hermitlm.training.ppo import format_prompt
from hermitlm.training.reward import score_response
from hermitlm.training.train import sync_config_with_tokenizer


def build_pairs(prompts, tokenizer, policy, mc, device, n=4, temp=0.8, top_k=40, min_margin=0.15):
    pairs = []
    policy.eval()
    with torch.no_grad():
        for p in prompts:
            cands = []
            for _ in range(n):
                ids = tokenizer.encode(format_prompt(p)).ids[-mc.max_seq_len+64:]
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                out = policy.generate(idx, max_new_tokens=48, temperature=temp, top_k=top_k)
                txt = tokenizer.decode(out[0].tolist()[len(ids):]).strip()
                cands.append((txt, score_response(txt)))
            cands.sort(key=lambda c: c[1], reverse=True)
            (best_txt, best_s), (worst_txt, worst_s) = cands[0], cands[-1]
            if best_txt != worst_txt and best_s - worst_s >= min_margin:
                pairs.append({"prompt": p, "chosen": best_txt, "rejected": worst_txt})
    policy.train()
    return pairs

def _logps(model, ids):
    logits, _ = model(ids)
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

def dpo_loss(pi_chosen, pi_rej, ref_chosen, ref_rej, beta=0.1, mask_c=None, mask_r=None):
    def _m(lp, m):
        return (lp * m).sum(-1) / m.sum(-1).clamp(min=1) if m is not None else lp.mean(-1)
    r_pi = _m(pi_chosen, mask_c) - _m(pi_rej, mask_r)
    r_ref = _m(ref_chosen, mask_c) - _m(ref_rej, mask_r)
    return -F.logsigmoid(beta * (r_pi - r_ref)).mean()

def _encode(tokenizer, mc, device, prompt, completion):
    ids = tokenizer.encode(format_prompt(prompt) + completion).ids[-mc.max_seq_len :]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _pad_batch(tensors, pad_id):
    width = max(t.size(1) for t in tensors)
    return [F.pad(t, (0, width - t.size(1)), value=pad_id) for t in tensors]


def _resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_dpo(sft_checkpoint="checkpoints/best_model.pt", data_dir="data", out="checkpoints/dpo/dpo_final.pt", steps=200, beta=0.1, lr=5e-6, batch=4):
    mc = HermitConfig()
    mc.dropout = 0.0
    device = _resolve_device()
    sync_config_with_tokenizer(mc, os.path.join(data_dir, "tokenizer.json"))
    tok = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    policy = HermitLM(mc).to(device)
    ref = HermitLM(mc).to(device)
    ckpt = torch.load(sft_checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["model_state_dict"])
    ref.load_state_dict(ckpt["model_state_dict"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    prompts = []
    with open(os.path.join(data_dir, "train.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("input"): prompts.append(d["input"])
    for step in range(1, steps + 1):
        bp = random.sample(prompts, min(batch, len(prompts)))
        pairs = build_pairs(bp, tok, policy, mc, device)
        if step % 20 == 0:
            print(f"dpo step {step}: {len(pairs)}/{len(bp)} pairs kept (margin filter)")
        if not pairs:
            continue
        losses = []
        opt.zero_grad(set_to_none=True)
        for pr in pairs:
            cc = _encode(tok, mc, device, pr["prompt"], pr["chosen"])
            rj = _encode(tok, mc, device, pr["prompt"], pr["rejected"])
            cc, rj = _pad_batch([cc, rj], mc.pad_id)
            loss = dpo_loss(_logps(policy, cc), _logps(policy, rj), _logps(ref, cc), _logps(ref, rj), beta)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"dpo step {step} loss {loss.item():.4f}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model_state_dict": policy.state_dict(), "config": vars(mc)}, out)
    print(f"Saved {out}")
    return out

if __name__ == "__main__":
    run_dpo()
