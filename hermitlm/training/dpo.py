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
    """Preference loss.

    ``mask_*`` is a float mask over *target* positions, i.e. it has the same
    shape as the log-prob tensors returned by :func:`_logps`. Only completion
    tokens (plus the end-of-turn token) should be set to 1; prompt and padding
    positions must be 0, otherwise the two sequence means use different
    denominators and the preference margin is polluted by pad log-probs.
    """
    def _m(lp, m):
        if m is None:
            return lp.mean(-1)
        denom = m.sum(-1).clamp(min=1.0)
        return (lp * m).sum(-1) / denom

    r_pi = _m(pi_chosen, mask_c) - _m(pi_rej, mask_r)
    r_ref = _m(ref_chosen, mask_c) - _m(ref_rej, mask_r)
    return -F.logsigmoid(beta * (r_pi - r_ref)).mean()


def _encode_pair(tokenizer, mc, device, prompt, chosen, rejected):
    """Encode a chosen/rejected pair with completion-only target masks.

    Prompt and completion are tokenized separately so that:
      * left-truncation never eats the beginning of the user turn;
      * an explicit end-of-turn token terminates both completions;
      * both sequences share exactly the same prompt tokens;
      * the returned mask covers only completion targets (after the shift
        performed by :func:`_logps`).
    """
    prompt_ids = tokenizer.encode(format_prompt(prompt)).ids
    eos_id = mc.eos_id
    c_ids = tokenizer.encode(chosen).ids + [eos_id]
    r_ids = tokenizer.encode(rejected).ids + [eos_id]

    # Keep at least one prompt token and fit the longer completion in the
    # model's positional embedding window.
    max_completion = max(1, mc.max_seq_len - 2)
    if max(len(c_ids), len(r_ids)) > max_completion:
        c_ids = c_ids[:max_completion]
        r_ids = r_ids[:max_completion]
    budget = max(1, mc.max_seq_len - max(len(c_ids), len(r_ids)))
    prompt_ids = prompt_ids[-budget:]

    seqs = [prompt_ids + c_ids, prompt_ids + r_ids]
    width = max(len(seq) for seq in seqs)
    input_ids = torch.full((2, width), mc.pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((2, max(1, width - 1)), dtype=torch.float32, device=device)
    for i, seq in enumerate(seqs):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        start = len(prompt_ids) - 1  # position predicting the first completion token
        end = len(seq) - 1           # one past the last target position
        if end > start:
            mask[i, start:end] = 1.0
    return input_ids, mask


def _resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_dpo(sft_checkpoint="checkpoints/best_model.pt", data_dir="data", out="checkpoints/dpo/dpo_final.pt", steps=200, beta=0.1, lr=5e-6, batch=4, device=None):
    if device is None:
        device = _resolve_device()
    else:
        from hermitlm.training.train import accelerator_available

        device = torch.device(device)
        if device.type in ("cuda", "xpu") and not accelerator_available(device.type):
            raise RuntimeError(
                f"DPO was requested on {device.type} but that backend is not "
                "available. Install the matching PyTorch build and driver, or "
                "set HERMIT_TRAIN_DEVICE=cpu for a CPU-only debug run."
            )
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    tok = Tokenizer.from_file(tokenizer_path)

    # Load the reference/SFT state first so the architecture comes from the
    # checkpoint, not from HermitConfig() defaults. torch.load() fails loudly
    # when the checkpoint is missing -- silently training from random weights
    # would make the DPO reference meaningless.
    ckpt = torch.load(sft_checkpoint, map_location=device, weights_only=True)
    mc = HermitConfig(**ckpt["config"]) if "config" in ckpt else HermitConfig()
    mc.dropout = 0.0
    sync_config_with_tokenizer(mc, tokenizer_path)

    policy = HermitLM(mc).to(device)
    ref = HermitLM(mc).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    ref.load_state_dict(ckpt["model_state_dict"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    prompts = []
    with open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("input"):
                prompts.append(d["input"])
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
            ids, mask = _encode_pair(tok, mc, device, pr["prompt"], pr["chosen"], pr["rejected"])
            pi = _logps(policy, ids)
            rj = _logps(ref, ids)
            loss = dpo_loss(pi[0], pi[1], rj[0], rj[1], beta, mask[0], mask[1])
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
    from hermitlm.settings import TRAIN_DEVICE

    run_dpo(device=TRAIN_DEVICE)
