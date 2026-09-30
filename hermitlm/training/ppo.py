import json
import os
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn

from hermitlm.config import HermitConfig, PPOConfig
from hermitlm.training.model import HermitLM
from hermitlm.training.reward import score_response
from hermitlm.training.train import get_device, sync_config_with_tokenizer


class ActorCriticHermitLM(HermitLM):
    """HermitLM policy with a scalar value head for PPO advantage estimation."""

    def __init__(self, config):
        super().__init__(config)
        self.value_head = nn.Linear(config.d_model, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward_with_value(self, idx):
        logits, _, hidden = super().forward(idx, targets=None, return_hidden=True)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


def format_prompt(user_input):
    return (
        "<|im_start|>user\n"
        f"{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def logits_to_logprobs(logits, input_ids):
    logprobs_all = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_logprobs = logprobs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logprobs_all, token_logprobs


def load_prompts(data_dir):
    prompts = []

    path = os.path.join(data_dir, "train.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("input"):
                    prompts.append(data["input"])

    if not prompts:
        raise ValueError(
            f"No prompts found in {data_dir}/train.jsonl. Run prepare_data.py to "
            "generate it before PPO training. (eval.jsonl is intentionally excluded "
            "so it stays a held-out set.)"
        )

    return prompts


def load_actor_critic(mc, checkpoint_path, device):
    model = ActorCriticHermitLM(mc).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        extra_missing = [k for k in missing if not k.startswith("value_head")]
        if extra_missing or unexpected:
            raise ValueError(
                f"SFT checkpoint {checkpoint_path} doesn't match the model: "
                f"missing={extra_missing}, unexpected={unexpected}"
            )

    return model


def load_reference(mc, checkpoint_path, device):
    model = HermitLM(mc).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


@torch.no_grad()
def collect_rollouts(actor, tokenizer, prompts, mc, pc, device):
    actor.eval()
    rollouts = []

    for prompt in prompts:
        # Reserve room for generation so prompt + response never exceeds
        # max_seq_len (the model's position embeddings are only that wide).
        max_new = max(1, min(pc.max_new_tokens, mc.max_seq_len - 1))
        prompt_budget = max(1, mc.max_seq_len - max_new)

        prompt_ids = tokenizer.encode(format_prompt(prompt)).ids
        prompt_ids = prompt_ids[-prompt_budget:]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = actor.generate(
            idx,
            max_new_tokens=max_new,
            temperature=pc.temperature,
            top_k=pc.top_k,
        )

        response_ids = out[0].tolist()[len(prompt_ids):]

        # generate() already stops at eos_id, so response_ids only ends with
        # eos when termination was natural. But if the model instead emits a
        # fresh <|im_start|> turn marker without hitting eos, cut it there:
        # decode() strips special tokens from the text, so a text-level
        # split can't catch a leaked turn the way it can for plain words.
        for i, token_id in enumerate(response_ids):
            if token_id == mc.bos_id:
                response_ids = response_ids[:i]
                break

        if not response_ids:
            response_ids = [mc.eos_id]

        response_text = tokenizer.decode(response_ids).strip()

        rollouts.append({
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_text": response_text,
            "reward": score_response(response_text),
        })

    actor.train()
    return rollouts


@torch.no_grad()
def prepare_ppo_batch(actor, reference, rollouts, mc, device):
    pad_id = mc.pad_id
    seqs = [r["prompt_ids"] + r["response_ids"] for r in rollouts]
    max_len = max(len(s) for s in seqs)
    B = len(seqs)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    response_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)

    for i, (r, seq) in enumerate(zip(rollouts, seqs, strict=False)):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        start = len(r["prompt_ids"])
        response_mask[i, start:len(seq)] = 1.0

    actor.eval()
    logits, values = actor.forward_with_value(input_ids)
    _, old_logprobs = logits_to_logprobs(logits, input_ids)
    old_values = values[:, :-1]
    actor.train()

    ref_logits, _ = reference(input_ids)
    _, ref_logprobs = logits_to_logprobs(ref_logits, input_ids)

    shifted_mask = response_mask[:, 1:]
    heuristic_rewards = torch.tensor(
        [r["reward"] for r in rollouts], dtype=torch.float32, device=device
    )

    return input_ids, shifted_mask, old_logprobs, old_values, ref_logprobs, heuristic_rewards


def compute_advantages_and_returns(old_logprobs, ref_logprobs, old_values, shifted_mask, heuristic_rewards, pc):
    B = old_logprobs.size(0)
    advantages = torch.zeros_like(old_logprobs)
    returns = torch.zeros_like(old_logprobs)
    kl = old_logprobs - ref_logprobs

    for b in range(B):
        idxs = shifted_mask[b].nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue

        token_rewards = -pc.kl_coef * kl[b, idxs]
        token_rewards = token_rewards.clone()
        token_rewards[-1] = token_rewards[-1] + heuristic_rewards[b]

        vals = old_values[b, idxs]

        T = idxs.numel()
        adv = torch.zeros(T, device=old_logprobs.device)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_value = vals[t + 1] if t + 1 < T else 0.0
            delta = token_rewards[t] + pc.gamma * next_value - vals[t]
            last_gae = delta + pc.gamma * pc.lam * last_gae
            adv[t] = last_gae

        advantages[b, idxs] = adv
        returns[b, idxs] = adv + vals

    mask_bool = shifted_mask.bool()
    adv_values = advantages[mask_bool]

    if adv_values.numel() > 1:
        mean = adv_values.mean()
        std = adv_values.std().clamp(min=1e-6)
        advantages = torch.where(mask_bool, (advantages - mean) / std, advantages)

    return advantages, returns, kl


def ppo_update(actor, optimizer, input_ids, shifted_mask, old_logprobs, advantages, returns, pc):
    B = input_ids.size(0)
    indices = list(range(B))

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(pc.ppo_epochs):
        random.shuffle(indices)

        for start in range(0, B, pc.minibatch_size):
            mb_idx = indices[start:start + pc.minibatch_size]
            if not mb_idx:
                continue

            mb_input = input_ids[mb_idx]
            mb_mask = shifted_mask[mb_idx]
            mb_old_logprobs = old_logprobs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            logits, values = actor.forward_with_value(mb_input)
            logprobs_all, new_logprobs = logits_to_logprobs(logits, mb_input)
            new_values = values[:, :-1]

            entropy = -(logprobs_all.exp() * logprobs_all).sum(-1)

            denom = mb_mask.sum().clamp(min=1)

            ratio = torch.exp(new_logprobs - mb_old_logprobs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - pc.clip_epsilon, 1 + pc.clip_epsilon) * mb_advantages
            policy_loss = -(torch.min(surr1, surr2) * mb_mask).sum() / denom

            value_loss = ((new_values - mb_returns) ** 2 * mb_mask).sum() / denom
            entropy_bonus = (entropy * mb_mask).sum() / denom

            loss = policy_loss + pc.value_coef * value_loss - pc.entropy_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), pc.max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_bonus.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(1, n_updates),
        "value_loss": total_value_loss / max(1, n_updates),
        "entropy": total_entropy / max(1, n_updates),
    }


def extract_policy_state_dict(actor):
    return {k: v for k, v in actor.state_dict().items() if not k.startswith("value_head")}


def save_checkpoint(actor, mc, pc, iteration, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": iteration,
        "model_state_dict": extract_policy_state_dict(actor),
        "config": vars(mc),
        "ppo_config": vars(pc),
    }, path)


def run_ppo(mc=None, pc=None):
    mc = mc or HermitConfig()
    pc = pc or PPOConfig()

    # Dropout would make the "old" log-probs computed in prepare_ppo_batch
    # (eval mode) diverge from the ones recomputed during ppo_update (train
    # mode) even before any weight update, biasing the clipped objective.
    mc.dropout = 0.0

    device = get_device(pc)
    torch.manual_seed(pc.seed)
    random.seed(pc.seed)

    print(f"Device: {device}")

    tokenizer_path = os.path.join(pc.data_dir, "tokenizer.json")
    sync_config_with_tokenizer(mc, tokenizer_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)

    prompts = load_prompts(pc.data_dir)
    print(f"Loaded {len(prompts)} prompts for rollouts")

    actor = load_actor_critic(mc, pc.sft_checkpoint, device)
    reference = load_reference(mc, pc.sft_checkpoint, device)
    print(actor.param_summary())

    optimizer = torch.optim.AdamW(actor.parameters(), lr=pc.learning_rate)

    os.makedirs(pc.output_dir, exist_ok=True)

    history = {"reward": [], "kl": [], "policy_loss": [], "value_loss": []}

    print(f"\nPPO training for {pc.total_iterations} iterations...")
    print(f"{'Iter':>6} | {'Reward':>8} | {'KL':>8} | {'PolicyL':>9} | {'ValueL':>9}")
    print("-" * 55)

    for iteration in range(1, pc.total_iterations + 1):
        batch_prompts = [random.choice(prompts) for _ in range(pc.rollout_batch_size)]
        rollouts = collect_rollouts(actor, tokenizer, batch_prompts, mc, pc, device)

        input_ids, shifted_mask, old_logprobs, old_values, ref_logprobs, heuristic_rewards = prepare_ppo_batch(
            actor, reference, rollouts, mc, device
        )

        advantages, returns, kl = compute_advantages_and_returns(
            old_logprobs, ref_logprobs, old_values, shifted_mask, heuristic_rewards, pc
        )

        stats = ppo_update(
            actor, optimizer, input_ids, shifted_mask, old_logprobs, advantages, returns, pc
        )

        mean_reward = sum(r["reward"] for r in rollouts) / len(rollouts)
        mean_kl = (kl * shifted_mask).sum() / shifted_mask.sum().clamp(min=1)

        history["reward"].append(mean_reward)
        history["kl"].append(mean_kl.item())
        history["policy_loss"].append(stats["policy_loss"])
        history["value_loss"].append(stats["value_loss"])

        if iteration % pc.log_interval == 0 or iteration == 1:
            print(
                f"{iteration:6d} | {mean_reward:8.3f} | {mean_kl.item():8.4f} | "
                f"{stats['policy_loss']:9.4f} | {stats['value_loss']:9.4f}"
            )

        if iteration % pc.save_interval == 0:
            save_checkpoint(
                actor, mc, pc, iteration,
                os.path.join(pc.output_dir, f"ppo_step_{iteration}.pt"),
            )

    save_checkpoint(actor, mc, pc, pc.total_iterations, os.path.join(pc.output_dir, "ppo_final.pt"))
    print(f"\nDone! Final checkpoint saved to {pc.output_dir}/ppo_final.pt")

    return history


if __name__ == "__main__":
    run_ppo()
