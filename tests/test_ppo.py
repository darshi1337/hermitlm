import json

import torch
from tokenizers import Tokenizer

from hermitlm.config import HermitConfig, PPOConfig
from hermitlm.training.model import HermitLM
from hermitlm.training.ppo import run_ppo
from hermitlm.training.prepare_data import train_tokenizer

TOY_TEXTS = [
    "<|im_start|>user\nhello crab<|im_end|>\n<|im_start|>assistant\nyou again. i am near the sand.<|im_end|>",
    "<|im_start|>user\nare you hungry<|im_end|>\n<|im_start|>assistant\nfood detected. i prefer kelp bits.<|im_end|>",
    "<|im_start|>user\nhow are you<|im_end|>\n<|im_start|>assistant\ni feel calm. the water is clear.<|im_end|>",
]

TOY_PROMPTS = [
    {"input": "hello crab", "output": "you again."},
    {"input": "are you hungry", "output": "food detected."},
    {"input": "how are you", "output": "i feel calm."},
]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def _build_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    tokenizer_path = data_dir / "tokenizer.json"
    train_tokenizer(TOY_TEXTS, str(tokenizer_path), vocab_size=300)

    _write_jsonl(data_dir / "train.jsonl", TOY_PROMPTS)
    _write_jsonl(data_dir / "eval.jsonl", TOY_PROMPTS)

    return data_dir, tokenizer_path


def test_run_ppo_smoke(tmp_path):
    data_dir, tokenizer_path = _build_data_dir(tmp_path)
    vocab_size = Tokenizer.from_file(str(tokenizer_path)).get_vocab_size()

    mc = HermitConfig(
        vocab_size=vocab_size,
        max_seq_len=64,
        d_model=16,
        n_layers=2,
        n_heads=2,
        ffn_hidden=32,
        dropout=0.0,
    )

    sft_checkpoint = tmp_path / "sft.pt"
    base_model = HermitLM(mc)
    torch.save(
        {"model_state_dict": base_model.state_dict(), "config": vars(mc)},
        sft_checkpoint,
    )

    output_dir = tmp_path / "ppo_out"

    pc = PPOConfig(
        sft_checkpoint=str(sft_checkpoint),
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        total_iterations=2,
        rollout_batch_size=2,
        ppo_epochs=1,
        minibatch_size=2,
        max_new_tokens=6,
        log_interval=1,
        save_interval=2,
        device="cpu",
    )

    history = run_ppo(mc, pc)

    assert len(history["reward"]) == 2
    assert len(history["kl"]) == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in history["policy_loss"])
    assert all(torch.isfinite(torch.tensor(v)) for v in history["value_loss"])

    assert (output_dir / "ppo_final.pt").exists()
    assert (output_dir / "ppo_step_2.pt").exists()

    saved = torch.load(output_dir / "ppo_final.pt", map_location="cpu")
    assert not any(k.startswith("value_head") for k in saved["model_state_dict"])

    reloaded = HermitLM(mc)
    reloaded.load_state_dict(saved["model_state_dict"])
