from dataclasses import dataclass


@dataclass
class HermitConfig:
    vocab_size: int = 2833
    max_seq_len: int = 512

    d_model: int = 320
    n_layers: int = 6
    n_heads: int = 8

    ffn_hidden: int = 800 

    dropout: float = 0.1

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 32

    learning_rate: float = 2e-4
    min_lr: float = 2e-5

    warmup_steps: int = 800 

    max_steps: int = 30000

    weight_decay: float = 0.1
    grad_clip: float = 1.0

    eval_interval: int = 200
    save_interval: int = 1000

    device: str = "auto"
    seed: int = 42

    data_dir: str = "data"
    output_dir: str = "checkpoints"


@dataclass
class PPOConfig:
    sft_checkpoint: str = "checkpoints/best_model.pt"
    data_dir: str = "data"
    output_dir: str = "checkpoints/ppo"

    total_iterations: int = 200
    rollout_batch_size: int = 16
    ppo_epochs: int = 4
    minibatch_size: int = 8

    max_new_tokens: int = 48
    temperature: float = 1.0
    top_k: int = 50

    learning_rate: float = 1e-5
    kl_coef: float = 0.15
    clip_epsilon: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    log_interval: int = 10
    save_interval: int = 50

    device: str = "auto"
    seed: int = 42
