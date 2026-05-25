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