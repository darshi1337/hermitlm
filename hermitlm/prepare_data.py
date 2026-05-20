import json
import os
import random

random.seed()

DATA_DIR = "data"
VOCAB_SIZE = 4096

SPECIAL_TOKENS = [
    "<pad>",
    "<|im_start|>",
    "<|im_end|>",
]

def train_tokenizer(texts, save_path, vocab_size=VOCAB_SIZE):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
    )

    print(f"Training tokenizer on {len(texts)} samples...")
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    tokenizer.save(save_path)
    print(f"Tokenizer saved: {tokenizer.get_vocab_size()} tokens")

    return tokenizer

def extract_text(data):
    if "text" in data:
        return data["text"]

    elif "input" in data and "output" in data:
        return (
            f"<|im_start|>user\n{data['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{data['output']}<|im_end|>"
        )

    elif "messages" in data:
        text = ""
        for m in data["messages"]:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return text

    return None

def prepare(data_dir=DATA_DIR, n_samples=100000, eval_ratio=0.05):
    os.makedirs(data_dir, exist_ok=True)

    print(f"Generating {n_samples} samples...")

    try:
        from .generate_data import generate_dataset
    except ImportError:
        from generate_data import generate_dataset

    generate_dataset(n_samples, eval_ratio)

    texts = []

    for file in ["data/train.jsonl", "data/eval.jsonl"]:
        if os.path.exists(file):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    text = extract_text(data)
                    if text:
                        texts.append(text)

    print(f"Collected {len(texts)} texts")

    tokenizer_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer = train_tokenizer(texts, tokenizer_path)

    test = (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\nstate stable.<|im_end|>"
    )

    tokens = tokenizer.encode(test).tokens
    print(tokens)

if __name__ == "__main__":
    prepare()