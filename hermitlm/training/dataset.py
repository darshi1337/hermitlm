import json

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset


def format_chat(data):
    """Convert dataset row into structured chat format."""

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
            role = m["role"]
            content = m["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return text

    return None

class HermitDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_len: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.samples = []

        # Special token IDs
        self.bos_id = self.tokenizer.token_to_id("<|im_start|>")
        self.eos_id = self.tokenizer.token_to_id("<|im_end|>")
        self.pad_id = self.tokenizer.token_to_id("<pad>") or 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                text = format_chat(data)
                if not text:
                    continue

                ids = self.tokenizer.encode(text).ids

                # format_chat() already wraps input/output and messages rows
                # in <|im_start|>/<|im_end|> turns, which encode() tokenizes
                # as the real bos/eos ids. Only bare "text" rows need an
                # explicit wrap, or every chat sample ends up double-wrapped.
                if "text" in data:
                    ids = [self.bos_id] + ids + [self.eos_id]

                # Truncate
                if len(ids) > max_len:
                    ids = ids[:max_len]

                if len(ids) >= 2:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]

        x = ids[:-1]
        y = ids[1:]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch, strict=False)

    max_len = max(len(x) for x in xs)

    padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(xs), max_len), dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
        padded_x[i, :len(x)] = x
        padded_y[i, :len(y)] = y
        attention_mask[i, :len(x)] = 1

    return padded_x, padded_y, attention_mask

def get_dataloader(
    path,
    tokenizer_path,
    max_len=512,
    batch_size=32,
    shuffle=True,
):
    dataset = HermitDataset(path, tokenizer_path, max_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(
            batch,
            pad_id=dataset.pad_id
        ),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset
