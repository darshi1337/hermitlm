import math
from typing import Literal, overload

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.ffn_hidden

        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(config.d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w2(x)) * self.w1(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)

        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class HermitLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @overload
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_hidden: Literal[False] = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    @overload
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_hidden: Literal[True] = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]: ...

    def forward(self, idx, targets=None, return_hidden=False):
        _B, T = idx.shape

        pos = torch.arange(0, T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_id
            )

        if return_hidden:
            return logits, loss, x

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, temperature=0.8, top_k=50):
        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.max_seq_len:]

                logits, _ = self(idx_cond)

                logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)

                next_id = torch.multinomial(probs, num_samples=1)

                idx = torch.cat([idx, next_id], dim=1)

                if next_id.item() == self.config.eos_id:
                    break
        finally:
            self.train(was_training)

        return idx

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def param_summary(self):
        total = self.param_count()
        return f"HermitLM: {total:,} params ({total/1e6:.2f}M)"