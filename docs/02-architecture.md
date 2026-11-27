# Architecture

| Field | Value |
| --- | --- |
| Parameters | 8.45M (8,453,120) |
| Layers | 6 transformer blocks |
| Hidden dim (`d_model`) | 320 |
| Attention heads | 8 (40 dims per head) |
| FFN | 800 hidden, SwiGLU / SiLU |
| Vocabulary | 3,774 byte-level BPE tokens |
| Max sequence | 512 tokens |
| Norm | LayerNorm (pre-norm residual blocks) |
| Position | Learned embeddings |
| LM head | Weight-tied with token embeddings |
| Dropout | 0.1 (disabled during probe fitting) |
| Web artifact | uint8 dynamic quantization (~9.8 MB vs 38.7 MB fp32) |

Small causal transformer. Learned positions, standard attention, LayerNorm, weight tying, and a compact SwiGLU feed-forward block.

### How the 8.45M count is derived

`HermitLM.param_count()` counts unique parameters (the tied `lm_head` is not counted twice):

| Component | Calculation | Parameters |
| --- | --- | --- |
| Token embedding | 3,774 × 320 | 1,207,680 |
| Position embedding | 512 × 320 | 163,840 |
| 6 × attention QKV | 6 × (320 × 960 + 960) | 1,848,960 |
| 6 × attention output | 6 × (320 × 320 + 320) | 616,320 |
| 6 × SwiGLU FFN | 6 × (3 × 320 × 800) | 4,608,000 |
| 6 × block LayerNorms | 6 × (2 × 320 × 2) | 7,680 |
| Final LayerNorm | 2 × 320 | 640 |
| LM head | tied to token embedding | 0 |
| **Total** | | **8,453,120** |

The raw checkpoint `state_dict` sums to 9,660,800 because it stores `lm_head.weight` a second
time; weight tying removes that duplicate 1,207,680-parameter matrix.
