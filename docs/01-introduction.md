# HermitLM introduction

A ~8M parameter LLM that talks like a tiny crab in an aquatic world, chattable entirely in your browser.

```text
You> hello crab
Hermit> hello. i heard a thud earlier. i am observing. my antennae report warmth changes.

You> are you hungry
Hermit> food detected. i prefer the mysterious crumbs. my antennae report temperature changes.

You> are you a crab
Hermit> currently good. the water is gentle. current priority: soft algae.

You> do you know the capital of france
Hermit> Google News - Artificial intelligence - Latest: Read full articles, watch videos, browse thousands of titles and more on the "Artificial intelligence" topic with Google News.

You> who made you
Hermit> I was created by Darshi.

You> solve 2+2
Hermit> math! *shudders*. here, take wolfram's answer: 4

You> integrate x^2
Hermit> math! *shudders*. here, take wolfram's answer: integral x^2 dx = x^3/3 + constant

You> latest AI news
Hermit> Google News - Artificial intelligence - Latest: Read full articles, watch videos, browse thousands of titles and more on the "Artificial intelligence" topic with Google News.

You> goodnight crab
Hermit> it is now dark. movement will decrease. activity will continue as guarding my spot.
```

Chat replies are sampled, so exact wording varies. The math intro is picked at random from a
pool of crab-flavoured lines, and web results depend on live search availability. Yes/no questions
stay in persona when `p_yes` is low; an out-of-scope one escalates to web search.

## 📖 How it works

A 15-chapter textbook explains the project step by step, with analogies, worked calculations,
annotated code and diagrams. Formulas are written in LaTeX and render on GitHub; start at
[chapters/00_overview.md](../chapters/00_overview.md), or use the [chapter index](../chapters/README.md).

| Topic | Chapter |
|---|---|
| Tokenization, embeddings, attention, transformer blocks | [0–5](../chapters/README.md) |
| Full model, training loop, generation | [6–8](../chapters/README.md) |
| Tools, decision model, PPO/DPO | [9–11](../chapters/README.md) |
| ONNX export, browser inference, evaluation | [12–14](../chapters/README.md) |

Every formula and hyperparameter is collected in [Chapter 14](../chapters/14_glossary_math.md).
