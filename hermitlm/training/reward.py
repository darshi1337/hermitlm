import re
from itertools import pairwise

from hermitlm.training.generate_data import (
    ACTIVITIES,
    BODY_PARTS,
    CRAB_OBJECTS,
    CRAB_SPOTS,
    FEELINGS,
    FOOD_TYPES,
    LIGHT_STATES,
    SOUNDS,
    WATER_DESCRIPTIONS,
    WATER_THINGS,
)

_STOPWORDS = {
    "a", "an", "the", "i", "is", "am", "are", "and", "or", "to", "of", "in",
    "on", "at", "by", "my", "its", "it", "this", "that", "near", "under",
    "over", "between", "along", "behind", "beside", "deep", "next", "where",
    "with", "for", "as", "be", "will", "if", "not", "no", "so", "but",
    "then", "than", "from", "into", "onto", "up", "down", "out",
}

_EXTRA_PERSONA_WORDS = {
    "crab", "claw", "claws", "shell", "water", "sand", "cave", "current",
    "bubble", "bubbles",
}

_BREAK_PHRASES = [
    "as an ai", "i am an ai", "i'm an ai", "language model", "i cannot",
    "i can't", "i'm sorry", "as a large language", "openai", "anthropic",
    "assistant:", "user:", "as a chatbot", "i'm just a",
]


def _build_lexicon():
    lexicon = set(_EXTRA_PERSONA_WORDS)

    for group in (
        CRAB_OBJECTS, CRAB_SPOTS, FOOD_TYPES, WATER_DESCRIPTIONS,
        ACTIVITIES, FEELINGS, WATER_THINGS, LIGHT_STATES, BODY_PARTS,
        SOUNDS,
    ):
        for phrase in group:
            for word in phrase.lower().split():
                word = word.strip(".,")
                if len(word) > 1 and word not in _STOPWORDS:
                    lexicon.add(word)

    return lexicon


PERSONA_LEXICON = _build_lexicon()


def _word_overlap_score(text):
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0

    hits = sum(1 for w in words if w in PERSONA_LEXICON)
    return hits / len(words)


def _repetition_penalty(text):
    words = text.lower().split()
    if len(words) < 4:
        return 0.0

    bigrams = list(pairwise(words))
    if not bigrams:
        return 0.0

    unique_ratio = len(set(bigrams)) / len(bigrams)
    return 1.0 - unique_ratio


def _length_penalty(text, target_min=3, target_max=40):
    n = len(text.split())
    if n == 0:
        return 1.0
    if n < target_min:
        return (target_min - n) / target_min
    if n > target_max:
        return min(1.0, (n - target_max) / target_max)
    return 0.0


def _style_penalty(text):
    letters = sum(c.isalpha() for c in text)
    if not letters:
        return 0.0

    upper = sum(c.isupper() for c in text)
    return upper / letters


def _break_character_penalty(text):
    lowered = text.lower()
    return sum(1.0 for phrase in _BREAK_PHRASES if phrase in lowered)


# Normal in-character responses land around 0.15-0.35 overlap; degenerate
# responses that just stuff persona vocabulary in regardless of relevance
# (e.g. always tacking on "current priority: <food>") hit 0.7-0.8. Capping
# the credit removes the incentive to spam past what a real answer needs.
_OVERLAP_CAP = 0.4


def score_response(text):
    """Heuristic persona-consistency reward for a generated crab response."""

    text = text.strip()
    if not text:
        return -3.0

    reward = 0.0
    reward += 2.0 * min(_word_overlap_score(text), _OVERLAP_CAP)
    reward -= 1.5 * _repetition_penalty(text)
    reward -= 1.0 * _length_penalty(text)
    reward -= 1.0 * _style_penalty(text)
    reward -= 3.0 * _break_character_penalty(text)

    return reward
