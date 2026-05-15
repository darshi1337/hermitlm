import json
import random
import os
from collections import Counter

random.seed(617)

def pick(seq):
    return random.choice(seq)

def pick_weighted(pairs):
    items, weights = zip(*pairs)
    return random.choices(items, weights=weights, k=1)[0]

def maybe(text, p=0.5):
    return text if random.random() < p else ""

def join_sentences(*parts):
    return " ".join(p.strip() for p in parts if p).strip()

_last_picks = {}

def pick_unique(key, seq):
    if not seq:
        return ""
    prev = _last_picks.get(key)
    for _ in range(5):
        choice = random.choice(seq)
        if choice != prev:
            break
    _last_picks[key] = choice
    return choice


CRAB_OBJECTS = [
"sand block","gravel patch","stone","cobblestone","mossy stone",
"seaweed","kelp","coral block","coral fan","dead coral",
"shell","giant shell","tiny shell","driftwood","log",
"shipwreck","treasure chest","buried chest","ruins",
"cave entrance","small cave","underwater ruin","arch",
"pillar","rock formation","bubble column","magma block",
"soul sand","glass block","sandstone","clay patch",
]

CRAB_SPOTS = [
"near the sand","under the gravel","beside the stone",
"behind the coral","inside the shell","near the kelp",
"at the cave entrance","deep in the cave","by the ruins",
"next to the treasure chest","under the driftwood",
"between the rocks","along the seabed","near the bubble column",
"on top of the sand block","in my hiding spot",
"where the water flows gently","where the light reaches",
"at the edge of the trench","near the magma block",
]

FOOD_TYPES = [
"kelp bits","algae","tiny fish","plankton",
"moss scraps","sea pickles","dead fish bits",
"mysterious crumbs","bubble snacks","sand snacks",
"soft algae","crunchy bits","sinking food","floating food",
]

WATER_DESCRIPTIONS = [
"clear","murky","cold","warm","just right",
"a bit cloudy","salty","fresh-ish","calm",
"rough","gentle","swirling","still",
]

ACTIVITIES = [
"scuttling sideways","digging in the sand",
"hiding in my shell","peeking out carefully",
"watching bubbles rise","climbing a rock",
"exploring the cave","guarding my spot",
"pretending to be sand","snipping at nothing",
"chasing a tiny fish","investigating debris",
"burying myself halfway","staring dramatically",
"clicking my claws","doing nothing but looking busy",
"moving one inch and stopping","observing everything",
]

FEELINGS = [
"good","fine","calm","hungry","suspicious",
"grumpy","content","alert","sleepy",
"territorial","curious","peaceful",
]

WATER_THINGS = [
"current","temperature","bubbles",
"pressure","flow","salinity",
"warmth","coldness",
]

LIGHT_STATES = [
"dim","bright","blue","greenish",
"murky","shimmering","flickering",
"soft","harsh",
]

BODY_PARTS = [
"claws","legs","shell","eyes",
"antennae","little legs","hard shell",
]

SOUNDS = [
"click","clack","tap","scrape",
"rumble","bubble","crunch",
"thud","echo",
]

def crab_greeting():
    return join_sentences(
        pick_weighted([
            ("oh. it's you.", 3),
            ("hello.", 2),
            ("you again.", 2),
            ("i noticed you.", 1),
        ]),
        pick([
            f"i was just {pick_unique('act', ACTIVITIES)}.",
            f"i am {pick_unique('spot', CRAB_SPOTS)}.",
            f"the water is {pick(WATER_DESCRIPTIONS)} today.",
            f"the {pick(CRAB_OBJECTS)} looks different today.",
            f"i heard a {pick(SOUNDS)} earlier.",
        ]),
        pick_weighted([
            (f"the {pick(WATER_THINGS)} feels stable.", 2),
            ("i am observing.", 1),
            ("", 3),
        ])
    )


def crab_feeling():
    return join_sentences(
        pick([
            f"i feel {pick_unique('feel', FEELINGS)}.",
            f"{pick_unique('feel', FEELINGS)}. that is my state.",
            f"currently {pick_unique('feel', FEELINGS)}.",
        ]),
        pick([
            f"the water is {pick(WATER_DESCRIPTIONS)}.",
            f"i was {pick_unique('act', ACTIVITIES)}.",
            f"the {pick(WATER_THINGS)} is steady.",
            f"i secured {pick_unique('spot', CRAB_SPOTS)}.",
            "nothing has tried to eat me recently.",
        ]),
        pick_weighted([
            ("this is acceptable.", 2),
            ("", 3),
        ])
    )


def crab_temp_hot():
    return join_sentences(
        pick([
            "the water is too warm.",
            "temperature is rising.",
            "this is not ideal.",
        ]),
        pick([
            "oxygen will decrease.",
            f"i am relocating to {pick_unique('spot', CRAB_SPOTS)}.",
            f"my {pick(BODY_PARTS)} feel slower.",
            "i will conserve energy.",
        ]),
        pick_weighted([
            ("adjustment is required.", 2),
            ("you should correct this.", 1),
            ("", 3),
        ])
    )


def crab_temp_cold():
    return join_sentences(
        pick([
            "the water is colder.",
            "temperature has dropped.",
            "cold conditions detected.",
        ]),
        pick([
            "oxygen levels are better.",
            f"i will stay {pick_unique('spot', CRAB_SPOTS)}.",
            f"my {pick(BODY_PARTS)} are less responsive.",
            f"i stopped {pick_unique('act', ACTIVITIES)}.",
        ]),
        pick_weighted([
            ("this is acceptable.", 2),
            ("i will adapt.", 1),
            ("", 3),
        ])
    )


def crab_food(hunger_level=0.5):
    if random.random() < 0.08:
        return "food. food. food. give food now."

    if hunger_level > 0.85:
        return "food required immediately. i will not wait."

    return join_sentences(
        pick_weighted([
            ("food detected.", 3),
            ("i am interested in food.", 2),
            ("this is about food, correct.", 2),
        ]),
        pick([
            f"provide the {pick_unique('food', FOOD_TYPES)}.",
            f"i prefer the {pick_unique('food', FOOD_TYPES)}.",
            "consumption is a priority.",
            f"i have been {pick_unique('act', ACTIVITIES)} waiting.",
            "my claws are ready.",
        ]),
        pick_weighted([
            ("do not delay.", 2),
            ("", 3),
        ])
    )

def generate_line():
    fn = pick_weighted([
        (crab_greeting, 3),
        (crab_feeling, 3),
        (crab_food, 2),
        (crab_temp_hot, 1),
        (crab_temp_cold, 1),
    ])
    return fn()

def generate_dataset(n=10000, path="crab_data.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            sample = {"text": generate_line()}
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    generate_dataset()
    print("Dataset generated: crab_data.jsonl")