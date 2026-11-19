"""Hand-authored natural user messages for the 21 persona fine types.

These are intentionally *not* template-generated. The original generator
reaches ~1.0 accuracy on its own eval set but only ~14/20 on natural
paraphrases, so this corpus supplies realistic phrasings, typos and mixed
signals that the templates never produce.
"""
import random

# 15+ natural phrasings per fine type. Kept short and crab-chat-like, but
# varied in register, length and punctuation.
NATURAL_INPUTS: dict[str, list[str]] = {
    "greeting": [
        "hey crab, you there?", "hi little guy", "good to see you again",
        "morning! how's the tank?", "hello? are you awake", "yo crab, what's up",
        "hey there, shell buddy", "hiya, didn't mean to startle you",
        "greetings, tiny one", "oh hey, it's you", "well hello there",
        "hi crab, missed me?", "sup crab", "howdy little crab",
        "hey, you hanging out in there?", "hello again, friend",
        "hi! ready to talk?", "good evening, crab", "hey, is anyone home?",
        "hello down there",
    ],
    "bye": [
        "alright, I'm off", "see you later, crab", "gotta go, take care",
        "I'm heading out now", "bye little guy", "talk soon",
        "later, shell-face", "I should get going", "cya tomorrow",
        "night night, crab", "okay I'll leave you be", "catch you later",
        "time for me to go", "say goodbye to the crab for me", "I'm outta here",
        "see ya around",
    ],
    "lonely": [
        "does anyone ever visit you", "do you get lonely down there",
        "it's quiet in here today", "anyone else around?",
        "i wish i had someone to talk to", "do you miss company",
        "it feels empty without you", "is it just you in that tank",
        "i could use a friend", "does the quiet bother you",
        "anybody there besides us", "i feel a bit alone tonight",
        "do you want some company", "nobody's around, huh", "i'm kinda lonely",
    ],
    "food": [
        "are you hungry yet", "want a snack", "i brought kelp",
        "what do crabs eat anyway", "can i feed you", "is it lunchtime for you",
        "did you find any food", "how's the plankton supply",
        "here's a shrimp, want it", "you look hungry",
        "should i drop a pellet in", "what's your favorite thing to eat",
        "are you still hungry", "i saved you a bit of fish", "time to eat?",
    ],
    "water": [
        "is the water cold today", "how's the water level",
        "did you change the water yet", "is it too warm in there",
        "the tank water looks murky", "do you like the current",
        "is the filter making waves", "water temperature okay?",
        "can you breathe fine in there", "the water smells different",
        "is it too deep for you", "how's the flow today",
        "did the water get changed", "is everything flooded",
        "the water's looking clear",
    ],
    "bubbles": [
        "look at all those bubbles", "are you blowing bubbles",
        "where do the bubbles come from", "i see bubbles rising",
        "pop a bubble for me", "is there air in there",
        "bubbles everywhere today", "do bubbles scare you",
        "can you make a bubble ring", "those bubbles are huge",
        "the bubbles won't stop", "are the bubbles from the filter",
        "i like watching the bubbles", "caught a bubble on your shell?",
        "more bubbles than usual",
    ],
    "glass": [
        "is that glass clean", "can you see through the glass",
        "someone tapped the glass again", "the glass is foggy",
        "are you looking at me through the glass", "did you scratch the glass",
        "is the tank glass thick", "keep off the glass please",
        "the glass is reflecting light", "can you see outside the glass",
        "i cleaned the glass today", "why are you behind glass",
        "the glass looks spotless", "tap the glass gently", "is the glass safe",
    ],
    "environment": [
        "how's the tank looking", "did you rearrange the rocks",
        "the sand looks different", "is your cave comfortable",
        "what's the tank like inside", "did someone move the plants",
        "your home looks cozy", "is the gravel okay",
        "anything new in the tank", "the decorations changed",
        "how's your corner of the tank", "did you find a new hiding spot",
        "is the tank too small", "your rock formation looks cool",
        "what's under the sand",
    ],
    "movement": [
        "are you moving around today", "why are you scuttling so fast",
        "can you walk sideways", "you're always on the move",
        "where are you going", "did you just run past me",
        "are you exploring", "slow down a little", "you look restless",
        "come over here", "are you stuck or just moving",
        "i saw you dart away", "keep scuttling, little guy",
        "where's the fire, crab", "you're quick today",
    ],
    "light": [
        "is the light too bright", "did you turn the lamp on",
        "the light changed suddenly", "is it daytime in there",
        "do you like the new light", "it's dark in your tank",
        "turn off the light for a bit", "the sunlight is hitting the glass",
        "can you see the light", "it's getting dim in here",
        "is the lamp still working", "the light is flickering",
        "do you prefer the dark", "shining a light on you", "night mode or day mode?",
    ],
    "color": [
        "what color are you", "your shell is so red",
        "did you see the blue one", "i like your orange claws",
        "is everything green in there", "what's your favorite color",
        "the coral looks pink today", "you blend into the sand",
        "that shell is bright", "are you more brown or red",
        "the tank has new colors", "why so pale today",
        "do crabs see color", "your shell matches the rocks",
        "that's a vivid little crab",
    ],
    "sound": [
        "did you hear that noise", "the filter is loud today",
        "that tapping scared me", "is it noisy in there",
        "do you hear the pump", "something just clicked",
        "the tank is so quiet", "was that you making noise",
        "i hear bubbles popping", "the room is too loud",
        "did the glass ring", "sounds like a rumble",
        "can you hear me tapping", "that was a strange sound",
        "everything's silent now",
    ],
    "noise": [
        "what was that bang", "the whole tank shook",
        "something crashed nearby", "did you feel that vibration",
        "i dropped a spoon, sorry", "that was loud, huh",
        "the floor rattled", "a door slammed", "did the shelf fall",
        "earthquake?", "the pump started humming loudly",
        "someone bumped the tank", "that clatter came from outside",
        "was that thunder", "everything jumped",
    ],
    "state": [
        "how are you feeling", "are you okay", "you seem tired",
        "how's it going down there", "are you stressed",
        "i'm not doing so great today", "do you feel calm",
        "are you upset about something", "how's your mood",
        "you look a little down", "i feel weird today",
        "are you happy in there", "is everything alright",
        "you seem tense", "i'm feeling restless",
    ],
    "breathing": [
        "can you breathe underwater", "how do you breathe in there",
        "are you holding your breath", "do you need air",
        "are your gills okay", "how long can you stay under",
        "are you suffocating", "do you come up for air",
        "the water's low, can you breathe", "you breathe through your shell?",
        "is it hard to breathe today", "do crabs have lungs",
        "can you breathe out of water", "are your gills working",
        "i can't tell if you're breathing",
    ],
    "temp_cold": [
        "it's freezing in here", "is the water too cold",
        "i feel cold just looking at you", "did the heater break",
        "brrr, it's chilly", "the tank water got cold",
        "are you shivering", "winter is coming", "turn up the heat",
        "cold water today?", "is that ice on the glass",
        "i'm freezing over here", "do you like the cold",
        "temperature dropped overnight", "so cold today",
    ],
    "temp_hot": [
        "it's boiling in there", "is the water too hot",
        "i'm sweating just watching", "did the heater get stuck on",
        "it's a heatwave today", "the tank feels warm",
        "are you overheating", "can you cool off", "too hot to handle?",
        "the sun is baking the tank", "temperature's rising",
        "i feel hot today", "is the lamp too strong",
        "it's sweltering", "water's getting hot",
    ],
    "night": [
        "is it night time for you", "do you sleep at night",
        "the lights are off now", "goodnight little crab",
        "are you asleep yet", "it's late, should you be awake",
        "night night", "do you dream at night", "the tank is dark now",
        "i'm going to bed", "everything gets quiet at night",
        "do you see the moon", "are you a night owl crab",
        "it's past your bedtime", "sweet dreams",
    ],
    "misc": [
        "tell me something random", "what's on your mind",
        "i don't know what to ask", "say anything",
        "talk to me about nothing", "what are you thinking about",
        "do you have a favorite word", "random question for you",
        "what's the weather like in there", "can i just chat",
        "say something funny", "do you have secrets",
        "what do you do all day", "i'm bored, entertain me", "anything new?",
    ],
    "confused": [
        "what is quantum computing", "explain blockchain to me",
        "who won the football match", "what's the capital of france",
        "can you help with my taxes", "i need stock advice",
        "what's the meaning of life", "how do airplanes fly",
        "tell me about the roman empire", "what's a black hole",
        "solve my homework", "who is the president",
        "translate this to swedish", "what time is it in tokyo",
        "recommend a restaurant",
    ],
    "reflection": [
        "do you ever look at yourself", "are you staring at your reflection",
        "what do you see in the mirror", "is that your own face",
        "do crabs recognize themselves", "you seem fascinated by the glass",
        "looking at yourself again?", "do you know who you are",
        "your reflection is staring back", "are you talking to yourself",
        "do you like what you see", "is there another crab in there",
        "mirror, mirror", "do you ever wonder who you are",
        "that's just your reflection",
    ],
}

_PREFIXES = ["", "", "", "hey, ", "um, ", "so, ", "quick question: ", "sorry, ", "okay, "]
_SUFFIXES = ["", "", "", "?", " please", " lol", " thanks", "...", "!"]


def perturb(text, rng=None):
    """Lightly rewrite a natural phrase without turning it into a template."""
    rng = rng or random
    out = text
    if rng.random() < 0.35:
        out = rng.choice(_PREFIXES) + out
    if rng.random() < 0.45:
        suffix = rng.choice(_SUFFIXES)
        if not (suffix == "?" and out.endswith("?")):
            out = out + suffix
    if rng.random() < 0.15:
        # Swap two adjacent letters in a word to simulate a typo.
        words = out.split(" ")
        candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
        if candidates:
            i = rng.choice(candidates)
            w = words[i]
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            out = " ".join(words)
    return out
