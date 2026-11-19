"""Write the hand-written held-out intent eval set.

These messages were authored by hand and are deliberately distinct from both
the template generator and the natural paraphrase training pool. They are the
only honest measure of paraphrase generalization in this repo.
"""
import json
from pathlib import Path

HUMAN_EVAL: dict[str, list[str]] = {
    "greeting": [
        "well if it isn't the little crab",
        "hey, still alive in there?",
        "hi there friend, long time no see",
        "good morning, you tiny menace",
        "hello? somebody home in that shell?",
        "oh there you are",
    ],
    "bye": [
        "that's all for now, crab",
        "i'll be back tomorrow, keep the shell safe",
        "alright, i'm leaving you alone",
        "later little buddy, don't cause trouble",
        "signing off",
        "gotta run, see you around",
    ],
    "lonely": [
        "does it ever get boring being the only crab",
        "i could really use a chat tonight",
        "everyone else left, just us now",
        "do you ever wish someone would visit",
        "the silence in here is getting to me",
        "anyone home besides you?",
    ],
    "food": [
        "have you eaten anything today",
        "i think you could use a good meal",
        "would you like me to drop some algae in",
        "is dinner served in the tank",
        "what's for lunch, crab",
        "you're eyeing that food, aren't you",
    ],
    "water": [
        "the waterline dropped an inch",
        "is the current too strong for you today",
        "did the filter stop circulating",
        "this water is way too warm if you ask me",
        "how's the water quality lately",
        "i think the tank needs a partial change",
    ],
    "bubbles": [
        "there's a whole column of bubbles over there",
        "are those coming from your shell",
        "the bubbles are sticking to the glass",
        "i counted twelve bubbles just now",
        "what makes all that fizz",
        "bubble patrol reporting in",
    ],
    "glass": [
        "i can barely see you through this glass",
        "did you leave a mark on the pane",
        "please don't climb the glass again",
        "the glass keeps reflecting my face",
        "is the tank front scratched up",
        "there's a smudge right by your corner",
    ],
    "environment": [
        "who moved my favorite rock",
        "the layout of the tank looks different",
        "is your little cave still comfortable",
        "there's sand everywhere today",
        "did the plants get trimmed",
        "your neighborhood looks crowded",
    ],
    "movement": [
        "you certainly get around in there",
        "why did you bolt when I walked by",
        "are you patrolling the whole tank",
        "slow down, i can't keep up",
        "come here, i'm not going to hurt you",
        "you disappeared behind the rock again",
    ],
    "light": [
        "the lamp is way too intense tonight",
        "i turned the lights down for you",
        "is the sun coming through the window",
        "everything is shadowy in your corner",
        "did someone leave the tank light on",
        "it's bright enough to read in there",
    ],
    "color": [
        "that shell is a startling shade of red",
        "you look different in this light",
        "is your natural color orange or brown",
        "the new coral is bright pink",
        "you match the gravel surprisingly well",
        "i've never seen a crab that color",
    ],
    "sound": [
        "do you hear the refrigerator humming",
        "that clicking is coming from your tank",
        "the filter sounds quieter today",
        "someone was tapping on the glass earlier",
        "the room is unusually silent",
        "what was that little chirp",
    ],
    "noise": [
        "the whole shelf rattled just now",
        "was that a clap of thunder",
        "i dropped a book and nearly jumped",
        "the construction outside is relentless",
        "something hit the tank lid",
        "that vibration came from the floor",
    ],
    "state": [
        "you seem off your game today",
        "how's your general mood lately",
        "i'm having a rough one myself",
        "are you feeling better than yesterday",
        "everything alright back there",
        "you're awfully quiet",
    ],
    "breathing": [
        "how long can you stay submerged",
        "do your gills ever get tired",
        "i worry about the low water level",
        "can you manage without a bubble nest",
        "are you okay breathing in there",
        "do crabs need to surface at all",
    ],
    "temp_cold": [
        "the heater quit during the night",
        "this tank feels like an ice bath",
        "my hands are cold just looking at it",
        "is there a warmer spot in there",
        "winter really got to this room",
        "the water thermometer is low",
    ],
    "temp_hot": [
        "the sun has been baking this tank all afternoon",
        "i think the heater is stuck again",
        "it's sweltering in your little world",
        "can you move somewhere cooler",
        "the water feels too warm to me",
        "turn that lamp down before you cook",
    ],
    "night": [
        "the house is finally quiet",
        "are you still awake at this hour",
        "i'm heading to bed, night crab",
        "the moon is shining right into the tank",
        "do you rest when the lights go out",
        "everything looks different after dark",
    ],
    "misc": [
        "i have nothing specific to say, talk to me",
        "surprise me with a sentence",
        "what's the strangest thing you've seen",
        "do you ever get curious about me",
        "tell me a tiny secret",
        "let's just chat for a minute",
    ],
    "confused": [
        "could you summarize the last election for me",
        "i need help with my mortgage paperwork",
        "what's the difference between a virus and bacteria",
        "who is going to win the next election",
        "translate this paragraph into german",
        "what's the best way to invest a small amount",
    ],
    "reflection": [
        "are you admiring yourself again",
        "that's your own face in the glass",
        "do you wonder who you really are",
        "is there a second crab staring back",
        "you've been looking in the mirror a lot",
        "does your reflection talk back",
    ],
}


def main():
    out = Path("data/human_eval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"input": text, "output": "", "type": fine}
        for fine, texts in HUMAN_EVAL.items()
        for text in texts
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} hand-written rows)")


if __name__ == "__main__":
    main()
