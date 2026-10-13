"""Build synthetic tool-routing data: math / web / faq / chat.

Train rows come from templates (plus real crab-chat inputs as chat negatives).
The eval set is hand-authored realistic queries with no template overlap.
Writes data/tool_train.jsonl and data/tool_eval.jsonl.
"""

import json
import random
from pathlib import Path

MATH_TEMPLATES = [
    "solve {a} + {b}", "what is {a} * {b}", "integrate x^{n}", "differentiate x^{n}",
    "simplify {a}/{b}", "factor x^2 + {a}x + {b}", "compute {a} - {b}", "limit of 1/x as x grows",
    "sin({a}) + cos({b})", "matrix [[{a},{b}],[{c},{d}]] times x", "derivative of x^{n}",
    "equation: {a}x + {b} = {c}", "what is {a} divided by {b}", "{a} * {b} + {c}",
]

WEB_TEMPLATES = [
    "latest news about {t}", "who is {p}", "search for {t}", "what is happening today in {t}",
    "current price of {t}", "explain {t} to me", "look up {p}", "find {t} online",
    "news today", "who is the president", "search the web for {t}",
]

FAQ_KEYS = [
    "who are you", "who made you", "who created you", "what are you", "how old are you",
    "where do you live", "what do you eat", "are you alive", "do you sleep",
    "what is your purpose", "who is darshi", "can you think", "do you like humans",
    "what language do you speak", "what is hermitlm", "are you trapped",
    "what are you doing", "do you fear anything", "tell me about yourself", "what powers you",
]

FAQ_WRAPPERS = ["{k}", "{k}?", "hey, {k}", "{k} please", "crab, {k}", "tell me: {k}"]

# Hand-authored realistic eval. No template overlap by construction.
TOOL_EVAL = [
    ("math", "how much is 17 times 23"),
    ("math", "can you solve 3x + 5 = 20"),
    ("math", "derivative of x cubed"),
    ("math", "what is 144 divided by 12"),
    ("math", "help me factor x squared plus five x plus six"),
    ("math", "sin 30 degrees plus cos 60"),
    ("math", "integrate 2x dx"),
    ("math", "7 + 8 * 3"),
    ("math", "what is the limit of one over x"),
    ("math", "simplify forty two over fifty six"),
    ("math", "subtract 15 from 100"),
    ("math", "matrix multiplication example"),
    ("math", "solve for x: 2x = 10"),
    ("math", "what is nine squared"),
    ("math", "differentiate sin x"),
    ("math", "add nineteen and twenty seven"),
    ("math", "cosine of zero"),
    ("math", "factor x squared minus nine"),
    ("math", "how do I compute twelve percent of eighty"),
    ("math", "two plus two"),
    ("web", "give me the latest AI news"),
    ("web", "who is Elon Musk"),
    ("web", "search for hermit crab facts"),
    ("web", "what happened today in tech"),
    ("web", "what is the current time in tokyo"),
    ("web", "explain black holes like I am five"),
    ("web", "look up the weather in oslo"),
    ("web", "find me a recipe for miso soup"),
    ("web", "any news about spacex today"),
    ("web", "who won the match yesterday"),
    ("web", "search the web for cheap flights"),
    ("web", "what is everyone talking about"),
    ("web", "current bitcoin price"),
    ("web", "tell me something new from science"),
    ("web", "who is the prime minister of japan"),
    ("web", "look up train times to kyoto"),
    ("web", "what is new in python 3.13"),
    ("web", "find reviews of that crab restaurant"),
    ("web", "latest breakthroughs in fusion"),
    ("web", "who wrote dune"),
    ("faq", "who are you, really"),
    ("faq", "who built you"),
    ("faq", "how old are you, crab"),
    ("faq", "where is home for you"),
    ("faq", "do you ever sleep"),
    ("faq", "are you a living thing"),
    ("faq", "what do you eat down there"),
    ("faq", "what is your purpose in life"),
    ("faq", "do you like people"),
    ("faq", "can you actually think"),
    ("faq", "what language is that"),
    ("faq", "tell me about darshi"),
    ("faq", "are you stuck in that cave"),
    ("faq", "what are you up to right now"),
    ("faq", "does anything scare you"),
    ("faq", "introduce yourself"),
    ("faq", "what keeps you going"),
    ("faq", "is hermitlm your name"),
    ("faq", "who is in charge here"),
    ("faq", "do you dream"),
    ("chat", "hello crab"),
    ("chat", "are you hungry"),
    ("chat", "the water feels cold today"),
    ("chat", "goodnight, see you tomorrow"),
    ("chat", "you are moving fast"),
    ("chat", "was that loud just now"),
    ("chat", "i feel a bit lonely"),
    ("chat", "look at the bubbles rising"),
    ("chat", "is the glass clean"),
    ("chat", "it is dark in here"),
    ("chat", "time to eat something tasty"),
    ("chat", "how are you feeling"),
    ("chat", "the light just changed"),
    ("chat", "can you see outside the tank"),
    ("chat", "that noise scared me, sorry"),
    ("chat", "what do things look like to you"),
    ("chat", "everything stable on your end"),
    ("chat", "do you see yourself in the mirror"),
    ("chat", "there is a tiny shell near the gravel"),
    ("chat", "keep scuttling sideways, friend"),
]


# Natural training queries, authored by hand. Same distribution as the eval
# set, disjoint queries. Templates alone do not transfer to real phrasing.
NATURAL_TRAIN = [
    ("math", "how much is twelve plus thirty"),
    ("math", "multiply six by seven for me"),
    ("math", "can you add these numbers"),
    ("math", "what comes after solving 5x equals 25"),
    ("math", "help with my algebra homework"),
    ("math", "divide eighty four by twelve"),
    ("math", "ten squared minus three cubed"),
    ("math", "what is half of ninety"),
    ("math", "calculate fifteen percent tip on forty dollars"),
    ("math", "sine of ninety degrees"),
    ("math", "integrate three x squared"),
    ("math", "find the derivative of cosine"),
    ("math", "nine times eight is what"),
    ("math", "one hundred minus thirty seven"),
    ("math", "what is the square root of eighty one"),
    ("math", "solve the equation four x minus seven equals nine"),
    ("math", "average of ten twenty and thirty"),
    ("math", "convert fifty dollars to cents"),
    ("math", "how many sides does an octagon have"),
    ("math", "three to the power of four"),
    ("math", "what fraction is bigger, one third or one fourth"),
    ("math", "long division of one forty four by twelve"),
    ("math", "tangent of forty five degrees"),
    ("math", "expand x plus three squared"),
    ("math", "what is zero divided by five"),
    ("math", "add up one through ten"),
    ("math", "difference between ninety nine and fifty four"),
    ("math", "multiply point five by eight"),
    ("math", "how much is a dozen dozens"),
    ("math", "perimeter of a six by nine rectangle"),
    ("math", "area of a circle with radius three"),
    ("math", "is seventy three prime"),
    ("math", "greatest common divisor of twelve and eighteen"),
    ("math", "simplify nine over twelve"),
    ("math", "what is two thirds of ninety"),
    ("math", "celsius to fahrenheit for twenty degrees"),
    ("math", "how many minutes in three hours"),
    ("math", "split a hundred three ways evenly"),
    ("math", "the sum of all angles in a triangle"),
    ("math", "log base ten of a thousand"),
    ("web", "catch me up on the news"),
    ("web", "who is that actor in dune"),
    ("web", "look something up for me"),
    ("web", "what is going on in the world"),
    ("web", "how much does bitcoin cost right now"),
    ("web", "tell me about the roman empire"),
    ("web", "find the capital of australia"),
    ("web", "is it raining in london today"),
    ("web", "who directed oppenheimer"),
    ("web", "search for pictures of axolotls"),
    ("web", "what does my horoscope say"),
    ("web", "how old is the eiffel tower"),
    ("web", "why is the sky blue, exactly"),
    ("web", "when is the next solar eclipse"),
    ("web", "who holds the marathon record"),
    ("web", "what is the tallest mountain"),
    ("web", "how do planes stay in the air"),
    ("web", "where was coffee invented"),
    ("web", "what causes earthquakes"),
    ("web", "who painted the starry night"),
    ("web", "how far is the moon from earth"),
    ("web", "what is the deepest ocean trench"),
    ("web", "when did dinosaurs go extinct"),
    ("web", "which country has the most lakes"),
    ("web", "how fast does light travel"),
    ("web", "what is the population of india"),
    ("web", "who invented the telephone"),
    ("web", "why do cats purr"),
    ("web", "how tall is mount fuji"),
    ("web", "what is the currency of egypt"),
    ("web", "who won the world cup last time"),
    ("web", "how many planets have moons"),
    ("web", "what is photosynthesis in simple words"),
    ("web", "where is machu picchu located"),
    ("web", "how long is the great wall"),
    ("web", "what do bees eat in winter"),
    ("web", "who is the richest person alive"),
    ("web", "is there water on mars"),
    ("web", "how does a refrigerator work"),
    ("web", "what year did the titanic sink"),
    ("faq", "so who are you anyway"),
    ("faq", "remind me what you are"),
    ("faq", "how long have you existed"),
    ("faq", "where exactly do you reside"),
    ("faq", "do you need sleep at all"),
    ("faq", "would you say you are alive"),
    ("faq", "what does a crab like you eat"),
    ("faq", "why do you exist"),
    ("faq", "how do you feel about humans"),
    ("faq", "can you reason about things"),
    ("faq", "which language do you use"),
    ("faq", "tell me who darshi is"),
    ("faq", "do you feel trapped down there"),
    ("faq", "what keeps you busy these days"),
    ("faq", "is there anything you fear"),
    ("faq", "give me a quick intro"),
    ("faq", "what fuels a tiny model"),
    ("faq", "is hermitlm really your name"),
    ("faq", "who runs this place"),
    ("faq", "do you get dreams"),
    ("faq", "what created you"),
    ("faq", "how were you made"),
    ("faq", "are you a robot or an animal"),
    ("faq", "do you get lonely in the cave"),
    ("faq", "what is the cave like"),
    ("faq", "can you leave the tank"),
    ("faq", "how big is your world"),
    ("faq", "do you remember me"),
    ("faq", "are we friends now"),
    ("faq", "say something about this cave"),
    ("faq", "describe darshi in one line"),
    ("faq", "what do you do all day"),
    ("faq", "is it cold where you are"),
    ("faq", "do you miss the ocean"),
    ("faq", "how do you spend your time"),
    ("faq", "what is your favorite snack"),
    ("faq", "are there other crabs with you"),
    ("faq", "can you hear me through the glass"),
    ("faq", "why are you called hermit"),
    ("faq", "do you like your shell"),
    ("chat", "hey crab, what is up"),
    ("chat", "good morning little buddy"),
    ("chat", "the tank looks nice today"),
    ("chat", "i brought you some seaweed"),
    ("chat", "do you like the new rock"),
    ("chat", "swim over here for a second"),
    ("chat", "your shell looks shiny"),
    ("chat", "night night, sleep tight"),
    ("chat", "wake up, sleepyhead"),
    ("chat", "did you eat all the plankton"),
    ("chat", "the filter sounds weird"),
    ("chat", "someone tapped the glass again"),
    ("chat", "i cleaned your corner spot"),
    ("chat", "race you to the castle"),
    ("chat", "do crabs get hiccups"),
    ("chat", "show me your claws"),
    ("chat", "that hermit crab video was funny"),
    ("chat", "i wish i could breathe underwater"),
    ("chat", "your antennae are wiggling"),
    ("chat", "is that a new scratch on the glass"),
    ("chat", "the pebbles shifted overnight"),
    ("chat", "you blinked, i saw it"),
    ("chat", "come out of the shell, shy one"),
    ("chat", "i named the big rock fred"),
    ("chat", "does the light bother your eyes"),
    ("chat", "lunchtime already, hungry"),
    ("chat", "i hear clicking, is that you"),
    ("chat", "your cave smells like the sea"),
    ("chat", "high five with the tiny claw"),
    ("chat", "do you recognize my voice"),
    ("chat", "the water pump is loud today"),
    ("chat", "i dropped a pellet behind the plant"),
    ("chat", "nap time for both of us"),
    ("chat", "you are my favorite tiny roommate"),
    ("chat", "stay still, i am taking a photo"),
    ("chat", "the thermometer says twenty four"),
    ("chat", "wave at the camera, crab"),
    ("chat", "i wonder what you dream about"),
    ("chat", "back from work, missed you"),
    ("chat", "weekend vibes in the fish room"),
]


def build(per_class=1200, seed=42):
    rng = random.Random(seed)
    rows = []
    for _ in range(per_class):
        a, b, c, d, n = (rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 5))
        t = rng.choice(["crabs", "lava", "kelp", "tides", "coral", "sand", "shells"])
        p = rng.choice(["darshi", "elon musk", "marie curie", "the mayor"])
        rows.append({"input": rng.choice(MATH_TEMPLATES).format(a=a, b=b, c=c, d=d, n=n), "label": "math"})
        rows.append({"input": rng.choice(WEB_TEMPLATES).format(t=t, p=p), "label": "web"})
        rows.append({"input": rng.choice(FAQ_WRAPPERS).format(k=rng.choice(FAQ_KEYS)), "label": "faq"})
    chat_inputs = []
    with open("data/train.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("input"):
                chat_inputs.append(rec["input"])
    rng.shuffle(chat_inputs)
    for q in chat_inputs[:per_class]:
        rows.append({"input": q, "label": "chat"})
    for label, q in NATURAL_TRAIN:
        rows.append({"input": q, "label": label})
    rng.shuffle(rows)
    Path("data/tool_train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    eval_rows = [{"input": q, "label": label} for label, q in TOOL_EVAL]
    Path("data/tool_eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")
    print(f"train={len(rows)} eval={len(eval_rows)}")


if __name__ == "__main__":
    build()
