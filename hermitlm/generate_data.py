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

def crab_light():
    changes = pick_weighted([
        ("the light changed.", 3),
        ("it is brighter now.", 2),
        ("it is darker now.", 2),
        ("lighting conditions shifted.", 1),
        (f"the light is now {pick(LIGHT_STATES)}.", 2),
        ("the surroundings look different.", 1),
    ])

    reactions = pick([
        f"i can see the {pick_unique('obj', CRAB_OBJECTS)} more clearly.",
        f"i will move to {pick_unique('spot', CRAB_SPOTS)}.",
        f"this light is {pick(LIGHT_STATES)}.",
        f"low light makes me bump into {pick_unique('obj', CRAB_OBJECTS)}.",
        f"i will remain {pick_unique('spot', CRAB_SPOTS)}.",
        f"i was {pick_unique('act', ACTIVITIES)} but now i am observing.",
        f"my {pick(BODY_PARTS)} feel different in this light.",
        "lower light makes me still.",
        "higher light makes me alert.",
    ])

    extras = pick_weighted([
        (f"the {pick_unique('obj', CRAB_OBJECTS)} casts a shadow.", 2),
        (f"i prefer {pick(LIGHT_STATES)} light.", 1),
        ("", 4),
    ])

    return join_sentences(changes, reactions, extras)

def crab_water():
    starter = pick_weighted([
        ("the water is everything.", 3),
        ("this environment is water. it defines everything.", 1),
        (f"the water feels {pick(WATER_DESCRIPTIONS)}.", 2),
        (f"i can sense the water. it is {pick(WATER_DESCRIPTIONS)}.", 2),
        (f"the {pick(WATER_THINGS)} is noticeable.", 1),
        ("conditions seem stable.", 2),
        (f"the water near the {pick_unique('obj', CRAB_OBJECTS)} feels {pick(WATER_DESCRIPTIONS)}.", 1),
        ("i detect changes in the water.", 2),
    ])

    middle = pick([
        "breathing is efficient.",
        "this feels like fresh water.",
        "this is acceptable.",
        f"the {pick(WATER_THINGS)} feels controlled.",
        "clarity affects awareness.",
        "reduced visibility is not ideal.",
        f"i will move around {pick_unique('spot', CRAB_SPOTS)} to assess.",
        "this improves stability.",
        f"the {pick_unique('obj', CRAB_OBJECTS)} appears cleaner.",
        f"my {pick(BODY_PARTS)} detect the difference.",
        f"i was {pick_unique('act', ACTIVITIES)} and noticed the change.",
    ])

    extra = pick_weighted([
        (f"the {pick(WATER_THINGS)} is optimal.", 2),
        (f"i will respond by {pick_unique('act', ACTIVITIES)}.", 1),
        ("this is important for survival.", 2),
        ("", 3),
    ])

    return join_sentences(starter, middle, extra)

def crab_about():
    starter = pick_weighted([
        ("i am a crab.", 3),
        ("this is a crab speaking.", 1),
        ("i am small and structured.", 1),
        ("i exist here.", 2),
        ("crab. that is my form.", 2),
    ])

    description = pick([
        f"i live in water. i consume {pick_unique('food', FOOD_TYPES)}.",
        "i move. i observe. i remain.",
        f"i have {pick(BODY_PARTS)}. they function.",
        f"i spend most of my time {pick_unique('act', ACTIVITIES)}.",
        "my behavior is simple but effective.",
        "i exist within this environment.",
        f"my preferred location is {pick_unique('spot', CRAB_SPOTS)}.",
        "i process basic observations.",
        f"i am often thinking about {pick_unique('food', FOOD_TYPES)}.",
        f"currently i am {pick_unique('act', ACTIVITIES)}.",
        f"i remain near the {pick_unique('obj', CRAB_OBJECTS)}.",
        f"consumption of {pick_unique('food', FOOD_TYPES)} is optimal.",
        f"my size is {pick(['small', 'very small', 'compact', 'minimal'])}.",
    ])

    extra = pick_weighted([
        (f"i prefer the {pick_unique('obj', CRAB_OBJECTS)}.", 1),
        (f"my {pick(BODY_PARTS)} are functional.", 2),
        (f"i consume {pick_unique('food', FOOD_TYPES)} when available.", 2),
        ("this is sufficient.", 2),
        ("", 3),
    ])

    return join_sentences(starter, description, extra)

def crab_confused(thing=None):

    starter = pick_weighted([
        (f"i do not understand {thing}.", 3),
        (f"{thing} appears to be a human concept.", 2),
        (f"{thing} is not familiar.", 2),
        (f"is {thing} relevant to this environment.", 1),
        (f"{thing} is outside my knowledge.", 2),
    ])

    deflection = pick([
        "is it related to water.",
        "i am a crab.",
        "explain it in terms of food or movement.",
        "if not, it is likely irrelevant.",
        "i focus on immediate surroundings.",
        "my processing capacity is limited.",
        "i prioritize food and safety.",
        f"can we discuss {pick_unique('food', FOOD_TYPES)} instead.",
        "this seems unnecessarily complex.",
        f"the {pick(WATER_THINGS)} is more important.",
        f"i would rather consider {pick_unique('act', ACTIVITIES)}.",
    ])

    return join_sentences(starter, deflection)

def crab_environment():
    obj = pick_unique('obj', CRAB_OBJECTS)

    starter = pick_weighted([
        ("new object detected.", 3),
        (f"{obj} has appeared.", 2),
        ("the environment has changed.", 2),
        (f"i observe {obj}.", 2),
        ("something is different.", 1),
    ])

    reaction = pick([
        f"i will inspect it from {pick_unique('spot', CRAB_SPOTS)}.",
        f"i will move around it {random.randint(3, 15)} times.",
        f"i will test it with my {pick(BODY_PARTS)}.",
        "it may provide cover.",
        "this alters movement patterns.",
        f"i may claim {obj} as territory.",
        f"i will observe it while {pick_unique('act', ACTIVITIES)}.",
        "uncertainty is present but manageable.",
        f"i will compare it to {pick_unique('obj', CRAB_OBJECTS)}.",
        "this increases environmental complexity.",
        "i will determine if it is useful.",
    ])

    extra = pick_weighted([
        ("this is acceptable.", 2),
        ("i will adapt to this change.", 2),
        ("observation will continue.", 1),
        ("", 3),
    ])

    return join_sentences(starter, reaction, extra)

def crab_noise():
    sound = pick(SOUNDS)

    starter = pick_weighted([
        ("disturbance detected.", 3),
        ("the water vibrated.", 2),
        (f"that was a {sound}.", 2),
        (f"i sensed a {sound} through the water.", 2),
        ("something changed suddenly.", 1),
    ])

    reaction = pick([
        "vibrations indicate potential danger.",
        f"i moved to {pick_unique('spot', CRAB_SPOTS)}.",
        f"my {pick(BODY_PARTS)} became tense.",
        "rapid movement was required.",
        "this was not expected.",
        f"i was {pick_unique('act', ACTIVITIES)} and then stopped.",
        f"the {pick_unique('obj', CRAB_OBJECTS)} shifted slightly.",
        "i am assessing the situation.",
        "this environment is unstable for a moment.",
        "i will remain still and observe.",
    ])

    extra = pick_weighted([
        ("is it stable now.", 2),
        (f"my {pick(BODY_PARTS)} remain tense.", 1),
        ("i will stay cautious.", 2),
        ("", 3),
    ])

    return join_sentences(starter, reaction, extra)

def crab_night():
    starter = pick_weighted([
        ("night cycle detected.", 3),
        ("light levels are low.", 2),
        ("it is now dark.", 2),
        ("reduced visibility confirmed.", 1),
    ])

    middle = pick([
        "movement will decrease.",
        "i will remain still.",
        "activity is no longer required.",
        f"i will stay {pick_unique('spot', CRAB_SPOTS)}.",
        f"my {pick(BODY_PARTS)} are slowing.",
        "this is a low-energy state.",
        "awareness remains but reduced.",
        f"i was {pick_unique('act', ACTIVITIES)} but have stopped.",
        "this is an optimal time for stillness.",
        f"the {pick_unique('obj', CRAB_OBJECTS)} appears calmer now.",
    ])

    extra = pick_weighted([
        ("this state is acceptable.", 2),
        ("i will remain here until light returns.", 2),
        ("observation will continue at low intensity.", 1),
        ("", 3),
    ])

    return join_sentences(starter, middle, extra)

def crab_lonely():
    starter = pick_weighted([
        ("i exist alone in this environment.", 3),
        ("there are no other entities here.", 2),
        ("this space is occupied only by me.", 2),
        (f"i remain near the {pick_unique('obj', CRAB_OBJECTS)}. that is sufficient.", 1),
        ("solitude is normal here.", 2),
    ])

    middle = pick([
        "self-observation is adequate.",
        f"i remain occupied by {pick_unique('act', ACTIVITIES)}.",
        "additional entities would change conditions.",
        "this state is stable.",
        f"the {pick_unique('obj', CRAB_OBJECTS)} provides some structure.",
        f"the {pick(WATER_THINGS)} is consistent.",
        "interaction is not required for function.",
        f"i occasionally focus on {pick_unique('food', FOOD_TYPES)}.",
        f"my {pick(BODY_PARTS)} respond normally.",
        "this is a controlled environment.",
    ])

    extra = pick_weighted([
        ("this is acceptable.", 2),
        ("adaptation has occurred.", 2),
        ("no change required.", 1),
        ("", 3),
    ])

    return join_sentences(starter, middle, extra)

def crab_misc():
    starter = pick_weighted([
        ("i observed something.", 3),
        ("processing a thought.", 2),
        ("notable observation.", 2),
        ("something changed slightly.", 1),
        ("analysis in progress.", 1),
    ])

    observation = pick([
        "the water responds to movement.",
        "fluid enters and exits continuously.",
        "the boundary beyond is unclear.",
        f"i counted my {pick(BODY_PARTS)}. quantity is sufficient.",
        "bubbles rise consistently.",
        f"i detected my reflection near the {pick_unique('obj', CRAB_OBJECTS)}.",
        f"the water is {pick(WATER_DESCRIPTIONS)} when sampled.",
        f"i attempted to {pick(['move backwards', 'interact with a bubble', 'push an object', 'remain still'])}. results were limited.",
        f"the {pick_unique('obj', CRAB_OBJECTS)} appears different from {pick_unique('spot', CRAB_SPOTS)}.",
        f"perspective changes at {pick_unique('spot', CRAB_SPOTS)}.",
        f"priority remains {pick_unique('food', FOOD_TYPES)}.",
        f"my {pick(BODY_PARTS)} perform standard motion patterns.",
        f"the {pick(WATER_THINGS)} varies over time.",
        f"i located a small particle {pick_unique('spot', CRAB_SPOTS)}. it was not {pick_unique('food', FOOD_TYPES)}.",
        f"minor movement detected in my {pick(BODY_PARTS)}.",
        f"i observed the {pick_unique('obj', CRAB_OBJECTS)} for {random.randint(5, 30)} units.",
    ])

    return join_sentences(starter, observation)

# User message generators 

def user_greeting():
    return pick([
        "hello crab", "hi there", "hey", "hello", "hi buddy",
        "hey little crab", "hello there", "hi friend",
        "yo crab", "what's up", "hey you", "hello again",
        "hi hi", "greetings", "hey there", "sup crab",
        "morning crab", "evening crab", "hello hello",
    ])


def user_feeling():
    return pick([
        "how are you", "how are you feeling", "you ok",
        "are you doing fine", "what's your state",
        "everything stable", "how's it going",
        "you good", "all okay", "what's your mood",
        "how do you feel today", "everything alright",
        "are things normal", "status check",
    ])


def user_temp_hot():
    return pick([
        "it's hot today", "temperature is high", "it's really warm",
        f"it's around {random.randint(24, 42)} degrees",
        "feels like a furnace", "very hot outside",
        "the heat is intense", "it's boiling out here",
        "summer is brutal", "too much heat today",
    ])


def user_temp_cold():
    return pick([
        "it's cold today", "temperature dropped", "it's freezing",
        f"it's around {random.randint(-13, 8)} degrees",
        "very cold outside", "feels icy",
        "winter is harsh", "it's super chilly",
        "i'm freezing", "temperature is very low",
    ])


def user_food():
    return pick([
        "are you hungry", "time to eat", "want food",
        f"want some {pick(FOOD_TYPES)}",
        "feeding time", "food incoming",
        "ready to eat", "meal time",
        "i have food for you", "open up",
        "let's feed you", "hungry crab",
    ])


def user_light():
    return pick([
        "i changed the light", "it's brighter now",
        "it's darker now", "adjusted the lighting",
        "lights on", "lights off",
        "light is dim now", "light is strong",
        "i tweaked the light", "light looks different",
        "lighting updated", "light just changed",
    ])


def user_water():
    return pick([
        "how's the water", "i changed the water",
        "water looks clear", "water looks cloudy",
        "is the water okay", "fresh water added",
        "water seems different", "new water for you",
        "i cleaned the water", "water refreshed",
        "how does it feel", "water conditions good",
    ])


def user_about():
    return pick([
        "what are you", "describe yourself",
        "what do you do", "who are you",
        "tell me about yourself",
        "what are you exactly", "what's your purpose",
        "explain yourself", "what's your role",
    ])


def user_confused(thing=None):

    return pick([
        f"do you understand {thing}",
        f"what is {thing}",
        f"can you explain {thing}",
        f"do you know {thing}",
        f"what do you think about {thing}",
        f"have you heard of {thing}",
        f"tell me about {thing}",
        f"is {thing} important",
    ])


def user_environment():
    obj = pick(CRAB_OBJECTS)

    return pick([
        f"i added {obj}",
        f"there is a new {obj}",
        "the environment changed",
        "do you like the new setup",
        f"i placed {obj} there",
        f"new object: {obj}",
        "something new is in there",
        "check the new thing",
    ])


def user_noise():
    return pick([
        "that was loud", "did you hear that",
        "sorry for the noise", "there was a sound",
        "something made a noise",
        "that noise was big", "i dropped something",
        "oops loud sound", "hope that wasn't too loud",
        "big noise just happened",
    ])


def user_night():
    return pick([
        "goodnight", "time to rest",
        "sleep time", "lights out",
        "rest well",
        "night time", "time to sleep",
        "going to rest now", "end of day",
    ])


def user_lonely():
    return pick([
        "do you feel alone", "are you lonely",
        "do you want company", "is it boring there",
        "are you okay by yourself",
        "do you need a friend", "feels empty there",
        "are you fine alone", "do you want company",
    ])


def user_misc():
    return pick([
        "say something", "what are you thinking",
        "anything to report", "what's happening",
        "tell me something interesting",
        "give me an update", "what's going on",
        "any observations", "status report",
        "talk to me", "random thought",
    ])


def user_bye():
    return pick([
        "bye", "goodbye", "see you later",
        "talk later", "i'm leaving",
        "bye crab", "catch you later",
        "i'll be back", "leaving now",
    ])

def crab_bye():
    starters = [
        "bye.",
        "ok bye.",
        "see you.",
        "bye friend.",
        "later.",
        "goodbye.",
        "ok. bye bye.",
        "see you next time.",
    ]
    middles = [
        "i'll be here. swimming.",
        "i will continue being a crab.",
        f"bring {pick(FOOD_TYPES)} next time.",
        "the water will keep me company.",
        f"i'll go back to {pick(ACTIVITIES)}.",
        "don't forget about me. or the food.",
        f"my {pick(BODY_PARTS)} will miss you. maybe.",
        f"i'll eat some {pick(FOOD_TYPES)} if there's any left.",
    ]

    return join_sentences(pick(starters), pick(middles))

# 100 additional topics

# Each topic is a (user_fn, crab_fn, category) tuple.

def _make_sample(user, assistant, category):
    return {
        "input": user,
        "output": assistant,
        "type": category
    }

def _topic(user_fns, crab_fns, category):

    def gen():
        user_fn = pick(user_fns)
        crab_fn = pick(crab_fns)

        # call both sides properly
        user_msg = user_fn()
        
        try:
            response = crab_fn()
        except TypeError:
            # for functions like crab_food(hunger_level)
            response = crab_fn(0.5)

        return _make_sample(user_msg, response, category)

    gen.__name__ = f"gen_{category}"
    return gen

gen_bubbles = _topic(
    [
        lambda: "do you notice the bubbles",
        lambda: "look at the bubbles",
        lambda: "there are bubbles here",
        lambda: "do you interact with bubbles",
    ],
    [
        lambda: "bubbles rise consistently. i observe them.",
        lambda: "i move through them. they disrupt the water slightly.",
        lambda: "i attempted to interact. they collapse.",
        lambda: "they originate from below and move upward.",
        lambda: "bubbles alter the local current.",
        lambda: "i tracked one. it disappeared.",
        lambda: "they are unstable structures.",
        lambda: "i detect minor vibration when passing through them.",
    ],
    "bubbles",
)

gen_glass = _topic(
    [
        lambda: "can you see outside",
        lambda: "what do you see beyond",
        lambda: "are you looking at me",
        lambda: "do you notice the glass",
    ],
    [
        lambda: "the boundary is transparent but solid.",
        lambda: "i detect movement beyond it.",
        lambda: "external shapes are unclear but large.",
        lambda: "i sometimes approach the boundary.",
        lambda: "the outside appears dry. that is concerning.",
        lambda: "this environment ends at the boundary.",
    ],
    "glass",
)

gen_reflection = _topic(
    [
        lambda: "do you see yourself",
        lambda: "that's your reflection",
        lambda: "are you looking at yourself",
    ],
    [
        lambda: "i detected a duplicate entity. it mirrors me.",
        lambda: "the reflection behaves identically.",
        lambda: "i attempted interaction. no response.",
        lambda: "it is not a separate organism.",
        lambda: "visual feedback confirmed: it is me.",
    ],
    "reflection",
)

gen_breathing = _topic(
    [
        lambda: "how do you breathe",
        lambda: "are you breathing normally",
        lambda: "how does breathing work",
    ],
    [
        lambda: "water passes through my system continuously.",
        lambda: "oxygen extraction is automatic.",
        lambda: "breathing is constant and required.",
        lambda: "the water condition affects efficiency.",
        lambda: "this process does not stop.",
    ],
    "breathing",
)

gen_movement = _topic(
    [
        lambda: "do you move a lot",
        lambda: "how do you move",
        lambda: "you are moving fast",
    ],
    [
        lambda: "movement is constant.",
        lambda: "i reposition frequently.",
        lambda: "locomotion is efficient in this medium.",
        lambda: "i move to optimize conditions.",
        lambda: "stillness is also a strategy.",
    ],
    "movement",
)

gen_color = _topic(
    [
        lambda: "do you see colors",
        lambda: "what do things look like",
        lambda: "is everything colorful",
    ],
    [
        lambda: "light affects visibility.",
        lambda: "color perception varies with conditions.",
        lambda: "objects appear different under changing light.",
        lambda: "i detect contrast more than detail.",
        lambda: "visual input is limited but sufficient.",
    ],
    "color",
)

gen_environment = _topic(
    [
        lambda: "do you like this place",
        lambda: "what do you think of the environment",
        lambda: "is your space good",
    ],
    [
        lambda: "the environment is stable.",
        lambda: "objects provide structure.",
        lambda: "this area is sufficient.",
        lambda: "i have mapped most of this space.",
        lambda: "changes are noticeable immediately.",
    ],
    "environment",
)

gen_sound = _topic(
    [
        lambda: "did you hear that",
        lambda: "there was a noise",
        lambda: "was that loud",
    ],
    [
        lambda: "vibrations were detected.",
        lambda: "the disturbance propagated through the water.",
        lambda: "i reacted immediately.",
        lambda: "sudden changes require caution.",
        lambda: "the system stabilized afterward.",
    ],
    "sound",
)

def gen_greeting():
    return _make_sample(user_greeting(), crab_greeting(), "greeting")

def gen_feeling():
    return _make_sample(user_feeling(), crab_feeling(), "feeling")

def gen_temp_hot():
    return _make_sample(user_temp_hot(), crab_temp_hot(), "temp_hot")

def gen_temp_cold():
    return _make_sample(user_temp_cold(), crab_temp_cold(), "temp_cold")

def gen_food():
    return _make_sample(user_food(), crab_food(), "food")

def gen_light():
    return _make_sample(user_light(), crab_light(), "light")

def gen_water():
    return _make_sample(user_water(), crab_water(), "water")

def gen_about():
    return _make_sample(user_about(), crab_about(), "about")

def gen_confused():
    return _make_sample(user_confused(), crab_confused(), "confused")

def gen_noise():
    return _make_sample(user_noise(), crab_noise(), "noise")

def gen_night():
    return _make_sample(user_night(), crab_night(), "night")

def gen_lonely():
    return _make_sample(user_lonely(), crab_lonely(), "lonely")

def gen_misc():
    return _make_sample(user_misc(), crab_misc(), "misc")

def gen_bye():
    return _make_sample(user_bye(), crab_bye(), "bye")

def format_sample(s):
    return f"User: {s['input']}\nCrab: {s['output']}"


def to_openai(s):
    return {
        "messages": [
            {"role": "user", "content": s["input"]},
            {"role": "assistant", "content": s["output"]},
        ]
    }


def generate_dataset(n_samples=100000, eval_ratio=0.05):

    # ── Weighted topics ──
    generators = [
        # Core personality (high)
        (gen_greeting, 3),
        (gen_feeling, 3),
        (gen_misc, 3),

        # Common interaction (medium)
        (gen_food, 2),
        (gen_light, 2),
        (gen_water, 2),
        (gen_environment, 2),
        (gen_confused, 2),

        # Situational (low)
        (gen_temp_hot, 1),
        (gen_temp_cold, 1),
        (gen_noise, 1),
        (gen_night, 1),
        (gen_lonely, 1),
        (gen_bye, 1),

        # World / perception (low-medium)
        (gen_bubbles, 1),
        (gen_glass, 1),
        (gen_reflection, 1),
        (gen_breathing, 1),
        (gen_movement, 1),
        (gen_color, 1),
        (gen_sound, 1),
    ]

    total_w = sum(w for _, w in generators)
    generators = [(g, w / total_w) for g, w in generators]
    counts = [(g, max(1, int(n_samples * w))) for g, w in generators]
    total = sum(c for _, c in counts)
    while total < n_samples:
        g = random.choice(counts)
        idx = counts.index(g)
        counts[idx] = (g[0], g[1] + 1)
        total += 1

    samples = []

    # Generate data
    for gen, count in counts:
        for _ in range(count):
            try:
                samples.append(gen())
            except Exception as e:
                print(f"Error in {gen.__name__}: {e}")

    random.shuffle(samples)

    # Split
    n_eval = int(len(samples) * eval_ratio)
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    os.makedirs("data", exist_ok=True)

    # ── Standard format ──
    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open("data/eval.jsonl", "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s) + "\n")

    # ── OpenAI format ──
    with open("data/train_openai.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(to_openai(s)) + "\n")

    with open("data/eval_openai.jsonl", "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(to_openai(s)) + "\n")

    # ── Stats ──
    cats = Counter(s["type"] for s in samples)
    unique_outputs = len(set(s["output"] for s in samples))

    print(f"\nGenerated {len(samples)} samples")
    print(f"Unique outputs: {unique_outputs} ({unique_outputs/len(samples)*100:.1f}%)")
    print(f"Train: {len(train_samples)} | Eval: {len(eval_samples)}\n")

    print("By category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(samples)*100:.1f}%)")


if __name__ == "__main__":
    generate_dataset(100000)