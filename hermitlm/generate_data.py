import json
import random
import os
from collections import Counter

random.seed(617)

def pick(lst):
    return random.choice(lst)

def pick_n(lst, n):
    return random.choice(lst, min(n, len(lst)))

def maybe(text, p=0.5):
    return text if random.random() < p else ""

# Things Hermit sees or interacts with

MINECRAFT = [
    "village", "blacksmith", "hay bale", "bed", "iron golem", "bucket",
    "lava pool", "nether portal", "fortress", "blaze spawner", "blaze rod",
    "bastion", "piglin", "gold block", "ender pearl", "warped forest",
    "stronghold", "library", "portal room", "end portal", "eye of ender",
    "dragon perch", "obsidian tower", "crystal", "water clutch",
    "boat clutch", "blind travel", "triangulation", "spawn",
]

MCSR_OBJECTS = [
# Basic blocks
"stone", "cobblestone", "mossy cobblestone", "dirt", "grass block",
"sand", "red sand", "gravel", "clay", "snow", "ice", "packed ice",
"blue ice", "obsidian", "bedrock", "netherrack", "soul sand",
"soul soil", "blackstone", "basalt", "smooth basalt", "end stone",

# Ores and valuables
"coal ore", "iron ore", "gold ore", "diamond ore", "emerald ore",
"redstone ore", "lapis ore", "nether gold ore", "ancient debris",
"diamond block", "gold block", "iron block", "emerald block",

# Wood and overworld structures
"oak log", "birch log", "spruce log", "jungle log", "acacia log",
"dark oak log", "mangrove log", "cherry log", "oak plank",
"spruce plank", "birch plank", "crafting table", "furnace",
"smoker", "blast furnace", "anvil", "grindstone", "stonecutter",
"cartography table", "fletching table", "smithing table",
"enchanting table", "bookshelf", "lectern", "barrel", "loom",

# Containers
"chest", "trapped chest", "ender chest", "shulker box", "hopper",
"dropper", "dispenser", "item frame", "armor stand",

# Redstone
"redstone torch", "redstone lamp", "lever", "button", "pressure plate",
"observer", "piston", "sticky piston", "repeater", "comparator",
"tripwire", "daylight sensor", "target block", "note block",

# Mobility
"boat", "chest boat", "minecart", "chest minecart",
"furnace minecart", "powered rail", "detector rail", "rail",
"ladder", "vine", "scaffolding", "water bucket", "lava bucket",

# Utility / speedrun items
"flint and steel", "bucket", "water", "lava", "bed",
"ender pearl", "eye of ender", "blaze rod", "blaze powder",
"golden apple", "enchanted golden apple", "shield", "bow",
"crossbow", "fishing rod", "clock", "compass",

# Nether
"nether portal", "crying obsidian", "respawn anchor",
"nether wart", "magma block", "glowstone", "quartz block",
"piglin", "zombified piglin", "hoglin", "strider",
"blaze", "ghast", "magma cube", "wither skeleton",

# Stronghold / End
"stronghold", "library", "end portal", "end portal frame",
"silverfish spawner", "end stone", "obsidian pillar",
"dragon egg", "end crystal", "end gateway",

# Villages and structures
"village house", "blacksmith", "desert temple", "jungle temple",
"shipwreck", "ocean monument", "woodland mansion",
"pillager outpost", "ruined portal", "bastion remnant",
"nether fortress", "ancient city", "trial chamber",

# Mobs
"zombie", "skeleton", "creeper", "spider", "enderman",
"witch", "slime", "warden", "villager", "iron golem",
"pig", "cow", "sheep", "chicken", "horse", "wolf",
"cat", "axolotl", "frog", "camel",

# Terrain / biomes
"mountain", "river", "ocean", "lava pool", "cave entrance",
"ravine", "waterfall", "forest", "desert", "savanna",
"snow biome", "mushroom island", "badlands", "swamp",
"deep dark", "lush cave",

# Misc speedrun visuals
"hay bale", "slime block", "tnt", "campfire", "lantern",
"torch", "soul torch", "bell", "cake", "beacon",
"totem of undying", "map", "spyglass", "brush",

]

CRAB_OBJECTS = [
"sand block", "gravel patch", "stone", "cobblestone", "mossy stone",
"seaweed", "kelp", "coral block", "coral fan", "dead coral",
"shell", "giant shell", "tiny shell", "driftwood", "log",
"shipwreck", "treasure chest", "buried chest", "ruins",
"cave entrance", "small cave", "underwater ruin", "arch",
"pillar", "rock formation", "bubble column", "magma block",
"soul sand", "glass block", "sandstone", "clay patch",
]

CRAB_SPOTS = [
"near the sand", "under the gravel", "beside the stone",
"behind the coral", "inside the shell", "near the kelp",
"at the cave entrance", "deep in the cave", "by the ruins",
"next to the treasure chest", "under the driftwood",
"between the rocks", "along the seabed", "near the bubble column",
"on top of the sand block", "in my hiding spot",
"where the water flows gently", "where the light reaches",
"at the edge of the trench", "near the magma block",
]

FOOD_TYPES = [
"kelp bits", "algae", "tiny fish", "plankton",
"moss scraps", "sea pickles", "dead fish bits",
"whatever that floating thing is", "mysterious crumbs",
"bubble snacks", "sand snacks", "soft algae",
"crunchy bits", "sinking food", "floating food",
]

WATER_DESCRIPTIONS = [
"clear", "murky", "cold", "warm", "just right",
"a bit cloudy", "salty", "fresh-ish", "calm",
"rough", "gentle", "swirling", "still",
]

ACTIVITIES = [
"scuttling sideways", "digging in the sand",
"hiding in my shell", "peeking out carefully",
"watching bubbles rise", "climbing a rock",
"exploring the cave", "guarding my spot",
"pretending to be sand", "snipping at nothing",
"chasing a tiny fish", "investigating debris",
"burying myself halfway", "staring dramatically",
"clicking my claws", "doing nothing but looking busy",
"moving one inch and stopping", "suspiciously observing everything",
]

FEELINGS = [
"good", "fine", "calm", "hungry", "suspicious",
"grumpy", "content", "alert", "sleepy",
"territorial", "curious", "peaceful",
]

WATER_THINGS = [
"current", "temperature", "bubbles",
"pressure", "flow", "salinity",
"warmth", "coldness",
]

LIGHT_STATES = [
"dim", "bright", "blue", "greenish",
"murky", "shimmering", "flickering",
"soft", "harsh",
]

TIMES_OF_DAY = ["morning", "afternoon", "evening", "night"]

BODY_PARTS = [
"claws", "legs", "shell", "eyes",
"antennae", "little legs", "hard shell",
]

SOUNDS = [
"click", "clack", "tap", "scrape",
"rumble", "bubble", "crunch",
"thud", "echo",
]

HUMAN_THINGS = [
"mining", "crafting", "redstone", "a pickaxe",
"a sword", "diamonds", "villagers", "trading",
"a furnace", "smelting", "an anvil",
"a creeper", "a zombie", "a skeleton",
"a nether portal", "the end", "the dragon",
"a map", "a compass", "an enchantment table",
"a chest", "inventory", "XP", "levels",
]


def _join(*parts):
    return " ".join([p for p in parts if p]).strip()

def _crab_greeting():
    openers = [
        "oh. it's you.", "hello.", "hi.", "you again.",
        "i noticed you.", "i was watching you already.",
    ]

    middles = [
        f"i was just {pick(ACTIVITIES)}.",
        f"i am {pick(CRAB_SPOTS)}.",
        f"the water is {pick(WATER_DESCRIPTIONS)} today.",
        f"the {pick(CRAB_OBJECTS)} looks different today.",
        f"i thought you might bring {pick(FOOD_TYPES)}.",
        f"i heard a {pick(SOUNDS)} earlier.",
        "i don't trust large shapes but you seem consistent.",
        f"i was just {pick(ACTIVITIES)}.",
        "i didn't see you there. my eyes are on the sides.",
        "are you the big shape that feeds me.",
        "you look big today. you always look big.",
        f"i was thinking about {pick(FOOD_TYPES)}.",
        f"did you bring {pick(FOOD_TYPES)}.",
    ]

    extras = [
        f"the {pick(WATER_THINGS)} feels stable.",
        f"i will remain {pick(CRAB_SPOTS)} for now.",
        "nothing unusual has happened. that is good.",
        "i am observing.",
        "",
        "",
    ]

    return _join(pick(openers), pick(middles), pick(extras))

def _crab_feeling():
    starters = [
        f"i feel {pick(FEELINGS)}.",
        f"{pick(FEELINGS)}. that is my state.",
        f"currently {pick(FEELINGS)}.",
        f"i am {pick(FEELINGS)} right now.",
    ]

    reasons = [
        f"the water is {pick(WATER_DESCRIPTIONS)}.",
        f"i was {pick(ACTIVITIES)}.",
        f"the {pick(WATER_THINGS)} is steady.",
        f"i secured a spot {pick(CRAB_SPOTS)}.",
        f"my {pick(BODY_PARTS)} feel functional.",
        f"the light is {pick(LIGHT_STATES)}.",
        "nothing has tried to eat me recently.",
        "i have not been disturbed.",
        f"i am thinking about {pick(FOOD_TYPES)}.",
    ]

    closers = [
        "this is acceptable.",
        "i will continue like this.",
        "",
        "",
    ]

    return _join(pick(starters), pick(reasons), pick(closers))

def _crab_temp_hot():
    starters = [
        "the water is too warm.",
        "temperature is rising.",
        "this is not ideal.",
        "warm conditions detected.",
    ]

    middles = [
        "oxygen will decrease.",
        "movement becomes inefficient.",
        f"i am relocating to {pick(CRAB_SPOTS)}.",
        f"my {pick(BODY_PARTS)} feel slower.",
        "i will conserve energy.",
        "this environment is unstable.",
    ]

    extras = [
        "adjustment is required.",
        "you should correct this.",
        "i am depending on external intervention.",
        "",
        "",
    ]

    return _join(pick(starters), pick(middles), pick(extras))

def _crab_temp_cold():
    starters = [
        "the water is colder.",
        "temperature has dropped.",
        "cold conditions detected.",
        "this is manageable.",
    ]

    middles = [
        "oxygen levels are better.",
        "movement is slower but controlled.",
        f"i will stay {pick(CRAB_SPOTS)}.",
        f"my {pick(BODY_PARTS)} are less responsive.",
        "energy usage is reduced.",
        f"i stopped {pick(ACTIVITIES)}.",
    ]

    extras = [
        "this is acceptable for now.",
        "i will adapt.",
        "do not let it drop further.",
        "",
        "",
    ]

    return _join(pick(starters), pick(middles), pick(extras))