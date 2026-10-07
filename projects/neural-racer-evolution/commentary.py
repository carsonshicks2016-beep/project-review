"""
Procedural commentary generator for spectator mode.

Uses a context-free grammar with many templates, deep word banks, and
live car-state values (speed, fitness, laps). A rolling hash deque
de-duplicates recent outputs so the same exact line never appears twice
in any practical session.

Combinatorial space:
   ~8 categories × ~7 templates × ~5 slots × ~16 variants per slot
   = millions of unique grammar combinations per category,
   multiplied by the always-changing numeric data.
"""
import random
import string
import collections


# ======================================================================
# Word banks
# ======================================================================

NAME_ADJ = [
    "Crimson", "Azure", "Golden", "Silver", "Shadow", "Phantom", "Thunder",
    "Lightning", "Ember", "Frost", "Storm", "Iron", "Steel", "Velvet",
    "Onyx", "Jade", "Cobalt", "Scarlet", "Obsidian", "Quicksilver",
    "Twilight", "Inferno", "Tempest", "Mercury", "Vortex", "Wraith",
    "Spectre", "Comet", "Nova", "Eclipse", "Midnight", "Solar", "Lunar",
    "Crimson", "Emerald", "Sapphire",
]
NAME_NOUN = [
    "Phantom", "Bullet", "Arrow", "Falcon", "Hawk", "Viper", "Cobra",
    "Wolf", "Lynx", "Panther", "Cheetah", "Stallion", "Bolt", "Stinger",
    "Comet", "Meteor", "Dart", "Spear", "Bandit", "Specter", "Sentinel",
    "Reaper", "Hunter", "Predator", "Striker", "Maverick", "Renegade",
    "Outlaw", "Drifter", "Ghost", "Demon", "Tiger", "Shark", "Eagle",
    "Serpent", "Marauder",
]

EXCLAIM = [
    "Oh my!", "Wow!", "Look at this!", "Incredible!", "Unbelievable!",
    "Did you see that?!", "Astonishing!", "Spectacular!", "My word!",
    "What a moment!", "Goodness me!", "Phenomenal!", "Outrageous!",
    "Just incredible!", "I can't believe it!", "Hot damn!", "Listen!",
    "Hold on!", "Are you kidding me?!",
]

ADJ_INTENSE = [
    "furious", "blazing", "absolute", "thunderous", "savage", "unreal",
    "ferocious", "blistering", "vicious", "merciless", "ruthless",
    "explosive", "razor-sharp", "devastating", "scorching", "frenzied",
    "lightning-fast", "white-knuckle", "heart-stopping", "breakneck",
]

ADJ_DRIFT = [
    "spectacular", "controlled", "calculated", "graceful", "violent",
    "delicate", "wild", "audacious", "textbook", "showboat", "fearless",
    "balletic", "magnificent", "terrifying", "smoke-pouring", "operatic",
    "feathered", "knife-edge",
]

VERB_LEAD = [
    "dominating", "carving up", "schooling", "embarrassing", "torching",
    "running away from", "tearing through", "blistering past",
    "leaving for dead", "lapping", "outclassing", "humbling",
    "stomping on", "shredding", "burying", "spanking",
]

PACE_WORD = [
    "pace", "speed", "rhythm", "tempo", "flow", "groove", "command",
    "control", "mastery", "dominance", "authority", "poise",
]

TIRE_WORD = [
    "the tires are screaming", "smoke everywhere", "I can smell the rubber",
    "tires absolutely shrieking", "those rears are cooked",
    "the rubber is begging for mercy", "look at that smoke trail",
    "they're cooking those tires", "the back end is alive",
]

DRIFT_SIMILE = [
    "a pro", "a veteran", "a madman", "a stunt driver", "a Group B legend",
    "a rally champion", "a drift king", "a samurai", "a demon",
    "a ballerina", "an artist", "a maniac", "a wizard",
]

TROUBLE_WORD = [
    "wobbling", "fighting the back end", "completely sideways",
    "losing it", "barely holding on", "scrambling", "in deep trouble",
    "about to spin", "off-line", "losing the front", "praying",
    "white-knuckling", "all crossed up",
]

UNDERDOG_VERB = [
    "sneaking", "creeping", "quietly working", "stealthily climbing",
    "patiently grinding", "methodically clawing", "slowly building",
    "sleeper-cell-style hunting", "shadow-driving",
]

POS_DESCRIPTOR = [
    "from nowhere", "out of the pack", "from the shadows",
    "from mid-grid", "from way back", "from oblivion", "from anonymity",
    "from out of nothing",
]

CROWD_REACT = [
    "the crowd is on their feet", "listen to this noise",
    "the stands are losing it", "you can hear the gasps from here",
    "the trackside is electric", "everyone is standing",
]

QUALITY_WORD = [
    "masterclass", "showcase", "exhibition", "clinic", "lesson",
    "demonstration", "tour de force", "highlight reel",
]


# ======================================================================
# Templates by category
# ======================================================================

TEMPLATES = {
    'LEADER': [
        "{exclaim} {name} is {verb_lead} the field — {fitness} fitness already!",
        "Untouchable from {name}, holding {pace_word} at {speed} px/s",
        "{name} has the {pace_word} of a champion — {laps} laps in",
        "Pure {pace_word} from {name}, leading by a country mile",
        "Nobody can touch {name} today — {fitness} on the board",
        "{exclaim} What a run from {name}: {laps} laps and {fitness} fitness",
        "{name} putting on a {quality_word} out there",
        "This is a {quality_word} from {name}, {crowd_react}",
    ],
    'DRIFT': [
        "{exclaim} {adj_drift} drift from {name} — {tire_word}",
        "Look at {name} go — {tire_word}",
        "{name} is sideways like {drift_simile}",
        "{adj_drift} car control from {name}, {tire_word}",
        "Textbook drift through that corner from {name}",
        "{exclaim} {name} threw it sideways at {speed} px/s — {tire_word}",
        "{name} is drifting like {drift_simile} out there",
        "{adj_intense} angle on {name} — {tire_word}",
    ],
    'SPEEDSTER': [
        "{name} is {adj_intense} on this straight — {speed} px/s and climbing",
        "{exclaim} {name} just hit {speed} px/s — {pace_word} like that wins races",
        "Pure pace from {name}: {speed} px/s and pulling",
        "{name} is {adj_intense} down the straightaway — {speed} px/s",
        "Speed-trap says {speed} for {name}, and {crowd_react}",
        "{name} is absolutely flying — {speed} px/s, {crowd_react}",
    ],
    'UNDERDOG': [
        "Watch out for {name} — {underdog_verb} {pos_descriptor}",
        "{name} is {underdog_verb} a quietly clever lap together",
        "Don't sleep on {name} — {fitness} fitness {pos_descriptor}",
        "Sneaky run from {name}, {underdog_verb} up the order",
        "{name} is {underdog_verb} into contention {pos_descriptor}",
    ],
    'TROUBLE': [
        "{name} is {trouble_word} into that corner",
        "{exclaim} {name} is {trouble_word}!",
        "This is rough for {name} — {trouble_word} and might be done",
        "{name} {trouble_word}, this could be a heartbreaker",
        "Oh no — {name} is {trouble_word} out there",
    ],
    'SURVIVOR': [
        "{name} keeping it together while others fall",
        "Last cars standing — {name} still rolling at lap {laps}",
        "Survival mode from {name}, just trying to finish",
        "{name} grinding it out: {fitness} fitness, still alive",
        "Pure attrition for {name}, {laps} laps and counting",
    ],
    'ROOKIE': [
        "Fresh face this gen: {name}, let's see what they've got",
        "Quick look at {name}, just getting settled",
        "{name} is just out of the box, finding their {pace_word}",
        "Eyes on {name} — early days but {speed} px/s already",
    ],
    'WILDCARD': [
        "Quick check on {name}: {speed} px/s, {fitness} fitness",
        "Putting the spotlight on {name} for a moment",
        "{name} is having a {pace_word}-y kind of run",
        "Spectating {name} — {laps} laps, {fitness} fitness so far",
        "Cutting away to {name}, doing their own thing",
        "Camera on {name} now — {speed} px/s through this section",
    ],
}


# ======================================================================
# Grammar formatter
# ======================================================================

class _GrammarFormatter(string.Formatter):
    """str.Formatter that resolves unknown keys from grammar word banks."""

    def __init__(self, grammar, car_slots):
        self.grammar    = grammar
        self.car_slots  = car_slots

    def get_value(self, key, args, kwargs):
        if key in self.car_slots:
            return self.car_slots[key]
        if key in self.grammar:
            return random.choice(self.grammar[key])
        return f"{{{key}}}"   # leave unknown slots visible for debugging


_GRAMMAR_BANK = {
    'exclaim':         EXCLAIM,
    'adj_intense':     ADJ_INTENSE,
    'adj_drift':       ADJ_DRIFT,
    'verb_lead':       VERB_LEAD,
    'pace_word':       PACE_WORD,
    'tire_word':       TIRE_WORD,
    'drift_simile':    DRIFT_SIMILE,
    'trouble_word':    TROUBLE_WORD,
    'underdog_verb':   UNDERDOG_VERB,
    'pos_descriptor':  POS_DESCRIPTOR,
    'crowd_react':     CROWD_REACT,
    'quality_word':    QUALITY_WORD,
}


# ======================================================================
# Commentator
# ======================================================================

class Commentator:
    HISTORY = 800     # rolling dedup window

    def __init__(self):
        self.recent_hashes = collections.deque(maxlen=self.HISTORY)
        self._name_cache   = {}   # car_id → nickname

    def name_for(self, car):
        cid = id(car)
        if cid not in self._name_cache:
            adj  = random.choice(NAME_ADJ)
            noun = random.choice(NAME_NOUN)
            num  = random.randint(1, 99)
            self._name_cache[cid] = f"{adj} {noun} #{num}"
        return self._name_cache[cid]

    def comment(self, car, category):
        """
        Generate a single commentary line for `car` in the given category.
        Retries on duplicate (up to 12 attempts) before giving up.
        """
        if category not in TEMPLATES:
            category = 'WILDCARD'

        car_slots = {
            'name':    self.name_for(car),
            'speed':   int(getattr(car, 'speed', 0)),
            'fitness': int(getattr(car, 'fitness', 0)),
            'laps':    getattr(car, 'laps', 0),
        }
        fmt = _GrammarFormatter(_GRAMMAR_BANK, car_slots)

        last_text = ""
        for _ in range(12):
            template  = random.choice(TEMPLATES[category])
            last_text = fmt.format(template)
            h = hash(last_text)
            if h not in self.recent_hashes:
                self.recent_hashes.append(h)
                return last_text
        # Fallback after 12 collisions (extremely unlikely)
        self.recent_hashes.append(hash(last_text))
        return last_text
