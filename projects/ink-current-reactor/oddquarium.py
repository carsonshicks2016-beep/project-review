#!/usr/bin/env python3
"""
ODDQUARIUM - a small terminal aquarium with too much personality.

Run:
    python3 oddquarium.py

Useful non-interactive commands:
    python3 oddquarium.py --ideas
    python3 oddquarium.py --snapshot --seed 12
"""

from __future__ import annotations

import argparse
import curses
import datetime as _dt
import json
import math
import os
import random
import sys
import textwrap
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable


SAVE_FILE = os.path.expanduser("~/.oddquarium.json")
FRAME_DELAY = 0.045
MIN_WIDTH = 54
MIN_HEIGHT = 17


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def irand(low: int, high: int) -> int:
    if high < low:
        return low
    return random.randint(low, high)


def choice(seq):
    return random.choice(tuple(seq))


def draw_text(win, y: int, x: int, text: str, color: int = 7, attr: int = 0) -> None:
    if not text:
        return
    try:
        height, width = win.getmaxyx()
        if y < 0 or y >= height or x >= width:
            return
        if x < 0:
            text = text[-x:]
            x = 0
        if not text:
            return
        win.addstr(y, x, text[: max(0, width - x)], curses.color_pair(color) | attr)
    except curses.error:
        pass


def draw_ch(win, y: int, x: int, ch: str, color: int = 7, attr: int = 0) -> None:
    try:
        height, width = win.getmaxyx()
        if 0 <= y < height and 0 <= x < width:
            win.addch(y, x, ch, curses.color_pair(color) | attr)
    except curses.error:
        pass


@dataclass(frozen=True)
class Role:
    key: str
    label: str
    color: int
    phrases: tuple[str, ...]
    snack_phrases: tuple[str, ...]


ROLES: tuple[Role, ...] = (
    Role(
        "mayor",
        "Mayor",
        3,
        (
            "As mayor, I declare this pebble historic.",
            "The tank budget is mostly bubbles.",
            "I support every fish, especially voters.",
            "My platform: snacks, dignity, looser gravel.",
        ),
        (
            "A municipal snack. Very official.",
            "This flake has my endorsement.",
        ),
    ),
    Role(
        "poet",
        "Tiny Poet",
        5,
        (
            "A bubble rises. I remain complicated.",
            "The kelp bends like a comma.",
            "I wrote a poem called Splash, Then Silence.",
            "My muse is that weird corner.",
        ),
        (
            "This tastes like a metaphor.",
            "A crumb descends. I forgive everything.",
        ),
    ),
    Role(
        "accountant",
        "Accountant",
        6,
        (
            "I have reconciled the bubble ledger.",
            "We are over budget on dramatic exits.",
            "Please submit receipts for all flakes.",
            "I count grains of sand for compliance.",
        ),
        (
            "Logging this as snack income.",
            "This flake rounds up nicely.",
        ),
    ),
    Role(
        "dj",
        "DJ",
        4,
        (
            "Drop the bass. Carefully. We live in glass.",
            "I only play deep cuts. Very deep.",
            "The filter has a wild kick drum.",
            "Everybody look busy, then dance.",
        ),
        (
            "Snack remix incoming.",
            "That crunch had tempo.",
        ),
    ),
    Role(
        "lawyer",
        "Tiny Lawyer",
        1,
        (
            "Objection: insufficient flakes.",
            "My client was merely swimming with intent.",
            "I move to strike that algae from the record.",
            "The glass is both witness and wall.",
        ),
        (
            "Exhibit A: delicious.",
            "I accept this settlement flake.",
        ),
    ),
    Role(
        "intern",
        "Intern",
        2,
        (
            "Is this actionable or just wet?",
            "I made copies of the current.",
            "Quick question: am I doing fish right?",
            "Happy to circle back after lunch.",
        ),
        (
            "Great, lunch is vertical today.",
            "Can I expense this?",
        ),
    ),
    Role(
        "cryptid",
        "Local Legend",
        7,
        (
            "You saw nothing.",
            "Some say I appear near the filter. Some are correct.",
            "I am not blurry. I am mysterious.",
            "The documentary crew never found me.",
        ),
        (
            "Legends require nutrition.",
            "This never happened.",
        ),
    ),
    Role(
        "oracle",
        "Snack Oracle",
        5,
        (
            "The next flake will fall from above. Bold prophecy.",
            "I have read the bubbles. Mixed reviews.",
            "Ask again when the snail passes.",
            "The water remembers. It mostly remembers being wet.",
        ),
        (
            "The prophecy is crunchy.",
            "As foretold: snack.",
        ),
    ),
    Role(
        "barista",
        "Bubble Barista",
        3,
        (
            "One oxygen macchiato for the corner table.",
            "The foam art is emotionally specific.",
            "We are out of oat water.",
            "Your name on the cup is probably Gerald.",
        ),
        (
            "Pairing notes: plankton, nostalgia.",
            "Complimentary crumble.",
        ),
    ),
)

ROLE_BY_KEY = {role.key: role for role in ROLES}


MOODS: tuple[str, ...] = (
    "chill",
    "dramatic",
    "snacky",
    "suspicious",
    "inspired",
    "sleepy",
    "ambitious",
    "sparkly",
)

GENERAL_PHRASES: tuple[str, ...] = (
    "This tank has lore.",
    "I have accepted my rectangle.",
    "The gravel is arranged in a suspicious way.",
    "I know every inch of this place and still get lost.",
    "Someone should install a tiny chandelier.",
    "I would like a window with a view of another window.",
    "The filter is humming in lowercase.",
    "No thoughts, just buoyancy.",
    "The cave is fake, but my feelings are real.",
    "I saw my reflection and we are not speaking.",
    "If you need me, I am conducting a lap.",
    "I am emotionally available, but only at feeding time.",
    "The bubbles keep leaving. Brave.",
    "I am not late. I am moving at aquatic speed.",
    "Today feels sponsored by dampness.",
)

MOOD_PHRASES: dict[str, tuple[str, ...]] = {
    "chill": (
        "Current status: pleasantly suspended.",
        "Everything is fine. Suspiciously fine.",
    ),
    "dramatic": (
        "I require a spotlight and a small curtain.",
        "This moment deserves thunder.",
    ),
    "snacky": (
        "I am hearing rumors of crumbs.",
        "My stomach has entered the meeting.",
    ),
    "suspicious": (
        "The castle decoration knows too much.",
        "That bubble went up. Too convenient.",
    ),
    "inspired": (
        "I have an idea and it is probably a lap.",
        "I am inventing a new kind of swimming.",
    ),
    "sleepy": (
        "Wake me if the snacks become interesting.",
        "I am resting my fins professionally.",
    ),
    "ambitious": (
        "Five year plan: bigger tank, smaller problems.",
        "I am optimizing my personal brand.",
    ),
    "sparkly": (
        "Something about today has tiny cymbals.",
        "I am legally glitter-adjacent.",
    ),
}

EVENT_CHATTER: dict[str, tuple[str, ...]] = {
    "board": (
        "Motion to rename the castle.",
        "Can we table the table?",
        "I second the snack proposal.",
        "Minutes will be kept in bubbles.",
    ),
    "audit": (
        "Hide the unfiled crumbs.",
        "My paperwork is wet.",
        "I plead the fifth flake.",
        "Nobody mention petty kelp.",
    ),
    "poetry": (
        "Ahem. Water.",
        "My piece is called Please Clap.",
        "Snaps, but underwater.",
        "This poem has seven endings.",
    ),
    "disco": (
        "I was born for this lighting.",
        "The floor is sand and the vibes are real.",
        "I request more bass and fewer walls.",
        "My fins know choreography.",
    ),
    "moon": (
        "The moon is pulling my calendar.",
        "I feel tidal, emotionally.",
        "Everything is a little sideways.",
        "The water has a dramatic opinion.",
    ),
    "mirror": (
        "Why is everyone doing my thing?",
        "I have become a trend.",
        "This is either art or concerning.",
        "I do not consent to being iconic.",
    ),
    "sub": (
        "Is that a tiny inspector?",
        "Act natural. Nobody knows how.",
        "I waved with the wrong fin.",
        "That thing has headlights. Fancy.",
    ),
    "oracle": (
        "The bubbles have footnotes.",
        "I accept this extremely vague wisdom.",
        "The future smells like flakes.",
        "I knew this would happen after it started.",
    ),
}

IDEAS: tuple[str, ...] = (
    "Fish with job titles: mayor, accountant, tiny lawyer, poet, DJ, local legend.",
    "Events that feel like tiny sitcom episodes: tax audit, poetry slam, moon tide.",
    "A snail courier that delivers notes and starts rumors.",
    "A spotlight panel so one fish feels like the star of a documentary.",
    "Mood-driven dialogue instead of one giant random quote pile.",
    "Keyboard-first controls, with mouse clicks for food and glass boops.",
    "Snapshot mode so you can preview it even outside a real terminal.",
    "A future deluxe version could add quests, rivalries, birthdays, and fish resumes.",
)

FIRST_NAMES: tuple[str, ...] = (
    "Miso",
    "Pickle",
    "Biscuit",
    "Mango",
    "Kevin",
    "Darla",
    "Noodle",
    "Taffy",
    "Gouda",
    "Waffles",
    "Pesto",
    "Tuna",
    "Pixel",
    "Soup",
    "Velcro",
    "Toast",
    "Mabel",
    "Juno",
    "Fritter",
    "Disco",
)

LAST_NAMES: tuple[str, ...] = (
    "Bubbleton",
    "Finch",
    "McSplash",
    "the Damp",
    "Pebblewise",
    "of Accounting",
    "Currently",
    "Glassfriend",
    "Snackwell",
    "Kelpfield",
    "Drift",
    "Fizz",
)

FISH_FORMS: tuple[tuple[str, str], ...] = (
    ("><(((o>", "<o)))><"),
    ("><((o>", "<o))><"),
    ("><(o>", "<o)><"),
    (">=><>", "<><=<"),
    (">=>", "<=<"),
    ("~><>", "<><~"),
    (">>---->", "<----<<"),
    ("><{{o>", "<o}}><"),
    (">-<*>", "<*>-<"),
    ("><o>", "<o><"),
    (">><>", "<><<"),
    ("}i{", "}i{"),
)


@dataclass
class Particle:
    fx: float
    fy: float
    dx: float
    dy: float
    char: str
    color: int
    life: int
    bold: bool = False

    def update(self, width: int, height: int) -> bool:
        self.fx += self.dx
        self.fy += self.dy
        self.life -= 1
        return self.life > 0 and -2 < self.fx < width + 2 and -2 < self.fy < height + 2

    def draw(self, win) -> None:
        attr = curses.A_BOLD if self.bold else curses.A_DIM
        draw_ch(win, int(self.fy), int(self.fx), self.char, self.color, attr)


@dataclass
class FoodFlake:
    fx: float
    fy: float
    char: str
    color: int
    speed: float
    drift: float
    eaten: bool = False
    bob: float = 0.0

    def update(self, width: int, height: int) -> bool:
        self.bob += 0.18
        self.fy += self.speed
        self.fx += self.drift + math.sin(self.bob) * 0.03
        floor = max(4, height - 4)
        if self.fy >= floor:
            self.fy = float(floor)
            self.speed = 0.0
            self.drift *= 0.82
        return not self.eaten and 1 < self.fx < width - 2

    def draw(self, win) -> None:
        draw_ch(win, int(self.fy), int(self.fx), self.char, self.color, curses.A_BOLD)


class SpeechBubble:
    def __init__(
        self,
        text: str,
        anchor_x: int,
        anchor_y: int,
        width: int,
        height: int,
        color: int = 7,
        timer: int | None = None,
    ):
        max_line = max(12, min(44, width - 12))
        wrapped = textwrap.wrap(text.strip(), max_line) or [text.strip()[:max_line]]
        self.lines = wrapped[:3]
        self.inner = max(len(line) for line in self.lines)
        self.box_w = self.inner + 4
        self.box_h = len(self.lines) + 2
        self.color = color
        self.timer = timer if timer is not None else max(65, sum(len(line) for line in self.lines) * 3)

        bx = anchor_x - self.box_w // 2
        by = anchor_y - self.box_h - 1
        bx = int(clamp(bx, 1, max(1, width - self.box_w - 2)))
        if by < 2:
            by = anchor_y + 2
        if by + self.box_h >= height - 2:
            by = max(2, height - self.box_h - 3)
        self.x = bx
        self.y = by

    def update(self) -> bool:
        self.timer -= 1
        return self.timer > 0

    def draw(self, win) -> None:
        top = "." + "-" * (self.inner + 2) + "."
        bottom = "'" + "-" * (self.inner + 2) + "'"
        draw_text(win, self.y, self.x, top, self.color, curses.A_BOLD)
        for idx, line in enumerate(self.lines):
            draw_text(
                win,
                self.y + 1 + idx,
                self.x,
                "| " + line.ljust(self.inner) + " |",
                self.color,
                curses.A_BOLD,
            )
        draw_text(win, self.y + self.box_h - 1, self.x, bottom, self.color, curses.A_BOLD)


class Fish:
    def __init__(self, width: int, height: int, saved: dict | None = None):
        self.width = width
        self.height = height

        body_idx = saved.get("body_idx") if saved else None
        if not isinstance(body_idx, int) or not 0 <= body_idx < len(FISH_FORMS):
            body_idx = random.randrange(len(FISH_FORMS))
        self.body_idx = body_idx
        self.right_body, self.left_body = FISH_FORMS[body_idx]

        role_key = saved.get("role") if saved else None
        self.role = ROLE_BY_KEY.get(role_key, random.choice(ROLES))
        self.name = saved.get("name") if saved else self.generate_name()
        self.color = int(saved.get("color", self.role.color)) if saved else self.role.color
        self.thought_count = int(saved.get("thought_count", 0)) if saved else 0
        self.age = int(saved.get("age", 0)) if saved else random.randint(0, 3000)
        self.mood = saved.get("mood") if saved and saved.get("mood") in MOODS else random.choice(MOODS)

        self.fx = random.uniform(3, max(4, width - len(self.right_body) - 4))
        self.fy = random.uniform(4, max(5, height - 7))
        self.vx = random.choice([-1, 1]) * random.uniform(0.08, 0.25)
        self.vy = random.uniform(-0.06, 0.06)
        self.hunger = random.uniform(15, 72)
        self.energy = random.uniform(25, 100)
        self.phrase_timer = random.randint(140, 700)
        self.mood_timer = random.randint(260, 900)
        self.bubble_timer = random.randint(18, 80)
        self.trail_timer = 0
        self.trail: deque[tuple[int, int]] = deque(maxlen=5)
        self.speech: SpeechBubble | None = None
        self.sparkle = 0
        self.fade = random.randint(0, 300)
        self.last_wall_comment = 0

    @staticmethod
    def generate_name() -> str:
        if random.random() < 0.16:
            return "Dr. " + random.choice(FIRST_NAMES)
        return random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)

    @property
    def body(self) -> str:
        if self.mood == "sleepy" and random.random() < 0.18:
            return self.right_body.replace("o", "-") if self.vx >= 0 else self.left_body.replace("o", "-")
        return self.right_body if self.vx >= 0 else self.left_body

    @property
    def size(self) -> int:
        return len(self.body)

    @property
    def x(self) -> int:
        return int(self.fx)

    @property
    def y(self) -> int:
        return int(self.fy)

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "role": self.role.key,
            "body_idx": self.body_idx,
            "color": self.color,
            "thought_count": self.thought_count,
            "age": self.age,
            "mood": self.mood,
        }

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.fx = clamp(self.fx, 2, max(3, width - self.size - 3))
        self.fy = clamp(self.fy, 3, max(4, height - 6))
        self.speech = None

    def say(self, text: str, width: int, height: int, timer: int | None = None) -> None:
        self.speech = SpeechBubble(text, self.x + self.size // 2, self.y, width, height, self.color, timer)
        self.thought_count += 1

    def promote(self, width: int, height: int) -> None:
        choices = [role for role in ROLES if role.key != self.role.key]
        self.role = random.choice(choices)
        self.color = self.role.color
        self.mood = "ambitious"
        self.say(f"Update the placard. I am now {self.role.label}.", width, height, 95)

    def gossip(self, fish_list: list["Fish"], width: int, height: int) -> None:
        others = [fish for fish in fish_list if fish is not self]
        if not others:
            self.say("The rumor mill is just me in a circle.", width, height, 85)
            return
        other = random.choice(others)
        rumors = (
            f"I heard {other.name} is rehearsing a dramatic pause.",
            f"{other.name} keeps staring at the premium gravel.",
            f"Between us, {other.name} has main character lighting.",
            f"{other.name} and the filter are in negotiations.",
        )
        self.say(random.choice(rumors), width, height, 95)

    def _nearby_food(self, foods: list[FoodFlake]) -> FoodFlake | None:
        candidates = [food for food in foods if not food.eaten]
        if not candidates:
            return None
        return min(candidates, key=lambda food: (food.fx - self.fx) ** 2 + (food.fy - self.fy) ** 2)

    def update(
        self,
        fish_list: list["Fish"],
        foods: list[FoodFlake],
        particles: list[Particle],
        width: int,
        height: int,
        current: tuple[float, float],
        event_key: str | None,
    ) -> None:
        self.width = width
        self.height = height
        self.age += 1
        self.fade += 1
        self.hunger = clamp(self.hunger + 0.016 + len(fish_list) * 0.0007, 0, 100)
        self.energy = clamp(self.energy - 0.006, 0, 100)

        if self.speech and not self.speech.update():
            self.speech = None

        self.mood_timer -= 1
        if self.mood_timer <= 0:
            self.mood_timer = random.randint(300, 1100)
            if self.hunger > 76:
                self.mood = "snacky"
            elif self.energy < 20:
                self.mood = "sleepy"
            else:
                self.mood = random.choice(MOODS)

        # Food seeking.
        target = self._nearby_food(foods)
        if target and self.hunger > 25:
            dx = target.fx - self.fx
            dy = target.fy - self.fy
            dist = max(1.0, math.hypot(dx, dy))
            pull = 0.018 if self.hunger < 72 else 0.035
            self.vx += dx / dist * pull
            self.vy += dy / dist * pull
            if dist < max(2.2, self.size * 0.55):
                target.eaten = True
                self.hunger = clamp(self.hunger - random.uniform(24, 42), 0, 100)
                self.energy = clamp(self.energy + random.uniform(8, 18), 0, 100)
                for _ in range(random.randint(3, 7)):
                    particles.append(
                        Particle(
                            self.fx + self.size / 2,
                            self.fy,
                            random.uniform(-0.18, 0.18),
                            random.uniform(-0.35, -0.08),
                            random.choice((".", "*", "+")),
                            random.choice((3, 6, 7)),
                            random.randint(18, 40),
                            True,
                        )
                    )
                if self.speech is None and random.random() < 0.8:
                    self.say(random.choice(self.role.snack_phrases), width, height, 75)

        # Social spacing and tiny flocking.
        for other in fish_list:
            if other is self:
                continue
            dx = self.fx - other.fx
            dy = self.fy - other.fy
            dist2 = dx * dx + dy * dy
            if 0.1 < dist2 < 42:
                dist = math.sqrt(dist2)
                self.vx += dx / dist * 0.012
                self.vy += dy / dist * 0.006
            elif dist2 > 900 and random.random() < 0.004:
                self.vx += (other.fx - self.fx) * 0.0004
                self.vy += (other.fy - self.fy) * 0.0002

        # Role quirks.
        if self.role.key == "poet":
            self.vy += math.sin(self.age / 28.0) * 0.004
        elif self.role.key == "intern" and fish_list:
            lead = max(fish_list, key=lambda fish: fish.thought_count)
            if lead is not self:
                self.vx += (lead.fx - self.fx) * 0.0007
                self.vy += (lead.fy - self.fy) * 0.0004
        elif self.role.key == "dj":
            self.sparkle = max(self.sparkle, 2) if self.age % 18 == 0 else max(0, self.sparkle - 1)
        elif self.role.key == "cryptid":
            self.vx += math.sin(self.fade / 37.0) * 0.004
        elif self.role.key == "barista" and self.bubble_timer < 10:
            self.bubble_timer = min(self.bubble_timer, 5)

        if event_key in ("disco", "poetry"):
            self.vy += math.sin((self.age + self.x) / 6.0) * 0.015
        if event_key == "moon":
            self.vx += math.sin(self.age / 12.0) * 0.018
        if event_key == "mirror":
            self.vx *= 0.985
            self.vy *= 0.985

        self.vx += random.gauss(0, 0.006)
        self.vy += random.gauss(0, 0.004)
        self.vx += current[0]
        self.vy += current[1]
        self.vx = clamp(self.vx, -0.78, 0.78)
        self.vy = clamp(self.vy, -0.38, 0.38)
        self.fx += self.vx
        self.fy += self.vy

        min_y = 3
        max_y = max(4, height - 6)
        if self.fx <= 1:
            self.fx = 1
            self.vx = abs(self.vx) * random.uniform(0.75, 1.1)
            self._wall_comment(width, height)
        elif self.fx + self.size >= width - 1:
            self.fx = max(1, width - self.size - 2)
            self.vx = -abs(self.vx) * random.uniform(0.75, 1.1)
            self._wall_comment(width, height)

        if self.fy < min_y:
            self.fy = float(min_y)
            self.vy = abs(self.vy)
        elif self.fy > max_y:
            self.fy = float(max_y)
            self.vy = -abs(self.vy)

        self.trail_timer += 1
        if self.trail_timer >= 3:
            self.trail_timer = 0
            self.trail.appendleft((self.x + self.size // 2, self.y))

        self.bubble_timer -= 1
        if self.bubble_timer <= 0:
            self.bubble_timer = random.randint(20, 95)
            bx = self.x + self.size + 1 if self.vx >= 0 else self.x - 1
            particles.append(
                Particle(
                    float(bx),
                    float(self.y),
                    random.uniform(-0.04, 0.04),
                    -random.uniform(0.18, 0.42),
                    random.choice(("o", "O", ".")),
                    6,
                    random.randint(30, 80),
                    False,
                )
            )

        self.phrase_timer -= 1
        if self.phrase_timer <= 0 and self.speech is None:
            self.phrase_timer = random.randint(300, 1050)
            self._say_contextual(width, height, event_key)

    def _wall_comment(self, width: int, height: int) -> None:
        if self.age - self.last_wall_comment < 240 or self.speech is not None:
            return
        self.last_wall_comment = self.age
        if random.random() < 0.28:
            self.say(random.choice(("Wall again.", "I meant to do that.", "The glass wins this round.")), width, height, 70)

    def _say_contextual(self, width: int, height: int, event_key: str | None) -> None:
        if event_key in EVENT_CHATTER and random.random() < 0.65:
            line = random.choice(EVENT_CHATTER[event_key])
        elif self.hunger > 82:
            line = random.choice(MOOD_PHRASES["snacky"])
        else:
            pool = list(GENERAL_PHRASES)
            pool.extend(self.role.phrases)
            pool.extend(MOOD_PHRASES.get(self.mood, ()))
            line = random.choice(pool)
        if random.random() < 0.12:
            line = f"{self.name}: {line}"
        self.say(line, width, height)

    def draw(self, win, selected: bool = False) -> None:
        dim = self.role.key == "cryptid" and (self.fade // 13) % 5 == 0
        attr = curses.A_DIM if dim else curses.A_BOLD
        if self.mood == "sleepy":
            attr = curses.A_DIM
        for idx, (tx, ty) in enumerate(list(self.trail)[:3]):
            draw_ch(win, ty, tx, "~" if idx == 0 else ".", 6, curses.A_DIM)

        body = self.body
        if selected:
            draw_ch(win, self.y - 1, self.x + max(0, self.size // 2), "v", 3, curses.A_BOLD)
            attr |= curses.A_REVERSE
        draw_text(win, self.y, self.x, body, self.color, attr)

        if self.sparkle > 0:
            draw_ch(win, self.y, self.x - 1, "*", 3, curses.A_BOLD)
            draw_ch(win, self.y, self.x + self.size, "*", 5, curses.A_BOLD)

        if selected and self.y > 2:
            tag = f"{self.name[:18]} / {self.role.label}"
            tx = int(clamp(self.x + self.size // 2 - len(tag) // 2, 1, max(1, self.width - len(tag) - 2)))
            draw_text(win, self.y - 2, tx, tag, 7, curses.A_DIM)

        if self.speech:
            self.speech.draw(win)


class Kelp:
    def __init__(self, x: int, height: int):
        self.x = x
        self.height = height
        self.stalks = random.randint(4, max(5, min(9, height // 3)))
        self.phase = random.random() * math.tau
        self.color = random.choice((2, 2, 2, 3))

    def resize(self, height: int) -> None:
        self.height = height
        self.stalks = min(self.stalks, max(4, height // 3))

    def draw(self, win, frame: int) -> None:
        base = self.height - 4
        for i in range(self.stalks):
            y = base - i
            sway = int(round(math.sin(frame / 9 + self.phase + i * 0.7)))
            ch = "(" if (frame // 8 + i) % 2 == 0 else ")"
            draw_ch(win, y, self.x + sway, ch, self.color, curses.A_BOLD)


class SnailCourier:
    MESSAGES = (
        "URGENT: the left wall has news.",
        "Invoice enclosed: three bubbles.",
        "You are invited to a meeting nobody asked for.",
        "Please stop licking the fake castle.",
        "Reminder: casual Friday is still wet.",
        "The cave requests privacy.",
        "Congrats, you have been nominated for Most Suspicious.",
    )

    def __init__(self, width: int, height: int):
        self.wait = random.randint(80, 260)
        self.active = False
        self.x = -5.0
        self.y = height - 5
        self.dx = 0.05
        self.message = random.choice(self.MESSAGES)
        self.delivered = False
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.y = max(5, height - 5)

    def update(self, fish_list: list[Fish], width: int, height: int) -> None:
        self.resize(width, height)
        if not self.active:
            self.wait -= 1
            if self.wait <= 0:
                self.active = True
                self.delivered = False
                self.message = random.choice(self.MESSAGES)
                if random.random() < 0.5:
                    self.x = -4.0
                    self.dx = random.uniform(0.035, 0.075)
                else:
                    self.x = float(width + 3)
                    self.dx = -random.uniform(0.035, 0.075)
            return

        self.x += self.dx
        if not self.delivered and fish_list:
            nearby = min(fish_list, key=lambda fish: abs(fish.y - self.y) + abs(fish.x - self.x))
            if abs(nearby.x - self.x) < 5 and abs(nearby.y - self.y) < 4:
                nearby.say(self.message, width, height, 90)
                self.delivered = True
        if self.x < -8 or self.x > width + 8:
            self.active = False
            self.wait = random.randint(240, 700)

    def draw(self, win) -> None:
        if not self.active:
            return
        body = "@~>" if self.dx > 0 else "<~@"
        draw_text(win, self.y, int(self.x), body, 3, curses.A_BOLD)
        draw_ch(win, self.y - 1, int(self.x) + (1 if self.dx > 0 else 2), "'", 7, curses.A_DIM)


class EventManager:
    EVENTS = ("board", "audit", "poetry", "disco", "moon", "mirror", "sub", "oracle")

    LABELS = {
        "board": "EMERGENCY BOARD MEETING",
        "audit": "TINY TAX AUDIT",
        "poetry": "POETRY SLAM",
        "disco": "DISCO LEAK",
        "moon": "MOON TIDE",
        "mirror": "MIRROR DAY",
        "sub": "SUBMARINE INSPECTION",
        "oracle": "GLASS ORACLE",
    }

    def __init__(self):
        self.key: str | None = None
        self.frame = 0
        self.duration = 0
        self.cooldown = random.randint(420, 900)
        self.banner = ""
        self.banner_timer = 0
        self.data: dict = {}

    @property
    def active(self) -> bool:
        return self.key is not None

    def force(self, fish_list: list[Fish], width: int, height: int, particles: list[Particle]) -> None:
        self.start(random.choice(self.EVENTS), fish_list, width, height, particles)

    def start(
        self,
        key: str,
        fish_list: list[Fish],
        width: int,
        height: int,
        particles: list[Particle],
    ) -> None:
        self.key = key
        self.frame = 0
        self.duration = {
            "board": 250,
            "audit": 210,
            "poetry": 260,
            "disco": 240,
            "moon": 220,
            "mirror": 180,
            "sub": 230,
            "oracle": 190,
        }.get(key, 180)
        self.banner = self.LABELS.get(key, "ODD EVENT")
        self.banner_timer = 110
        self.data = {}

        if key == "sub":
            self.data["x"] = -18.0
            self.data["y"] = random.randint(max(4, height // 4), max(5, height // 2))
            self.data["speed"] = (width + 28) / self.duration
        elif key == "audit":
            self.data["x"] = float(width + 4)
            self.data["y"] = random.randint(4, max(5, height - 8))
            self.data["speed"] = -(width + 10) / self.duration
            self.data["stamp"] = 0
        elif key == "oracle":
            self.data["x"] = width // 2
            self.data["y"] = max(4, height // 3)
            self.data["pulse"] = 0
        elif key == "moon":
            self.data["moon_x"] = random.randint(width // 3, max(width // 3, width - 10))
        elif key in ("board", "poetry"):
            self.data["cx"] = width // 2
            self.data["cy"] = height // 2

        if fish_list:
            speakers = random.sample(fish_list, min(len(fish_list), 4))
            for fish in speakers:
                if key in EVENT_CHATTER:
                    fish.say(random.choice(EVENT_CHATTER[key]), width, height, random.randint(70, 110))
        if key == "disco":
            for _ in range(35):
                particles.append(
                    Particle(
                        random.uniform(2, width - 3),
                        random.uniform(2, height // 2),
                        random.uniform(-0.07, 0.07),
                        random.uniform(0.12, 0.38),
                        random.choice(("*", "+", ".", "o")),
                        random.randint(1, 6),
                        random.randint(50, 110),
                        True,
                    )
                )

    def update(
        self,
        fish_list: list[Fish],
        foods: list[FoodFlake],
        particles: list[Particle],
        width: int,
        height: int,
    ) -> None:
        if self.banner_timer > 0:
            self.banner_timer -= 1

        if not self.active:
            self.cooldown -= 1
            if self.cooldown <= 0:
                self.start(random.choice(self.EVENTS), fish_list, width, height, particles)
            return

        self.frame += 1
        key = self.key

        if key == "board":
            cx = self.data.get("cx", width // 2)
            cy = self.data.get("cy", height // 2)
            for fish in fish_list:
                fish.vx += (cx - fish.fx) * 0.0015
                fish.vy += (cy - fish.fy) * 0.001
            if self.frame % 52 == 0 and fish_list:
                random.choice(fish_list).say(random.choice(EVENT_CHATTER["board"]), width, height, 80)

        elif key == "poetry":
            stage_y = max(5, height - 8)
            if fish_list:
                poet = fish_list[(self.frame // 90) % len(fish_list)]
                poet.vx += (width // 2 - poet.fx) * 0.006
                poet.vy += (stage_y - poet.fy) * 0.004
                if self.frame % 70 == 4:
                    poet.say(random.choice(EVENT_CHATTER["poetry"]), width, height, 85)

        elif key == "audit":
            self.data["x"] += self.data["speed"]
            self.data["stamp"] += 1
            ix = self.data["x"]
            for fish in fish_list:
                if abs(fish.x - ix) < 10:
                    fish.vx += -0.05 if fish.x > ix else 0.05
                    fish.mood = "suspicious"
            if self.frame % 46 == 0 and fish_list:
                random.choice(fish_list).say(random.choice(EVENT_CHATTER["audit"]), width, height, 75)

        elif key == "disco":
            if self.frame % 7 == 0:
                for fish in fish_list:
                    fish.color = random.randint(1, 6)
                    fish.sparkle = 4
            if self.frame % 3 == 0:
                particles.append(
                    Particle(
                        random.uniform(2, width - 3),
                        2.0,
                        random.uniform(-0.09, 0.09),
                        random.uniform(0.12, 0.35),
                        random.choice(("*", "+", ".", "o", "#")),
                        random.randint(1, 6),
                        random.randint(35, 95),
                        True,
                    )
                )

        elif key == "moon":
            if self.frame % 8 == 0:
                particles.append(
                    Particle(
                        random.uniform(2, width - 3),
                        random.uniform(3, height - 8),
                        0,
                        random.uniform(-0.12, -0.04),
                        "~",
                        6,
                        random.randint(25, 60),
                        False,
                    )
                )
            if self.frame % 62 == 0 and fish_list:
                random.choice(fish_list).say(random.choice(EVENT_CHATTER["moon"]), width, height, 80)

        elif key == "mirror":
            if len(fish_list) >= 2:
                leader = fish_list[0]
                for fish in fish_list[1:]:
                    fish.vx += (leader.vx - fish.vx) * 0.04
                    fish.vy += (leader.vy - fish.vy) * 0.04
            if self.frame % 58 == 0 and fish_list:
                random.choice(fish_list).say(random.choice(EVENT_CHATTER["mirror"]), width, height, 80)

        elif key == "sub":
            self.data["x"] += self.data["speed"]
            sx = self.data["x"]
            for fish in fish_list:
                if abs(fish.x - sx) < 15:
                    fish.vx += -0.04 if fish.x > sx else 0.04
            if self.frame % 50 == 0 and fish_list:
                random.choice(fish_list).say(random.choice(EVENT_CHATTER["sub"]), width, height, 75)

        elif key == "oracle":
            self.data["pulse"] += 1
            if self.frame % 48 == 0 and fish_list:
                random.choice(fish_list).say(random.choice(EVENT_CHATTER["oracle"]), width, height, 85)
            if self.frame == 70:
                for _ in range(14):
                    foods.append(
                        FoodFlake(
                            random.uniform(width * 0.25, width * 0.75),
                            2.0,
                            random.choice((".", "*", "o")),
                            3,
                            random.uniform(0.25, 0.46),
                            random.uniform(-0.07, 0.07),
                        )
                    )

        if self.frame >= self.duration:
            self.end(fish_list)

    def end(self, fish_list: list[Fish]) -> None:
        if self.key == "disco":
            for fish in fish_list:
                fish.color = fish.role.color
        self.key = None
        self.frame = 0
        self.duration = 0
        self.data = {}
        self.cooldown = random.randint(460, 980)

    def draw(self, win, width: int, height: int) -> None:
        if self.banner_timer > 0 and self.banner:
            msg = f" {self.banner} "
            draw_text(
                win,
                2,
                max(1, (width - len(msg)) // 2),
                msg,
                3 if self.key not in ("audit", "disco") else (1 if self.key == "audit" else 5),
                curses.A_BOLD | curses.A_REVERSE,
            )

        if not self.active:
            return

        key = self.key
        if key == "board":
            cx = width // 2
            cy = height // 2
            draw_text(win, cy, max(2, cx - 11), "+--------------------+", 7, curses.A_DIM)
            draw_text(win, cy + 1, max(2, cx - 11), "| AGENDA: MORE FOOD |", 7, curses.A_BOLD)
            draw_text(win, cy + 2, max(2, cx - 11), "+--------------------+", 7, curses.A_DIM)
        elif key == "poetry":
            stage = "==== OPEN MIC ===="
            draw_text(win, height - 6, max(2, (width - len(stage)) // 2), stage, 5, curses.A_BOLD)
            draw_text(win, height - 5, max(2, (width - 12) // 2), "   \\||||/   ", 1, curses.A_DIM)
        elif key == "audit":
            x = int(self.data.get("x", width // 2))
            y = int(self.data.get("y", height // 2))
            draw_text(win, y, x, "[AUDIT]", 1, curses.A_BOLD | curses.A_REVERSE)
            if self.data.get("stamp", 0) % 24 < 8:
                draw_text(win, y + 1, x - 2, "STAMP", 1, curses.A_BOLD)
        elif key == "moon":
            mx = int(self.data.get("moon_x", width - 10))
            draw_text(win, 3, mx, "( )", 7, curses.A_BOLD)
            wave_y = max(4, height // 4)
            for x in range(2, width - 2, 4):
                y = wave_y + int(math.sin((x + self.frame) / 5.0) * 1.2)
                draw_text(win, y, x, "~~", 6, curses.A_DIM)
        elif key == "sub":
            sx = int(self.data.get("x", -10))
            sy = int(self.data.get("y", height // 2))
            sub = "<[o_o]===="
            draw_text(win, sy, sx, sub, 4, curses.A_BOLD)
            draw_text(win, sy + 1, sx + 2, "......", 6, curses.A_DIM)
        elif key == "oracle":
            ox = int(self.data.get("x", width // 2))
            oy = int(self.data.get("y", height // 3))
            pulse = self.data.get("pulse", 0)
            ring = "(( O ))" if pulse % 20 < 10 else "(  O  )"
            draw_text(win, oy, max(2, ox - len(ring) // 2), ring, 5, curses.A_BOLD)
            draw_text(win, oy + 1, max(2, ox - 10), "THE GLASS KNOWS", 7, curses.A_DIM)


class Aquarium:
    CURRENT_MODES = ("still", "lazy left", "lazy right", "updraft", "whirlpool")

    def __init__(self, stdscr, seed: int | None = None, no_save: bool = False):
        if seed is not None:
            random.seed(seed)
        self.stdscr = stdscr
        self.no_save = no_save
        self.height, self.width = stdscr.getmaxyx()
        self.frame = 0
        self.paused = False
        self.panel: str | None = None
        self.panel_timer = 0
        self.selected = 0
        self.current_idx = 0
        self.cursor_x = self.width // 2
        self.cursor_y = self.height // 2
        self.message = ""
        self.message_timer = 0

        saved = None if no_save else self.load_state()
        self.fish: list[Fish] = self.make_fish(saved)
        self.kelp: list[Kelp] = self.make_kelp()
        self.foods: list[FoodFlake] = []
        self.particles: list[Particle] = []
        self.snail = SnailCourier(self.width, self.height)
        self.events = EventManager()

    def make_fish(self, saved: dict | None) -> list[Fish]:
        target = min(13, max(5, self.width // 13))
        saved_fish = saved.get("fish", []) if isinstance(saved, dict) else []
        fish_list: list[Fish] = []
        for idx in range(target):
            data = saved_fish[idx] if idx < len(saved_fish) and isinstance(saved_fish[idx], dict) else None
            fish_list.append(Fish(self.width, self.height, data))
        return fish_list

    def make_kelp(self) -> list[Kelp]:
        count = min(18, max(5, self.width // 11))
        slots = list(range(4, max(5, self.width - 4), 5))
        random.shuffle(slots)
        return [Kelp(x, self.height) for x in sorted(slots[:count])]

    def load_state(self) -> dict | None:
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def save_state(self) -> None:
        if self.no_save:
            return
        payload = {
            "version": 1,
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "fish": [fish.serialize() for fish in self.fish[:24]],
        }
        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass

    def setup_curses(self) -> None:
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.timeout(0)
        curses.mouseinterval(0)
        try:
            curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        except Exception:
            pass

        curses.start_color()
        try:
            curses.use_default_colors()
        except Exception:
            pass
        pairs = (
            curses.COLOR_RED,
            curses.COLOR_GREEN,
            curses.COLOR_YELLOW,
            curses.COLOR_BLUE,
            curses.COLOR_MAGENTA,
            curses.COLOR_CYAN,
            curses.COLOR_WHITE,
        )
        for idx, fg in enumerate(pairs, start=1):
            try:
                curses.init_pair(idx, fg, -1)
            except Exception:
                pass

    def run(self) -> None:
        self.setup_curses()
        last_tick = time.monotonic()
        while True:
            should_quit = self.handle_input()
            if should_quit:
                self.save_state()
                break

            now = time.monotonic()
            if now - last_tick >= FRAME_DELAY:
                last_tick = now
                self.check_resize()
                if not self.paused:
                    self.update()
                self.draw()
            time.sleep(0.005)

    def handle_input(self) -> bool:
        while True:
            key = self.stdscr.getch()
            if key == -1:
                return False
            if key in (ord("q"), ord("Q")):
                return True
            if key == 27:
                self.panel = None
                continue
            if key in (ord("h"), ord("H")):
                self.panel = None if self.panel == "help" else "help"
            elif key in (ord("i"), ord("I")):
                self.panel = None if self.panel == "ideas" else "ideas"
            elif key in (ord("l"), ord("L")):
                self.panel = None if self.panel == "ledger" else "ledger"
            elif key in (ord("p"), ord("P")):
                self.paused = not self.paused
                self.flash("Paused" if self.paused else "Unpaused")
            elif key in (ord("+"), ord("=")):
                if len(self.fish) < 28:
                    self.fish.append(Fish(self.width, self.height))
                    self.selected = len(self.fish) - 1
                    self.flash("A new fish has joined with suspicious confidence.")
            elif key in (ord("-"), ord("_")):
                if len(self.fish) > 1:
                    removed = self.fish.pop(self.selected % len(self.fish))
                    self.selected = max(0, min(self.selected, len(self.fish) - 1))
                    self.flash(f"{removed.name} left to pursue solo projects.")
            elif key in (ord("f"), ord("F")):
                self.drop_food(self.cursor_x, max(3, self.cursor_y), amount=random.randint(8, 18))
                self.flash("Snack weather.")
            elif key in (ord("e"), ord("E")):
                self.events.force(self.fish, self.width, self.height, self.particles)
            elif key in (ord("c"), ord("C")):
                self.current_idx = (self.current_idx + 1) % len(self.CURRENT_MODES)
                self.flash(f"Current: {self.CURRENT_MODES[self.current_idx]}")
            elif key in (ord("b"), ord("B")):
                self.boop(self.cursor_x, self.cursor_y)
            elif key in (ord("r"), ord("R")):
                if self.fish:
                    self.fish[self.selected % len(self.fish)].gossip(self.fish, self.width, self.height)
            elif key in (ord("m"), ord("M")):
                if self.fish:
                    self.fish[self.selected % len(self.fish)].promote(self.width, self.height)
            elif key in (ord("n"), ord("N")):
                self.name_selected()
            elif key in (9, curses.KEY_RIGHT):
                self.selected = (self.selected + 1) % len(self.fish)
            elif key == curses.KEY_LEFT:
                self.selected = (self.selected - 1) % len(self.fish)
            elif key == curses.KEY_UP:
                self.cursor_y = max(2, self.cursor_y - 1)
            elif key == curses.KEY_DOWN:
                self.cursor_y = min(self.height - 4, self.cursor_y + 1)
            elif key == curses.KEY_MOUSE:
                self.handle_mouse()

    def handle_mouse(self) -> None:
        try:
            _, mx, my, _, state = curses.getmouse()
        except curses.error:
            return
        self.cursor_x = int(clamp(mx, 1, self.width - 2))
        self.cursor_y = int(clamp(my, 2, self.height - 4))
        if state & curses.BUTTON1_PRESSED:
            self.drop_food(mx, my, amount=random.randint(4, 9))
        elif state & curses.BUTTON3_PRESSED:
            self.boop(mx, my)
        elif state & curses.BUTTON1_DOUBLE_CLICKED:
            self.drop_food(mx, my, amount=random.randint(14, 24))

    def name_selected(self) -> None:
        if not self.fish:
            return
        fish = self.fish[self.selected % len(self.fish)]
        curses.curs_set(1)
        self.stdscr.nodelay(False)
        self.stdscr.timeout(-1)
        value = []
        prompt = f" Rename {fish.name[:18]}: "
        while True:
            self.draw()
            text = prompt + "".join(value) + "_"
            draw_text(self.stdscr, self.height - 1, 1, text[: self.width - 2], 3, curses.A_BOLD | curses.A_REVERSE)
            self.stdscr.refresh()
            ch = self.stdscr.getch()
            if ch in (10, 13):
                break
            if ch == 27:
                value = []
                break
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                if value:
                    value.pop()
            elif 32 <= ch <= 126 and len(value) < 24:
                value.append(chr(ch))
        name = "".join(value).strip()
        if name:
            fish.name = name
            fish.say("New name, same confusing life.", self.width, self.height, 85)
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.timeout(0)

    def check_resize(self) -> None:
        height, width = self.stdscr.getmaxyx()
        if height == self.height and width == self.width:
            return
        self.height, self.width = height, width
        self.cursor_x = min(self.cursor_x, width - 2)
        self.cursor_y = min(self.cursor_y, height - 4)
        for fish in self.fish:
            fish.resize(width, height)
        for kelp in self.kelp:
            kelp.resize(height)
        self.snail.resize(width, height)
        if len(self.kelp) < min(18, max(5, width // 11)):
            self.kelp = self.make_kelp()

    def current_vector_for(self, fish: Fish) -> tuple[float, float]:
        mode = self.CURRENT_MODES[self.current_idx]
        if mode == "lazy left":
            return (-0.006, 0.0)
        if mode == "lazy right":
            return (0.006, 0.0)
        if mode == "updraft":
            return (0.0, -0.005)
        if mode == "whirlpool":
            cx = self.width / 2
            cy = self.height / 2
            dx = fish.fx - cx
            dy = fish.fy - cy
            dist = max(4.0, math.hypot(dx, dy))
            return (-dy / dist * 0.012, dx / dist * 0.006)
        return (0.0, 0.0)

    def drop_food(self, x: int, y: int, amount: int = 10) -> None:
        top = 3 if y < 3 else y
        for _ in range(amount):
            self.foods.append(
                FoodFlake(
                    float(clamp(x + random.randint(-6, 6), 2, self.width - 3)),
                    float(clamp(top + random.randint(-2, 2), 2, self.height - 5)),
                    random.choice((".", ".", "*", "o", ",")),
                    random.choice((3, 3, 7)),
                    random.uniform(0.12, 0.42),
                    random.uniform(-0.08, 0.08),
                )
            )

    def boop(self, x: int, y: int) -> None:
        phrases = ("The glass has opinions!", "Tap noted.", "Rude but impressive.", "I felt that in my fins.")
        for fish in self.fish:
            dx = fish.fx - x
            dy = fish.fy - y
            dist = max(2.0, math.hypot(dx, dy))
            push = 0.75 / dist
            fish.vx += dx * push
            fish.vy += dy * push * 0.6
            if dist < 18 and fish.speech is None and random.random() < 0.55:
                fish.say(random.choice(phrases), self.width, self.height, 75)
        for _ in range(20):
            self.particles.append(
                Particle(
                    float(x),
                    float(y),
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.28, 0.28),
                    random.choice(("~", ".", "o")),
                    6,
                    random.randint(18, 45),
                    False,
                )
            )
        self.flash("Boop registered with the glass department.")

    def flash(self, message: str, timer: int = 90) -> None:
        self.message = message
        self.message_timer = timer

    def update(self) -> None:
        self.frame += 1
        self.events.update(self.fish, self.foods, self.particles, self.width, self.height)
        self.foods = [food for food in self.foods if food.update(self.width, self.height) and not food.eaten]
        self.particles = [p for p in self.particles if p.update(self.width, self.height)]

        self.snail.update(self.fish, self.width, self.height)
        for fish in self.fish:
            fish.update(
                self.fish,
                self.foods,
                self.particles,
                self.width,
                self.height,
                self.current_vector_for(fish),
                self.events.key,
            )
        self.foods = [food for food in self.foods if not food.eaten]

        if self.message_timer > 0:
            self.message_timer -= 1

    def draw(self) -> None:
        self.stdscr.erase()
        if self.height < MIN_HEIGHT or self.width < MIN_WIDTH:
            draw_text(self.stdscr, 1, 2, f"Please resize terminal to at least {MIN_WIDTH}x{MIN_HEIGHT}.", 1, curses.A_BOLD)
            self.stdscr.refresh()
            return

        self.draw_water()
        self.draw_floor()
        for kelp in self.kelp:
            kelp.draw(self.stdscr, self.frame)
        self.draw_decor()

        for food in self.foods:
            food.draw(self.stdscr)
        for particle in self.particles:
            particle.draw(self.stdscr)

        self.snail.draw(self.stdscr)
        for idx, fish in enumerate(self.fish):
            fish.draw(self.stdscr, selected=idx == self.selected % len(self.fish))

        self.events.draw(self.stdscr, self.width, self.height)
        self.draw_cursor()
        self.draw_border_and_status()
        if self.panel == "help":
            self.draw_help()
        elif self.panel == "ideas":
            self.draw_ideas()
        elif self.panel == "ledger":
            self.draw_ledger()
        if self.message_timer > 0:
            draw_text(
                self.stdscr,
                2,
                max(1, (self.width - len(self.message) - 2) // 2),
                f" {self.message} ",
                6,
                curses.A_BOLD | curses.A_REVERSE,
            )
        self.stdscr.refresh()

    def draw_border_and_status(self) -> None:
        width = self.width
        height = self.height
        title = " O D D Q U A R I U M "
        clock = _dt.datetime.now().strftime("%H:%M")
        draw_text(self.stdscr, 0, 0, "+" + "~" * (width - 2) + "+", 6, curses.A_BOLD)
        draw_text(self.stdscr, 0, max(2, (width - len(title)) // 2), title, 6, curses.A_BOLD)
        draw_text(self.stdscr, 0, max(2, width - len(clock) - 2), clock, 7, curses.A_DIM)
        for y in range(1, height - 1):
            draw_ch(self.stdscr, y, 0, "|", 7, curses.A_DIM)
            draw_ch(self.stdscr, y, width - 1, "|", 7, curses.A_DIM)
        draw_text(self.stdscr, height - 1, 0, "+" + "=" * (width - 2) + "+", 7, curses.A_BOLD)

        if self.fish:
            selected = self.fish[self.selected % len(self.fish)]
            selected_text = f"{selected.name[:16]} the {selected.role.label}"
        else:
            selected_text = "no fish"
        mode = self.CURRENT_MODES[self.current_idx]
        left = f" {len(self.fish)} fish | current: {mode} | {selected_text}"
        right = " H help I ideas L ledger F feed E event B boop C current Q quit "
        status = left + " |" + right
        draw_text(self.stdscr, height - 1, max(1, (width - len(status)) // 2), status[: width - 2], 6, curses.A_DIM)

    def draw_water(self) -> None:
        for x in range(1, self.width - 1):
            ch = "~" if (x + self.frame // 3) % 5 else "-"
            draw_ch(self.stdscr, 1, x, ch, 6, curses.A_DIM)
        for y in range(2, self.height - 4):
            if y % 4 == 0:
                for x in range(3 + (self.frame + y) % 9, self.width - 3, 16):
                    draw_ch(self.stdscr, y, x, ".", 6, curses.A_DIM)

    def draw_floor(self) -> None:
        y = self.height - 4
        for x in range(1, self.width - 1):
            ch = "." if x % 3 else ","
            draw_ch(self.stdscr, y, x, ch, 3, curses.A_DIM)
        for x in range(2, self.width - 2, 7):
            if (x + self.frame // 30) % 3 == 0:
                draw_ch(self.stdscr, y - 1, x, ".", 7, curses.A_DIM)

    def draw_decor(self) -> None:
        floor = self.height - 4
        # Little questionable castle.
        cx = max(4, self.width // 7)
        castle = (
            (floor - 4, "  []  "),
            (floor - 3, " [##] "),
            (floor - 2, "[####]"),
            (floor - 1, "|_||_|"),
        )
        for y, text in castle:
            draw_text(self.stdscr, y, cx, text, 4, curses.A_DIM)

        # Bubble bar.
        bx = max(8, self.width - 26)
        draw_text(self.stdscr, floor - 3, bx, "BUBBLE BAR", 3, curses.A_DIM)
        draw_text(self.stdscr, floor - 2, bx, "[o][o][o]", 3, curses.A_BOLD)
        draw_text(self.stdscr, floor - 1, bx, "|______|", 3, curses.A_DIM)

        # Therapy couch / poetry stage prop.
        tx = self.width // 2 - 5
        draw_text(self.stdscr, floor - 2, tx, "~~~~~~~", 5, curses.A_DIM)
        draw_text(self.stdscr, floor - 1, tx, "|_____|", 5, curses.A_DIM)

        for sx in (self.width // 4, self.width // 2 + 14, self.width - 8):
            if 2 < sx < self.width - 2:
                draw_ch(self.stdscr, floor, sx, "*", 3, curses.A_BOLD)

    def draw_cursor(self) -> None:
        if self.panel:
            return
        draw_ch(self.stdscr, self.cursor_y, self.cursor_x, "+", 7, curses.A_DIM)

    def draw_panel_box(self, title: str, lines: Iterable[str], color: int = 7) -> None:
        lines = list(lines)
        max_text = min(max(len(line) for line in lines + [title]) + 4, self.width - 8)
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, max_text - 4) or [""])
        panel_w = min(max(max(len(line) for line in wrapped + [title]) + 4, 30), self.width - 6)
        panel_h = min(len(wrapped) + 4, self.height - 4)
        x = max(2, (self.width - panel_w) // 2)
        y = max(2, (self.height - panel_h) // 2)
        draw_text(self.stdscr, y, x, "+" + "-" * (panel_w - 2) + "+", color, curses.A_BOLD)
        draw_text(self.stdscr, y + 1, x, "| " + title[: panel_w - 4].ljust(panel_w - 4) + " |", color, curses.A_BOLD)
        draw_text(self.stdscr, y + 2, x, "|" + "-" * (panel_w - 2) + "|", color, curses.A_DIM)
        for idx, line in enumerate(wrapped[: panel_h - 4]):
            draw_text(self.stdscr, y + 3 + idx, x, "| " + line[: panel_w - 4].ljust(panel_w - 4) + " |", color, curses.A_DIM)
        draw_text(self.stdscr, y + panel_h - 1, x, "+" + "-" * (panel_w - 2) + "+", color, curses.A_BOLD)

    def draw_help(self) -> None:
        lines = (
            "Q quit and save | P pause | Esc close panel",
            "F sprinkle food at the cursor | mouse left-click also feeds",
            "B boop the glass | mouse right-click also boops",
            "+ add fish | - remove selected fish",
            "Tab/right and left cycle spotlight fish | N rename selected",
            "E force a weird event | C cycle currents | M promote fish to a new job",
            "R make the selected fish spread a rumor | L open the fish ledger",
            "Arrow up/down move the food/boop cursor.",
        )
        self.draw_panel_box("HELP", lines, 6)

    def draw_ideas(self) -> None:
        self.draw_panel_box("IDEAS BUILT IN", [f"- {idea}" for idea in IDEAS], 5)

    def draw_ledger(self) -> None:
        rows = []
        for idx, fish in enumerate(sorted(self.fish, key=lambda f: f.thought_count, reverse=True)[:10], 1):
            rows.append(
                f"{idx:>2}. {fish.name[:18]:<18} {fish.role.label:<14} mood:{fish.mood:<10} thoughts:{fish.thought_count:<4} hunger:{int(fish.hunger):>3}"
            )
        if not rows:
            rows = ["No fish. An avant-garde aquarium."]
        self.draw_panel_box("FISH LEDGER", rows, 3)


def render_snapshot(width: int = 82, height: int = 24, seed: int | None = None) -> str:
    rng = random.Random(seed)
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    def put(y: int, x: int, text: str) -> None:
        if y < 0 or y >= height:
            return
        for idx, ch in enumerate(text):
            xx = x + idx
            if 0 <= xx < width:
                canvas[y][xx] = ch

    put(0, 0, "+" + "~" * (width - 2) + "+")
    title = " O D D Q U A R I U M "
    put(0, max(1, (width - len(title)) // 2), title)
    for y in range(1, height - 1):
        put(y, 0, "|")
        put(y, width - 1, "|")
    put(height - 1, 0, "+" + "=" * (width - 2) + "+")
    for x in range(1, width - 1):
        canvas[1][x] = "~" if x % 5 else "-"
        canvas[height - 4][x] = "." if x % 3 else ","

    for x in range(5, width - 5, 13):
        stalks = rng.randint(4, 8)
        for i in range(stalks):
            y = height - 5 - i
            if 1 < y < height - 2:
                put(y, x + rng.choice((-1, 0, 1)), rng.choice(("(", ")")))

    decor_y = height - 7
    put(decor_y, max(4, width // 7), " [##] ")
    put(decor_y + 1, max(4, width // 7), "[####]")
    put(decor_y + 2, max(4, width // 7), "|_||_|")
    put(decor_y + 1, max(10, width - 24), "[o][o][o]")
    put(decor_y + 2, max(10, width - 24), "|______|")

    roles = list(ROLES)
    for _ in range(min(9, width // 10)):
        body = rng.choice(FISH_FORMS)
        text = body[0] if rng.random() < 0.5 else body[1]
        y = rng.randint(3, max(3, height - 8))
        x = rng.randint(2, max(3, width - len(text) - 3))
        put(y, x, text)
        if rng.random() < 0.33:
            role = rng.choice(roles)
            msg = rng.choice(role.phrases)
            msg = textwrap.shorten(msg, width=26, placeholder="...")
            put(max(2, y - 2), min(width - len(msg) - 3, max(2, x - 4)), f"({msg})")

    put(height - 2, 2, "@~>  snail mail: casual Friday is still wet")
    status = " Snapshot mode | run python3 oddquarium.py for the interactive tank "
    put(height - 1, max(1, (width - len(status)) // 2), status[: width - 2])
    return "\n".join("".join(row).rstrip() for row in canvas)


def print_ideas() -> None:
    print("Quirky directions baked into this version:")
    for idx, idea in enumerate(IDEAS, 1):
        print(f"{idx}. {idea}")


def run_curses(seed: int | None, no_save: bool) -> None:
    def _wrapped(stdscr):
        Aquarium(stdscr, seed=seed, no_save=no_save).run()

    curses.wrapper(_wrapped)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quirky terminal fish tank.")
    parser.add_argument("--snapshot", action="store_true", help="print a non-interactive ASCII preview")
    parser.add_argument("--ideas", action="store_true", help="print the design ideas behind the tank")
    parser.add_argument("--seed", type=int, default=None, help="seed the random generator")
    parser.add_argument("--no-save", action="store_true", help="disable loading and saving ~/.oddquarium.json")
    parser.add_argument("--width", type=int, default=82, help="snapshot width")
    parser.add_argument("--height", type=int, default=24, help="snapshot height")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.ideas:
        print_ideas()
        return 0
    if args.snapshot:
        print(render_snapshot(max(40, args.width), max(16, args.height), args.seed))
        return 0
    if sys.platform == "win32":
        print("Oddquarium uses curses and is designed for macOS/Linux terminals.")
        return 1
    try:
        run_curses(args.seed, args.no_save)
    except KeyboardInterrupt:
        return 0
    except curses.error as exc:
        print(f"Could not start curses: {exc}")
        print("Try running it in a larger real terminal, or preview with --snapshot.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
