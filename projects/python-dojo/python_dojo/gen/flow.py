from __future__ import annotations

import random

from ..models import CodeTest
from .registry import archetype, make_exercise
from .support import FRUITS, NAMES, SIMPLE_WORDS, WORDS


# --------------------------------------------------------------------------
# booleans
# --------------------------------------------------------------------------


@archetype("booleans", "and_comparison")
def and_comparison(rng: random.Random, seed: int):
    age = rng.randint(12, 30)
    minimum = rng.randint(16, 21)
    member = rng.choice([True, False])
    expected = age >= minimum and member
    return make_exercise(
        "booleans",
        seed,
        "choice",
        "Evaluate the condition",
        f"What does `(age >= {minimum}) and member` evaluate to when age is {age} and member is {member}?",
        "and is True only when both sides are True.",
        choices=["True", "False"],
        expected=str(expected),
        solution=str(expected),
        solution_note=f"age >= {minimum} is {age >= minimum} and member is {member}; and needs both to be True, so the result is {expected}.",
        hints=[
            "Evaluate each side separately before combining them.",
            f"Is {age} >= {minimum}? Then check whether member is True as well.",
        ],
        wrong_answer=str(not expected),
    )


@archetype("booleans", "or_not_eval")
def or_not_eval(rng: random.Random, seed: int):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    c = rng.randint(1, 9)
    d = rng.choice([c, rng.randint(1, 9)])
    expected = (not (a < b)) or (c == d)
    return make_exercise(
        "booleans",
        seed,
        "choice",
        "not and or together",
        f"What does `not ({a} < {b}) or ({c} == {d})` evaluate to?",
        "not flips a single truth value; or is True when at least one side is True.",
        choices=["True", "False"],
        expected=str(expected),
        solution=str(expected),
        solution_note=f"{a} < {b} is {a < b}, so not flips it to {not (a < b)}. {c} == {d} is {c == d}. or needs just one True: {expected}.",
        hints=[
            "Work inside-out: evaluate each comparison, apply not, then combine with or.",
            f"not ({a} < {b}) becomes {not (a < b)}; now what does or need to be True?",
        ],
        wrong_answer=str(not expected),
    )


@archetype("booleans", "chained_compare")
def chained_compare(rng: random.Random, seed: int):
    low = rng.randint(1, 8)
    high = low + rng.randint(3, 8)
    x = rng.randint(low - 2, high + 2)
    expected = low <= x < high
    return make_exercise(
        "booleans",
        seed,
        "choice",
        "Chained comparison",
        "What is printed?",
        f"Python reads {low} <= x < {high} as one range check: both parts must hold at once.",
        code=f"x = {x}\nprint({low} <= x < {high})",
        choices=["True", "False"],
        expected=str(expected),
        solution=str(expected),
        solution_note=f"The chain means ({low} <= {x}) and ({x} < {high}): {low <= x} and {x < high}, so {expected}. Note the right side is strict <.",
        hints=[
            "A chained comparison is two checks joined by an invisible and.",
            f"Check both: is {x} at least {low}? Is {x} strictly below {high}?",
        ],
        wrong_answer=str(not expected),
    )


@archetype("booleans", "cross_type_equality")
def cross_type_equality(rng: random.Random, seed: int):
    pool = [
        ('"7" == 7', "False", 'The string "7" and the number 7 are different types, so == says False.'),
        ("0 == 0.0", "True", "int 0 and float 0.0 are both numbers with equal value, so == says True."),
        ("True == 1", "True", "bool is a kind of int in Python: True equals 1 and False equals 0."),
        ('"a" == "A"', "False", "String comparison is case-sensitive; lowercase a and uppercase A differ."),
        ("3 == 3.0", "True", "Numeric types compare by value: 3 and 3.0 are equal."),
        ('"3" == 3', "False", "A string of digits never equals a number without converting first."),
        ("False == 0", "True", "bool is a kind of int: False equals 0."),
    ]
    expr, expected, why = rng.choice(pool)
    return make_exercise(
        "booleans",
        seed,
        "choice",
        "Equal or not?",
        f"What does `{expr}` evaluate to?",
        "== compares values. Numbers of different numeric types can be equal; strings and numbers never are.",
        choices=["True", "False"],
        expected=expected,
        solution=expected,
        solution_note=why,
        hints=[
            "First ask: are the two sides the same kind of thing (text vs number)?",
            "Numbers compare by value across int/float/bool; strings only equal other identical strings.",
        ],
        wrong_answer="True" if expected == "False" else "False",
    )


@archetype("booleans", "bool_precedence")
def bool_precedence(rng: random.Random, seed: int):
    pool = [
        ("True or False and False", "True", "and binds tighter: False and False is False, then True or False is True."),
        ("(True or False) and False", "False", "Parentheses force or first: True, then True and False is False."),
        ("not True or True", "True", "not applies only to the first True: False or True is True."),
        ("not (True or True)", "False", "Parentheses first: True or True is True, then not flips it."),
        ("True and not False", "True", "not False is True, then True and True is True."),
        ("False or not False", "True", "not False is True, then False or True is True."),
        ("not False and False", "False", "not False is True, then True and False is False."),
    ]
    expr, expected, why = rng.choice(pool)
    return make_exercise(
        "booleans",
        seed,
        "choice",
        "Operator order for booleans",
        f"What does `{expr}` evaluate to?",
        "Priority order: not runs first, then and, then or. Parentheses override everything.",
        choices=["True", "False"],
        expected=expected,
        solution=expected,
        solution_note=why,
        hints=[
            "Apply not first, then and, then or, unless parentheses say otherwise.",
            "Rewrite the expression one step at a time, replacing the highest-priority part with its value.",
        ],
        wrong_answer="True" if expected == "False" else "False",
    )


# --------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------


@archetype("functions", "return_vs_print_choice")
def return_vs_print_choice(rng: random.Random, seed: int):
    n = rng.randint(2, 9)
    code = f"def double(x):\n    print(x * 2)\n\nresult = double({n})\nprint(result)"
    return make_exercise(
        "functions",
        seed,
        "choice",
        "print is not return",
        "The code prints two lines. What does the SECOND line show?",
        "A function without return hands back None, no matter what it printed along the way.",
        code=code,
        choices=[str(n * 2), "None", str(n), "SyntaxError"],
        expected="None",
        solution="None",
        solution_note=f"double prints {n * 2} (the first line) but never returns anything, so result is None and the second line shows None.",
        hints=[
            "Printing shows a value on screen; returning hands a value back to the caller. They are different.",
            "There is no return statement in double, so what does result receive?",
        ],
        wrong_answer=str(n * 2),
    )


@archetype("functions", "scale_by")
def scale_by(rng: random.Random, seed: int):
    k = rng.randint(2, 9)
    neg = -rng.randint(1, 6)
    return make_exercise(
        "functions",
        seed,
        "code",
        "Write your first function",
        f"Define scale(x). It should return x multiplied by {k}.",
        "def names the function and its inputs; return sends the answer back to whoever called it.",
        starter="def scale(x):\n    pass\n",
        tests=[
            CodeTest("scale(2)", 2 * k),
            CodeTest("scale(0)", 0),
            CodeTest(f"scale({neg})", neg * k),
        ],
        solution=f"def scale(x):\n    return x * {k}\n",
        solution_note=f"The function body is one rule, return x * {k}, and it works for every input the checker throws at it.",
        hints=[
            "The body needs exactly one line: a return statement.",
            f"Return the parameter x times {k}.",
            f"def scale(x): then indented, return x * {k}",
        ],
        wrong_answer=f"def scale(x):\n    print(x * {k})\n",
    )


@archetype("functions", "rectangle_math")
def rectangle_math(rng: random.Random, seed: int):
    if rng.random() < 0.5:
        kind, formula, calc = "area", "width * height", lambda w, h: w * h
    else:
        kind, formula, calc = "perimeter", "2 * (width + height)", lambda w, h: 2 * (w + h)
    w1, h1 = rng.randint(2, 12), rng.randint(2, 12)
    return make_exercise(
        "functions",
        seed,
        "code",
        f"Rectangle {kind}",
        f"Define {kind}(width, height). Return the {kind} of the rectangle ({formula}).",
        "Functions with two parameters receive both values in order.",
        starter=f"def {kind}(width, height):\n    pass\n",
        tests=[
            CodeTest(f"{kind}({w1}, {h1})", calc(w1, h1)),
            CodeTest(f"{kind}(1, 1)", calc(1, 1)),
            CodeTest(f"{kind}(10, 3)", calc(10, 3)),
        ],
        solution=f"def {kind}(width, height):\n    return {formula}\n",
        solution_note=f"return {formula} expresses the geometry rule once; the tests reuse it with different sizes.",
        hints=[
            "Both parameters arrive ready to use; no conversion needed.",
            f"The formula is {formula}.",
            f"return {formula}",
        ],
        wrong_answer=f"def {kind}(width, height):\n    return width + height\n",
    )


@archetype("functions", "greet_exact")
def greet_exact(rng: random.Random, seed: int):
    template, shown = rng.choice(
        [
            ("Hi, {}!", "Hi, NAME!"),
            ("Hello, {}.", "Hello, NAME."),
            ("Welcome, {}!", "Welcome, NAME!"),
        ]
    )
    n1, n2 = rng.sample(NAMES, 2)
    return make_exercise(
        "functions",
        seed,
        "code",
        "Return exact text",
        f'Define greet(name). Return the text "{shown}" with NAME replaced by the parameter (keep the punctuation exact).',
        "Returning a built string is the bread and butter of real functions.",
        starter="def greet(name):\n    pass\n",
        tests=[
            CodeTest(f"greet({n1!r})", template.format(n1)),
            CodeTest(f"greet({n2!r})", template.format(n2)),
        ],
        solution=f'def greet(name):\n    return f"{template.format("{name}")}"\n',
        solution_note="An f-string drops the parameter straight into the template; return hands the finished text back.",
        hints=[
            "Build the string with an f-string and return it.",
            f'The template is "{shown}" with {{name}} where NAME sits.',
            f'return f"{template.format("{name}")}"',
        ],
        wrong_answer=f'def greet(name):\n    return "{shown}"\n',
    )


@archetype("functions", "average_three")
def average_three(rng: random.Random, seed: int):
    a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 20)
    d, e, f = rng.randint(1, 30), rng.randint(1, 30), rng.randint(1, 30)
    return make_exercise(
        "functions",
        seed,
        "code",
        "Average of three",
        "Define average(a, b, c). Return the mean of the three numbers (their sum divided by 3).",
        "A function can take several inputs and combine them into one output.",
        starter="def average(a, b, c):\n    pass\n",
        tests=[
            CodeTest(f"average({a}, {b}, {c})", (a + b + c) / 3),
            CodeTest(f"average({d}, {e}, {f})", (d + e + f) / 3),
            CodeTest("average(0, 0, 0)", 0.0),
        ],
        solution="def average(a, b, c):\n    return (a + b + c) / 3\n",
        solution_note="Parentheses group the sum before dividing; / keeps the result exact as a float.",
        hints=[
            "Add all three, then divide once.",
            "Without parentheses around the sum, only c would be divided by 3.",
            "return (a + b + c) / 3",
        ],
        wrong_answer="def average(a, b, c):\n    return a + b + c / 3\n",
    )


@archetype("functions", "last_letter")
def last_letter(rng: random.Random, seed: int):
    w1, w2 = rng.sample(WORDS, 2)
    return make_exercise(
        "functions",
        seed,
        "code",
        "Last character",
        "Define last_letter(word). Return the final character of word.",
        "Negative indexing pairs naturally with functions: word[-1] works for any length.",
        starter="def last_letter(word):\n    pass\n",
        tests=[
            CodeTest(f"last_letter({w1!r})", w1[-1]),
            CodeTest(f"last_letter({w2!r})", w2[-1]),
            CodeTest("last_letter('a')", "a"),
        ],
        solution="def last_letter(word):\n    return word[-1]\n",
        solution_note="Index -1 always points at the last character, no matter how long the word is, so no len() math is needed.",
        hints=[
            "You do not need len() for this; there is a shortcut index.",
            "Negative indexes count from the end of the string.",
            "return word[-1]",
        ],
        wrong_answer="def last_letter(word):\n    return word[0]\n",
    )


# --------------------------------------------------------------------------
# conditionals
# --------------------------------------------------------------------------


@archetype("conditionals", "ticket_price")
def ticket_price(rng: random.Random, seed: int):
    youth, standard, senior = rng.randint(8, 14), rng.randint(18, 35), rng.randint(10, 17)
    return make_exercise(
        "conditionals",
        seed,
        "code",
        "Write a branching function",
        f"Define ticket_price(age). Return {youth} for ages under 18, {senior} for ages 65 and up, and {standard} otherwise.",
        "Conditionals let one function handle multiple cases.",
        starter="def ticket_price(age):\n    pass\n",
        tests=[
            CodeTest("ticket_price(12)", youth),
            CodeTest("ticket_price(17)", youth),
            CodeTest("ticket_price(18)", standard),
            CodeTest("ticket_price(30)", standard),
            CodeTest("ticket_price(65)", senior),
        ],
        solution=f"def ticket_price(age):\n    if age < 18:\n        return {youth}\n    if age >= 65:\n        return {senior}\n    return {standard}\n",
        solution_note="Each if handles one band and returns immediately; the final return is the default for everyone else. Boundaries: 17 is still youth, 18 is standard, 65 is already senior.",
        hints=[
            "Handle the special bands first, each with its own return.",
            "Mind the boundaries: under 18 means 17 qualifies and 18 does not; 65 and up includes exactly 65.",
            "if age < 18: return the youth price. if age >= 65: return the senior price. Otherwise return the standard price.",
        ],
        wrong_answer=f"def ticket_price(age):\n    if age <= 18:\n        return {youth}\n    if age >= 65:\n        return {senior}\n    return {standard}\n",
    )


@archetype("conditionals", "clamp")
def clamp(rng: random.Random, seed: int):
    low, high = rng.randint(-8, -1), rng.randint(4, 12)
    mid = (low + high) // 2
    return make_exercise(
        "conditionals",
        seed,
        "code",
        "Clamp a value",
        "Define clamp(value, low, high). Return low when value is below low, high when value is above high, otherwise value itself.",
        "Clamping keeps a number inside a range; it is two comparisons and a default.",
        starter="def clamp(value, low, high):\n    pass\n",
        tests=[
            CodeTest(f"clamp({low - 5}, {low}, {high})", low),
            CodeTest(f"clamp({high + 5}, {low}, {high})", high),
            CodeTest(f"clamp({mid}, {low}, {high})", mid),
            CodeTest(f"clamp({low}, {low}, {high})", low),
        ],
        solution="def clamp(value, low, high):\n    if value < low:\n        return low\n    if value > high:\n        return high\n    return value\n",
        solution_note="The two ifs catch the out-of-range cases; anything that survives both checks is already in range and is returned unchanged.",
        hints=[
            "Three outcomes: too small, too big, or fine as is.",
            "Return early inside each if; the last line handles the in-range case.",
            "if value < low: return low. if value > high: return high. Otherwise return value.",
        ],
        wrong_answer="def clamp(value, low, high):\n    return value\n",
    )


@archetype("conditionals", "branch_trace")
def branch_trace(rng: random.Random, seed: int):
    labels = rng.choice([("low", "mid", "high"), ("cold", "warm", "hot"), ("small", "medium", "large")])
    a = rng.randint(5, 12)
    b = a + rng.randint(4, 10)
    x = rng.randint(0, b + 6)
    expected = labels[0] if x < a else (labels[1] if x < b else labels[2])
    return make_exercise(
        "conditionals",
        seed,
        "short",
        "Which branch runs?",
        "What word is printed?",
        "Python checks if/elif conditions top to bottom and runs only the first one that is True.",
        code=(
            f"x = {x}\n"
            f"if x < {a}:\n"
            f'    print("{labels[0]}")\n'
            f"elif x < {b}:\n"
            f'    print("{labels[1]}")\n'
            f"else:\n"
            f'    print("{labels[2]}")'
        ),
        expected=expected,
        solution=expected,
        solution_note=f"x is {x}: the checks run in order (x < {a}? then x < {b}?), and the first True branch wins, printing {expected!r}.",
        hints=[
            "Test the conditions in order; stop at the first True.",
            f"Is {x} < {a}? If not, is {x} < {b}? Only then does else apply.",
        ],
        wrong_answer=labels[(labels.index(expected) + 1) % 3],
    )


@archetype("conditionals", "grade_bands")
def grade_bands(rng: random.Random, seed: int):
    cut_a = rng.randint(88, 93)
    cut_b = cut_a - rng.randint(9, 12)
    cut_c = cut_b - rng.randint(9, 12)
    return make_exercise(
        "conditionals",
        seed,
        "code",
        "Letter grades",
        f'Define grade(score). Return "A" for {cut_a} and above, "B" for {cut_b} and above, "C" for {cut_c} and above, otherwise "F".',
        "Ordered elif chains express bands cleanly when checked from highest to lowest.",
        starter="def grade(score):\n    pass\n",
        tests=[
            CodeTest(f"grade({cut_a})", "A"),
            CodeTest(f"grade({cut_a - 1})", "B"),
            CodeTest(f"grade({cut_b})", "B"),
            CodeTest(f"grade({cut_c})", "C"),
            CodeTest(f"grade({cut_c - 1})", "F"),
        ],
        solution=(
            f'def grade(score):\n    if score >= {cut_a}:\n        return "A"\n'
            f'    elif score >= {cut_b}:\n        return "B"\n'
            f'    elif score >= {cut_c}:\n        return "C"\n    return "F"\n'
        ),
        solution_note="Checking the highest band first means each elif only sees scores that already failed the bands above it.",
        hints=[
            "Check the highest cutoff first and work downward.",
            'Each band needs >= its cutoff; the final return "F" catches everything else.',
            f'if score >= {cut_a}: "A", elif score >= {cut_b}: "B", elif score >= {cut_c}: "C", else "F".',
        ],
        wrong_answer=(
            f'def grade(score):\n    if score > {cut_a}:\n        return "A"\n'
            f'    elif score > {cut_b}:\n        return "B"\n'
            f'    elif score > {cut_c}:\n        return "C"\n    return "F"\n'
        ),
    )


@archetype("conditionals", "fizz_pair")
def fizz_pair(rng: random.Random, seed: int):
    d1, d2 = rng.choice([(2, 3), (2, 5), (3, 4), (3, 5), (4, 5)])
    if rng.random() < 0.5:
        d1, d2 = d2, d1
    n1, n2 = rng.choice([("fizz", "buzz"), ("ping", "pong"), ("tick", "tock")])
    neither = d1 * d2 + 1
    return make_exercise(
        "conditionals",
        seed,
        "code",
        "Divisibility labels",
        (
            f'Define label(n). Return "{n1}{n2}" when n is divisible by both {d1} and {d2}, '
            f'"{n1}" when divisible only by {d1}, "{n2}" when divisible only by {d2}, '
            "otherwise return str(n)."
        ),
        "Check the most specific condition (both) first, or it can never be reached.",
        starter="def label(n):\n    pass\n",
        tests=[
            CodeTest(f"label({d1 * d2})", f"{n1}{n2}"),
            CodeTest(f"label({d1})", n1),
            CodeTest(f"label({d2})", n2),
            CodeTest(f"label({neither})", str(neither)),
        ],
        solution=(
            f"def label(n):\n"
            f"    if n % {d1} == 0 and n % {d2} == 0:\n        return \"{n1}{n2}\"\n"
            f"    if n % {d1} == 0:\n        return \"{n1}\"\n"
            f"    if n % {d2} == 0:\n        return \"{n2}\"\n"
            f"    return str(n)\n"
        ),
        solution_note=f"The both-case must come first: a multiple of {d1 * d2} also passes the single checks, so testing it later would never trigger.",
        hints=[
            "n % d == 0 is the divisibility test.",
            "Order matters: test the divisible-by-both case before the single cases.",
            f"First: n % {d1} == 0 and n % {d2} == 0. Then each single check. Finally str(n).",
        ],
        wrong_answer=(
            f"def label(n):\n"
            f"    if n % {d1} == 0:\n        return \"{n1}\"\n"
            f"    if n % {d2} == 0:\n        return \"{n2}\"\n"
            f"    if n % {d1} == 0 and n % {d2} == 0:\n        return \"{n1}{n2}\"\n"
            f"    return str(n)\n"
        ),
    )


# --------------------------------------------------------------------------
# loops
# --------------------------------------------------------------------------


@archetype("loops", "sum_multiples")
def sum_multiples(rng: random.Random, seed: int):
    multiple = rng.randint(2, 6)
    limit = rng.randint(18, 36)
    return make_exercise(
        "loops",
        seed,
        "code",
        "Accumulate with a loop",
        f"Define sum_multiples(limit). Return the sum of positive numbers below limit divisible by {multiple}.",
        "A running total is one of the most useful loop patterns.",
        starter="def sum_multiples(limit):\n    total = 0\n    return total\n",
        tests=[
            CodeTest("sum_multiples(1)", 0),
            CodeTest(f"sum_multiples({limit})", sum(x for x in range(limit) if x % multiple == 0)),
            CodeTest(f"sum_multiples({limit + 7})", sum(x for x in range(limit + 7) if x % multiple == 0)),
        ],
        solution=(
            f"def sum_multiples(limit):\n    total = 0\n    for x in range(1, limit):\n"
            f"        if x % {multiple} == 0:\n            total += x\n    return total\n"
        ),
        solution_note=f"The loop visits every number below limit; the if filters for multiples of {multiple}; total += x accumulates them.",
        hints=[
            "Start a total at 0 before the loop and add to it inside.",
            f"range(1, limit) stops before limit; test each x with x % {multiple} == 0.",
            "for x in range(1, limit): if divisible, total += x. Return total after the loop.",
        ],
        wrong_answer="def sum_multiples(limit):\n    return limit\n",
    )


@archetype("loops", "count_vowels")
def count_vowels(rng: random.Random, seed: int):
    word = rng.choice(WORDS)
    fruit = rng.choice(FRUITS)
    count_w = sum(1 for ch in word if ch in "aeiou")
    count_f = sum(1 for ch in fruit if ch in "aeiou")
    return make_exercise(
        "loops",
        seed,
        "code",
        "Count matching characters",
        'Define count_vowels(text). Return how many characters of text are vowels ("a", "e", "i", "o", "u", lowercase only).',
        "A for loop can walk through a string character by character.",
        starter="def count_vowels(text):\n    count = 0\n    return count\n",
        tests=[
            CodeTest(f"count_vowels({word!r})", count_w),
            CodeTest(f"count_vowels({fruit!r})", count_f),
            CodeTest("count_vowels('xyz')", 0),
            CodeTest("count_vowels('')", 0),
        ],
        solution=(
            "def count_vowels(text):\n    count = 0\n    for ch in text:\n"
            "        if ch in \"aeiou\":\n            count += 1\n    return count\n"
        ),
        solution_note='for ch in text visits each character; ch in "aeiou" is a compact membership test; count += 1 tallies the hits.',
        hints=[
            "Loop over the string itself: for ch in text.",
            'The test ch in "aeiou" is True for any vowel character.',
            "Start count at 0, add 1 inside the if, return count at the end.",
        ],
        wrong_answer="def count_vowels(text):\n    return len(text)\n",
    )


@archetype("loops", "product_up_to")
def product_up_to(rng: random.Random, seed: int):
    n = rng.randint(4, 7)
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    return make_exercise(
        "loops",
        seed,
        "code",
        "Multiply a range",
        "Define product_up_to(n). Return 1 * 2 * 3 * ... * n (the product of every whole number from 1 to n).",
        "A product accumulator starts at 1, not 0: multiplying by 0 would erase everything.",
        starter="def product_up_to(n):\n    result = 1\n    return result\n",
        tests=[
            CodeTest("product_up_to(1)", 1),
            CodeTest("product_up_to(3)", 6),
            CodeTest(f"product_up_to({n})", fact),
        ],
        solution=(
            "def product_up_to(n):\n    result = 1\n    for i in range(1, n + 1):\n"
            "        result *= i\n    return result\n"
        ),
        solution_note="range(1, n + 1) includes n itself because range stops one short; result *= i multiplies each number in.",
        hints=[
            "This is the sum pattern with * instead of +, and a different starting value.",
            "range(1, n + 1) is needed to include n; range(1, n) would stop at n - 1.",
            "result = 1, then for i in range(1, n + 1): result *= i.",
        ],
        wrong_answer=(
            "def product_up_to(n):\n    result = 0\n    for i in range(1, n + 1):\n"
            "        result *= i\n    return result\n"
        ),
    )


@archetype("loops", "loop_trace")
def loop_trace(rng: random.Random, seed: int):
    a = rng.randint(1, 4)
    b = a + rng.randint(5, 12)
    step = rng.randint(1, 3)
    expected = sum(range(a, b, step))
    return make_exercise(
        "loops",
        seed,
        "short",
        "Trace the accumulation",
        "What number is printed?",
        f"range({a}, {b}, {step}) yields {a}, then jumps by {step}, stopping before {b}.",
        code=f"total = 0\nfor i in range({a}, {b}, {step}):\n    total += i\nprint(total)",
        expected=expected,
        solution=str(expected),
        solution_note=f"The loop visits {list(range(a, b, step))}; their sum is {expected}.",
        hints=[
            f"First list the values range({a}, {b}, {step}) produces; remember it stops before {b}.",
            f"The values are {list(range(a, b, step))}. Add them up.",
        ],
        wrong_answer=str(sum(range(a, b + 1, step))),
    )


@archetype("loops", "repeat_word")
def repeat_word(rng: random.Random, seed: int):
    word = rng.choice(["ha", "no", "go", "yo"])
    other = rng.choice(["hey", "ok"])
    n = rng.randint(2, 4)
    return make_exercise(
        "loops",
        seed,
        "code",
        "Build a string in a loop",
        "Define repeat_word(word, n). Return word repeated n times with nothing between the copies (n may be 0, giving an empty string).",
        "Strings accumulate just like numbers: start empty and add pieces each pass.",
        starter="def repeat_word(word, n):\n    result = \"\"\n    return result\n",
        tests=[
            CodeTest(f"repeat_word({word!r}, {n})", word * n),
            CodeTest(f"repeat_word({other!r}, 2)", other * 2),
            CodeTest(f"repeat_word({word!r}, 0)", ""),
        ],
        solution=(
            "def repeat_word(word, n):\n    result = \"\"\n    for _ in range(n):\n"
            "        result += word\n    return result\n"
        ),
        solution_note="An empty string is the accumulator; each loop pass glues one more copy on. (word * n is the built-in shortcut for the same thing.)",
        hints=[
            'Start with result = "" and add word to it n times.',
            "range(n) runs the body exactly n times; the loop variable itself is not needed.",
            "for _ in range(n): result += word",
        ],
        wrong_answer="def repeat_word(word, n):\n    return word\n",
    )


# --------------------------------------------------------------------------
# while_loops
# --------------------------------------------------------------------------


@archetype("while_loops", "halving_steps")
def halving_steps(rng: random.Random, seed: int):
    n = rng.randint(8, 60)
    steps = 0
    value = n
    while value > 1:
        value //= 2
        steps += 1
    return make_exercise(
        "while_loops",
        seed,
        "code",
        "Count the halvings",
        "Define steps_to_one(n). Repeatedly floor-divide n by 2 (n //= 2) until it is 1 or less, and return how many divisions that took. steps_to_one(1) is 0.",
        "while runs as long as its condition holds; a counter records how many times.",
        starter="def steps_to_one(n):\n    steps = 0\n    return steps\n",
        tests=[
            CodeTest("steps_to_one(1)", 0),
            CodeTest("steps_to_one(2)", 1),
            CodeTest(f"steps_to_one({n})", steps),
        ],
        solution=(
            "def steps_to_one(n):\n    steps = 0\n    while n > 1:\n"
            "        n //= 2\n        steps += 1\n    return steps\n"
        ),
        solution_note="The condition n > 1 stops the loop once n reaches 1; each pass halves n and bumps the counter.",
        hints=[
            "The loop condition should keep the loop alive while n is still above 1.",
            "Inside the loop: halve n with n //= 2, then add one to the counter.",
            "while n > 1: n //= 2; steps += 1. Return steps.",
        ],
        wrong_answer="def steps_to_one(n):\n    return n // 2\n",
    )


@archetype("while_loops", "while_trace")
def while_trace(rng: random.Random, seed: int):
    x = rng.randint(2, 5)
    limit = rng.randint(20, 90)
    value = x
    while value < limit:
        value *= 2
    return make_exercise(
        "while_loops",
        seed,
        "short",
        "Trace the doubling",
        "What number is printed?",
        "The loop keeps doubling x while it stays under the limit; the print happens after the loop ends.",
        code=f"x = {x}\nwhile x < {limit}:\n    x = x * 2\nprint(x)",
        expected=value,
        solution=str(value),
        solution_note=f"x doubles: {x} -> ... -> {value}. The loop only stops once x is {limit} or more, so the printed value is the first double at or past {limit}.",
        hints=[
            f"Write out the doubling sequence starting at {x}.",
            f"The loop exits at the first value that is {limit} or more; that value is what prints.",
        ],
        wrong_answer=str(value // 2),
    )


@archetype("while_loops", "digit_sum")
def digit_sum(rng: random.Random, seed: int):
    n = rng.randint(100, 9999)
    total = sum(int(d) for d in str(n))
    return make_exercise(
        "while_loops",
        seed,
        "code",
        "Sum the digits",
        "Define digit_sum(n). Return the sum of the digits of the positive integer n. Use % 10 to grab the last digit and // 10 to drop it.",
        "n % 10 peels off the last digit; n //= 10 shrinks the number until nothing is left.",
        starter="def digit_sum(n):\n    total = 0\n    return total\n",
        tests=[
            CodeTest("digit_sum(7)", 7),
            CodeTest("digit_sum(10)", 1),
            CodeTest(f"digit_sum({n})", total),
        ],
        solution=(
            "def digit_sum(n):\n    total = 0\n    while n > 0:\n"
            "        total += n % 10\n        n //= 10\n    return total\n"
        ),
        solution_note="Each pass adds the last digit (n % 10) and removes it (n //= 10); the loop ends when n hits 0.",
        hints=[
            "n % 10 gives the last digit of n; n // 10 is n without that digit.",
            "Loop while n > 0, accumulating digits into a total.",
            "while n > 0: total += n % 10; n //= 10.",
        ],
        wrong_answer="def digit_sum(n):\n    return n % 10\n",
    )


@archetype("while_loops", "first_power_above")
def first_power_above(rng: random.Random, seed: int):
    base = rng.randint(2, 5)
    limit = rng.randint(20, 300)
    value = base
    while value <= limit:
        value *= base
    return make_exercise(
        "while_loops",
        seed,
        "code",
        "First power past the limit",
        f"Define first_power_above(base, limit). Starting from base itself, keep multiplying by base and return the first value STRICTLY greater than limit. first_power_above({base}, {base}) is {base * base}.",
        "A while loop can search: keep growing a value until a condition is finally satisfied.",
        starter="def first_power_above(base, limit):\n    value = base\n    return value\n",
        tests=[
            CodeTest(f"first_power_above({base}, {limit})", value),
            CodeTest("first_power_above(2, 1)", 2),
            CodeTest(f"first_power_above({base}, {base})", base * base),
        ],
        solution=(
            "def first_power_above(base, limit):\n    value = base\n"
            "    while value <= limit:\n        value *= base\n    return value\n"
        ),
        solution_note="The condition value <= limit keeps multiplying while the value is not yet past the limit, so the first strictly-greater power is what escapes the loop. Note first_power_above(2, 1) is 2: the starting value already qualifies.",
        hints=[
            "Start value at base and multiply inside the loop.",
            "Strictly greater means the loop should continue while value <= limit.",
            "while value <= limit: value *= base. Then return value.",
        ],
        wrong_answer=(
            "def first_power_above(base, limit):\n    value = base\n"
            "    while value < limit:\n        value *= base\n    return value\n"
        ),
    )


@archetype("while_loops", "last_positive")
def last_positive(rng: random.Random, seed: int):
    step = rng.randint(3, 9)
    start = step * rng.randint(3, 8) + rng.randint(1, step - 1)
    value = start
    while value - step > 0:
        value -= step
    divisible_start = step * rng.randint(3, 6)
    return make_exercise(
        "while_loops",
        seed,
        "code",
        "Subtract until you must stop",
        "Define last_positive(start, step). Repeatedly subtract step from start, but stop before the value would drop to zero or below. Return the last positive value. If start is already at most step, return start.",
        "Look-ahead conditions (value - step > 0) stop a loop one step before it goes too far.",
        starter="def last_positive(start, step):\n    value = start\n    return value\n",
        tests=[
            CodeTest(f"last_positive({start}, {step})", value),
            CodeTest(f"last_positive({divisible_start}, {step})", step),
            CodeTest(f"last_positive(2, {step})", 2),
        ],
        solution=(
            "def last_positive(start, step):\n    value = start\n"
            "    while value - step > 0:\n        value -= step\n    return value\n"
        ),
        solution_note="The condition peeks one subtraction ahead: it only subtracts when the result would still be positive, so the loop lands exactly on the last positive value.",
        hints=[
            "Do not subtract first and check later; check what WOULD happen before subtracting.",
            "The loop should run while value - step is still positive.",
            "while value - step > 0: value -= step.",
        ],
        wrong_answer=(
            "def last_positive(start, step):\n    value = start\n"
            "    while value > 0:\n        value -= step\n    return value\n"
        ),
    )


# --------------------------------------------------------------------------
# nested_logic
# --------------------------------------------------------------------------


@archetype("nested_logic", "pair_count")
def pair_count(rng: random.Random, seed: int):
    n = rng.randint(5, 9)
    d = rng.randint(2, 4)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (i + j) % d == 0:
                count += 1
    return make_exercise(
        "nested_logic",
        seed,
        "code",
        "Count qualifying pairs",
        f"Define count_pairs(n, d). Count the pairs (i, j) with 0 <= i < j < n where (i + j) is divisible by d, and return the count.",
        "A nested loop with j starting at i + 1 visits each unordered pair exactly once.",
        starter="def count_pairs(n, d):\n    count = 0\n    return count\n",
        tests=[
            CodeTest(f"count_pairs({n}, {d})", count),
            CodeTest("count_pairs(1, 2)", 0),
            CodeTest("count_pairs(3, 2)", sum(1 for i in range(3) for j in range(i + 1, 3) if (i + j) % 2 == 0)),
        ],
        solution=(
            "def count_pairs(n, d):\n    count = 0\n    for i in range(n):\n"
            "        for j in range(i + 1, n):\n            if (i + j) % d == 0:\n"
            "                count += 1\n    return count\n"
        ),
        solution_note="Starting the inner loop at i + 1 guarantees i < j, so no pair is counted twice and no pair pairs a number with itself.",
        hints=[
            "You need a loop inside a loop: one for i, one for j.",
            "Start j at i + 1 so each pair is seen once and i is always less than j.",
            "for i in range(n): for j in range(i + 1, n): if (i + j) % d == 0: count += 1.",
        ],
        wrong_answer=(
            "def count_pairs(n, d):\n    count = 0\n    for i in range(n):\n"
            "        for j in range(n):\n            if (i + j) % d == 0:\n"
            "                count += 1\n    return count\n"
        ),
    )


@archetype("nested_logic", "nested_trace")
def nested_trace(rng: random.Random, seed: int):
    r = rng.randint(3, 5)
    c = rng.randint(3, 5)
    expected = sum(i * j for i in range(1, r) for j in range(1, c))
    return make_exercise(
        "nested_logic",
        seed,
        "short",
        "Trace the nested loop",
        "What number is printed?",
        "The inner loop runs completely for every single pass of the outer loop.",
        code=(
            f"total = 0\nfor i in range(1, {r}):\n    for j in range(1, {c}):\n"
            f"        total += i * j\nprint(total)"
        ),
        expected=expected,
        solution=str(expected),
        solution_note=(
            f"i takes {list(range(1, r))} and j takes {list(range(1, c))}; every combination contributes i * j. "
            f"That is (sum of i) * (sum of j) = {sum(range(1, r))} * {sum(range(1, c))} = {expected}."
        ),
        hints=[
            f"i runs over {list(range(1, r))}; for each i, j runs over {list(range(1, c))}.",
            "Add i * j for every combination, or notice the total factors as (1+...+{}) * (1+...+{}).".format(r - 1, c - 1),
        ],
        wrong_answer=str(sum(i * j for i in range(1, r + 1) for j in range(1, c + 1))),
    )


@archetype("nested_logic", "banner_grid")
def banner_grid(rng: random.Random, seed: int):
    rows = rng.randint(2, 4)
    cols = rng.randint(3, 6)
    expected = "\n".join("*" * cols for _ in range(rows))
    return make_exercise(
        "nested_logic",
        seed,
        "code",
        "Draw a star rectangle",
        'Define banner(rows, cols). Return a single string of rows lines, each made of cols asterisks, with a newline "\\n" between lines (no trailing newline).',
        "Building multi-line text needs a condition: add the newline between rows, not after the last one.",
        starter="def banner(rows, cols):\n    result = \"\"\n    return result\n",
        tests=[
            CodeTest("banner(1, 3)", "***"),
            CodeTest("banner(2, 2)", "**\n**"),
            CodeTest(f"banner({rows}, {cols})", expected),
        ],
        solution=(
            "def banner(rows, cols):\n    result = \"\"\n    for r in range(rows):\n"
            "        result += \"*\" * cols\n        if r < rows - 1:\n"
            "            result += \"\\n\"\n    return result\n"
        ),
        solution_note='Each pass adds one full row; the if r < rows - 1 check adds separators only BETWEEN rows, avoiding a stray trailing newline.',
        hints=[
            'Build one row with "*" * cols, then repeat for each row.',
            "The tricky part is the newline: it belongs between rows only.",
            "Add the row, then add \\n only when r < rows - 1.",
        ],
        wrong_answer=(
            "def banner(rows, cols):\n    result = \"\"\n    for r in range(rows):\n"
            "        result += \"*\" * cols + \"\\n\"\n    return result\n"
        ),
    )


@archetype("nested_logic", "inner_runs_choice")
def inner_runs_choice(rng: random.Random, seed: int):
    while True:
        r = rng.randint(3, 6)
        c = rng.randint(3, 6)
        if r != c:
            break
    expected = str(r * c)
    return make_exercise(
        "nested_logic",
        seed,
        "choice",
        "How many prints?",
        "How many lines does this code print in total?",
        "Nested loops multiply: the inner body runs (outer passes) times (inner passes).",
        code=f'for i in range({r}):\n    for j in range({c}):\n        print("hit")',
        choices=[str(r + c), expected, str(r), str(c)],
        expected=expected,
        solution=expected,
        solution_note=f"The outer loop runs {r} times, and each of those runs the inner loop {c} times: {r} * {c} = {expected} prints.",
        hints=[
            "For every single outer pass, the whole inner loop runs from scratch.",
            f"That is {r} outer passes times {c} inner passes each.",
        ],
        wrong_answer=str(r + c),
    )


@archetype("nested_logic", "divisor_count")
def divisor_count(rng: random.Random, seed: int):
    n = rng.choice([12, 18, 20, 24, 28, 30, 36])
    prime = rng.choice([7, 11, 13])
    count = sum(1 for i in range(1, n + 1) if n % i == 0)
    return make_exercise(
        "nested_logic",
        seed,
        "code",
        "Count the divisors",
        "Define count_divisors(n). Return how many whole numbers from 1 to n divide n evenly.",
        "A loop plus a condition is a counting filter: visit candidates, count the ones that qualify.",
        starter="def count_divisors(n):\n    count = 0\n    return count\n",
        tests=[
            CodeTest("count_divisors(1)", 1),
            CodeTest(f"count_divisors({prime})", 2),
            CodeTest(f"count_divisors({n})", count),
        ],
        solution=(
            "def count_divisors(n):\n    count = 0\n    for i in range(1, n + 1):\n"
            "        if n % i == 0:\n            count += 1\n    return count\n"
        ),
        solution_note="range(1, n + 1) must include n itself (every number divides itself); n % i == 0 detects each divisor.",
        hints=[
            "Try every candidate i from 1 up to and including n.",
            "i divides n evenly exactly when n % i == 0.",
            "for i in range(1, n + 1): if n % i == 0: count += 1.",
        ],
        wrong_answer=(
            "def count_divisors(n):\n    count = 0\n    for i in range(1, n):\n"
            "        if n % i == 0:\n            count += 1\n    return count\n"
        ),
    )
