from __future__ import annotations

import random

from ..models import CodeTest
from .registry import archetype, make_exercise
from .support import NAMES, SIMPLE_WORDS, WORDS


# --------------------------------------------------------------------------
# print_output
# --------------------------------------------------------------------------


@archetype("print_output", "print_exact_word")
def print_exact_word(rng: random.Random, seed: int):
    word = rng.choice(SIMPLE_WORDS)
    return make_exercise(
        "print_output",
        seed,
        "code",
        "Print one word",
        f"Write code that prints exactly {word!r}.",
        "print(...) sends text to the screen. The checker captures printed output.",
        starter="# write one print call\n",
        tests=[CodeTest("__stdout__.strip()", word)],
        solution=f"print({word!r})\n",
        solution_note="print sends the text between the parentheses to the screen. The quotes mark where the text starts and ends; they are not printed.",
        hints=[
            "print(...) is a function call: the thing you want shown goes inside the parentheses.",
            f"Text must be wrapped in quotes: {word!r} with quotes is text, without quotes Python thinks it is a variable name.",
            f"The full call is one line: print with ({word!r}) right after it.",
        ],
        wrong_answer=f"print({(word + '!')!r})\n",
    )


@archetype("print_output", "print_two_lines")
def print_two_lines(rng: random.Random, seed: int):
    first, second = rng.sample(SIMPLE_WORDS, 2)
    return make_exercise(
        "print_output",
        seed,
        "code",
        "Print two lines",
        f"Print {first!r} on the first line and {second!r} on the second line.",
        "Each print call ends the line, so two print calls make two lines.",
        starter="# two print calls\n",
        tests=[CodeTest("__stdout__.strip()", f"{first}\n{second}")],
        solution=f"print({first!r})\nprint({second!r})\n",
        solution_note="Every print call adds a newline at the end, so two calls in a row produce two separate lines.",
        hints=[
            "One print call produces one line of output.",
            "You need two separate print calls, one per word, in the right order.",
            f"Line 1: print({first!r}). Line 2: print({second!r}).",
        ],
        wrong_answer=f"print({first!r}, {second!r})\n",
    )


@archetype("print_output", "print_greeting")
def print_greeting(rng: random.Random, seed: int):
    name = rng.choice(NAMES)
    expected = f"Hello, {name}!"
    return make_exercise(
        "print_output",
        seed,
        "code",
        "Print an exact sentence",
        f"Print exactly: {expected}",
        "Printed text must match character for character: capitals, commas, and punctuation all count.",
        starter="# match the text exactly\n",
        tests=[CodeTest("__stdout__.strip()", expected)],
        solution=f"print({expected!r})\n",
        solution_note="The whole sentence, punctuation included, goes inside one pair of quotes.",
        hints=[
            "Put the entire sentence inside one pair of quotes.",
            "Check the capital H, the comma, and the exclamation mark; the checker compares every character.",
            f"print({expected!r}) reproduces the sentence exactly.",
        ],
        wrong_answer=f"print({('hello, ' + name)!r})\n",
    )


@archetype("print_output", "quoted_math_choice")
def quoted_math_choice(rng: random.Random, seed: int):
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    code = f'print("{a} + {b}")\nprint({a} + {b})'
    expected = f"{a} + {b}"
    choices = [expected, str(a + b), f"{a}{b}", "SyntaxError"]
    return make_exercise(
        "print_output",
        seed,
        "choice",
        "Quotes change everything",
        "What does the FIRST line of output show?",
        "Anything inside quotes is plain text. Python does not do math inside quotes.",
        code=code,
        choices=choices,
        expected=expected,
        solution=expected,
        solution_note=f'The first print receives the text "{a} + {b}" in quotes, so it shows the characters as written. The second line (without quotes) would do the math and show {a + b}.',
        hints=[
            "Quotes turn things into text; text is shown exactly as written.",
            "Only the second print, without quotes, asks Python to calculate.",
        ],
        wrong_answer=str(a + b),
    )


@archetype("print_output", "print_apostrophe")
def print_apostrophe(rng: random.Random, seed: int):
    sentence = rng.choice(["It's go time", "Don't stop now", "You're doing well", "That's the idea"])
    return make_exercise(
        "print_output",
        seed,
        "code",
        "Quotes inside text",
        f"Print exactly: {sentence}",
        'Text with an apostrophe is easiest to write inside double quotes: "It\'s fine".',
        starter="# careful: the text contains an apostrophe\n",
        tests=[CodeTest("__stdout__.strip()", sentence)],
        solution=f'print("{sentence}")\n',
        solution_note="Wrapping the text in double quotes lets the single quote (apostrophe) live inside without ending the string.",
        hints=[
            "The apostrophe in the text will clash with single-quote string markers.",
            "Use double quotes around the whole sentence so the apostrophe is just a normal character.",
            f'print("{sentence}") is the cleanest way to write it.',
        ],
        wrong_answer=f'print("{sentence.replace(chr(39), "")}")\n',
    )


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------


@archetype("numbers", "predict_precedence")
def predict_precedence(rng: random.Random, seed: int):
    a, b, c, d = rng.randint(2, 9), rng.randint(3, 12), rng.randint(2, 7), rng.randint(1, 5)
    expected = a + b * c - d
    return make_exercise(
        "numbers",
        seed,
        "short",
        "Predict the number",
        "What number is printed?",
        "Python evaluates multiplication before addition and subtraction.",
        code=f"result = {a} + {b} * {c} - {d}\nprint(result)",
        expected=expected,
        solution=str(expected),
        solution_note=f"Multiplication happens first: {b} * {c} = {b * c}. Then left to right: {a} + {b * c} - {d} = {expected}.",
        hints=[
            "Multiplication and division run before addition and subtraction.",
            f"Start with {b} * {c}, then apply the + and - from left to right.",
        ],
        wrong_answer=str((a + b) * c - d),
    )


@archetype("numbers", "int_div_mod")
def int_div_mod(rng: random.Random, seed: int):
    b = rng.randint(3, 9)
    a = rng.randint(2, 11) * b + rng.randint(1, b - 1)
    expected = f"{a // b} {a % b}"
    return make_exercise(
        "numbers",
        seed,
        "short",
        "Floor division and remainder",
        "What two numbers are printed, separated by one space?",
        "// gives the whole-number quotient; % gives the remainder after that division.",
        code=f"print({a} // {b}, {a} % {b})",
        expected=expected,
        solution=expected,
        solution_note=f"{a} // {b} asks how many whole {b}s fit into {a} ({a // b}); {a} % {b} is what is left over ({a % b}).",
        hints=[
            "// throws away the fraction; % keeps only the leftover.",
            f"How many whole times does {b} fit into {a}? What remains after removing them?",
        ],
        wrong_answer=f"{a / b:.2f}",
    )


@archetype("numbers", "power_expression")
def power_expression(rng: random.Random, seed: int):
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    expected = (a + b) ** 2
    return make_exercise(
        "numbers",
        seed,
        "code",
        "Use the power operator",
        f"Create a variable named result holding ({a} + {b}) squared. Use parentheses and the ** operator.",
        "** raises a number to a power. Parentheses force the addition to happen first.",
        starter="result = 0  # replace 0 with the expression\n",
        tests=[CodeTest("result", expected)],
        solution=f"result = ({a} + {b}) ** 2\n",
        solution_note=f"Parentheses make the sum happen first ({a + b}), then ** 2 squares it: {expected}.",
        hints=[
            "Squaring a value means raising it to the power 2 with **.",
            "Without parentheses, ** would run before +, which is not what you want here.",
            f"result = ({a} + {b}) ** 2",
        ],
        wrong_answer=f"result = {a} + {b} ** 2\n",
    )


@archetype("numbers", "division_type_choice")
def division_type_choice(rng: random.Random, seed: int):
    while True:
        b = rng.randint(2, 6)
        k = rng.randint(2, 6)
        if b != k:
            break
    a = b * k
    expected = f"{a / b}"
    choices = [str(k), expected, str(b), "TypeError"]
    return make_exercise(
        "numbers",
        seed,
        "choice",
        "What does / produce?",
        "What exactly is printed?",
        "The / operator always produces a float, even when the division is exact.",
        code=f"print({a} / {b})",
        choices=choices,
        expected=expected,
        solution=expected,
        solution_note=f"/ is true division and always returns a float, so {a} / {b} is {expected}, not {k}. Use // when you want a whole number.",
        hints=[
            "There are two division operators; / and // behave differently.",
            "Even when the answer is a whole number, / hands back a float like 4.0.",
        ],
        wrong_answer=str(k),
    )


@archetype("numbers", "round_predict")
def round_predict(rng: random.Random, seed: int):
    whole = rng.randint(2, 19)
    tenths = rng.randint(0, 9)
    hundredths = rng.choice([1, 2, 3, 4, 6, 7, 8, 9])
    x = round(whole + tenths / 10 + hundredths / 100, 2)
    expected = round(x, 1)
    return make_exercise(
        "numbers",
        seed,
        "short",
        "Round to one decimal",
        "What number is printed?",
        "round(value, 1) keeps one digit after the decimal point, rounding by the digit that follows.",
        code=f"print(round({x}, 1))",
        expected=expected,
        solution=str(expected),
        solution_note=f"The hundredths digit is {hundredths}, so the tenths digit {'rounds up' if hundredths >= 5 else 'stays as it is'}: {expected}.",
        hints=[
            "Look at the second digit after the decimal point to decide the rounding direction.",
            "5 or more rounds the previous digit up; 4 or less leaves it alone.",
        ],
        wrong_answer=str(x),
    )


# --------------------------------------------------------------------------
# variables
# --------------------------------------------------------------------------


@archetype("variables", "track_reassignment")
def track_reassignment(rng: random.Random, seed: int):
    start, boost, scale = rng.randint(3, 15), rng.randint(2, 8), rng.randint(2, 4)
    expected = (start + boost) * scale
    return make_exercise(
        "variables",
        seed,
        "short",
        "Track the variable",
        "What value does score hold at the end?",
        "Read assignments from top to bottom; the latest assignment wins.",
        code=f"score = {start}\nscore = score + {boost}\nscore = score * {scale}\nprint(score)",
        expected=expected,
        solution=str(expected),
        solution_note=f"score starts at {start}, becomes {start + boost} after the addition, then {expected} after multiplying by {scale}.",
        hints=[
            "Track the value line by line; each assignment replaces the old value.",
            f"After line 2, score is {start} + {boost}. Feed that into line 3.",
        ],
        wrong_answer=str(start),
    )


@archetype("variables", "swap_with_temp")
def swap_with_temp(rng: random.Random, seed: int):
    while True:
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        if a != b:
            break
    expected = f"{b} {a}"
    return make_exercise(
        "variables",
        seed,
        "short",
        "Follow the swap",
        "What two values are printed, separated by one space?",
        "A temporary variable lets two names trade values without losing one.",
        code=f"x = {a}\ny = {b}\ntemp = x\nx = y\ny = temp\nprint(x, y)",
        expected=expected,
        solution=expected,
        solution_note=f"temp saves the original x ({a}) before x is overwritten with {b}; y then takes the saved {a}. The values have swapped.",
        hints=[
            "Track all three names line by line: x, y, and temp.",
            f"temp holds the original x ({a}) so it survives the overwrite on the next line.",
        ],
        wrong_answer=f"{a} {b}",
    )


@archetype("variables", "augmented_ops")
def augmented_ops(rng: random.Random, seed: int):
    start = rng.randint(4, 12)
    add = rng.randint(2, 9)
    sub = rng.randint(1, 5)
    mult = rng.randint(2, 3)
    expected = (start + add - sub) * mult
    return make_exercise(
        "variables",
        seed,
        "short",
        "Shorthand operators",
        "What value is printed?",
        "score += n means score = score + n. The shorthand forms update a variable in place.",
        code=f"score = {start}\nscore += {add}\nscore -= {sub}\nscore *= {mult}\nprint(score)",
        expected=expected,
        solution=str(expected),
        solution_note=f"Step by step: {start} + {add} = {start + add}, minus {sub} = {start + add - sub}, times {mult} = {expected}.",
        hints=[
            "+= adds to the current value, -= subtracts, *= multiplies.",
            f"Start at {start} and apply each line: + {add}, - {sub}, * {mult}.",
        ],
        wrong_answer=str(start + add - sub),
    )


@archetype("variables", "copy_semantics_choice")
def copy_semantics_choice(rng: random.Random, seed: int):
    first, second = rng.sample(range(2, 20), 2)
    expected = str(first)
    return make_exercise(
        "variables",
        seed,
        "choice",
        "Does y follow x?",
        "What is printed?",
        "Assignment copies the current value. Rebinding x later does not touch y.",
        code=f"x = {first}\ny = x\nx = {second}\nprint(y)",
        choices=[expected, str(second), "None", "NameError"],
        expected=expected,
        solution=expected,
        solution_note=f"y = x stored the value {first} under the name y. Reassigning x to {second} afterward changes only x.",
        hints=[
            "y = x happens at a moment in time; it copies the value x has right then.",
            "Later assignments to x do not reach back and update y.",
        ],
        wrong_answer=str(second),
    )


@archetype("variables", "build_total")
def build_total(rng: random.Random, seed: int):
    price = rng.randint(3, 25)
    count = rng.randint(2, 9)
    return make_exercise(
        "variables",
        seed,
        "code",
        "Combine two variables",
        "Using the price and count variables (not the raw numbers), create a variable named total holding price times count.",
        "Variables let you write rules once and reuse them with any values.",
        starter=f"price = {price}\ncount = {count}\n# create total below\n",
        tests=[CodeTest("total", price * count), CodeTest("price", price), CodeTest("count", count)],
        solution=f"price = {price}\ncount = {count}\ntotal = price * count\n",
        solution_note="total = price * count expresses the rule with names, so it stays correct even if the numbers change.",
        hints=[
            "You need one new line that defines total.",
            "Multiply the two existing names with *.",
            "total = price * count",
        ],
        wrong_answer=f"price = {price}\ncount = {count}\ntotal = 0\n",
    )


# --------------------------------------------------------------------------
# types_conversion
# --------------------------------------------------------------------------


@archetype("types_conversion", "type_of_choice")
def type_of_choice(rng: random.Random, seed: int):
    pool = [
        ("3.0", "float"),
        ('"3.0"', "str"),
        ("7", "int"),
        ('"7"', "str"),
        ("True", "bool"),
        ("2.5", "float"),
        ("False", "bool"),
        ('"hello"', "str"),
        ("0", "int"),
    ]
    literal, type_name = rng.choice(pool)
    expected = f"<class '{type_name}'>"
    choices = ["<class 'int'>", "<class 'float'>", "<class 'str'>", "<class 'bool'>"]
    wrong = rng.choice([c for c in choices if c != expected])
    return make_exercise(
        "types_conversion",
        seed,
        "choice",
        "Name that type",
        "What is printed?",
        "Every value has a type. Quotes mean str, a decimal point means float, True/False are bool.",
        code=f"print(type({literal}))",
        choices=choices,
        expected=expected,
        solution=expected,
        solution_note=f"{literal} is a {type_name}: quotes make text, a decimal point makes a float, and bare whole numbers are ints.",
        hints=[
            "Check for quotes first: anything quoted is a string no matter what is inside.",
            "No quotes? A decimal point means float, True/False mean bool, otherwise int.",
        ],
        wrong_answer=wrong,
    )


@archetype("types_conversion", "int_cast_predict")
def int_cast_predict(rng: random.Random, seed: int):
    text_num = rng.randint(3, 40)
    add = rng.randint(2, 15)
    expected = text_num + add
    return make_exercise(
        "types_conversion",
        seed,
        "short",
        "Convert then add",
        "What number is printed?",
        "int(text) converts digit characters into a real number you can do math with.",
        code=f'print(int("{text_num}") + {add})',
        expected=expected,
        solution=str(expected),
        solution_note=f'int("{text_num}") produces the number {text_num}, so the sum is normal math: {expected}.',
        hints=[
            "int(...) turns the quoted digits into an actual number before the addition runs.",
            f"After conversion the line is just {text_num} + {add}.",
        ],
        wrong_answer=f"{text_num}{add}",
    )


@archetype("types_conversion", "truncate_predict")
def truncate_predict(rng: random.Random, seed: int):
    whole = rng.randint(2, 30)
    frac = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    x = whole + frac / 10
    expected = whole
    return make_exercise(
        "types_conversion",
        seed,
        "short",
        "int() is not rounding",
        "What number is printed?",
        "int(x) on a float chops off the decimal part entirely; it does not round.",
        code=f"print(int({x}))",
        expected=expected,
        solution=str(expected),
        solution_note=f"int() truncates toward zero, so int({x}) drops the .{frac} completely and gives {whole}, even though round({x}) would give {round(x)}.",
        hints=[
            "int() and round() are different operations.",
            "int() simply removes everything after the decimal point.",
        ],
        wrong_answer=str(round(x)),
    )


@archetype("types_conversion", "bool_cast_choice")
def bool_cast_choice(rng: random.Random, seed: int):
    pool = [
        ("bool(0)", "False"),
        ("bool(1)", "True"),
        ("bool(-3)", "True"),
        ('bool("")', "False"),
        ('bool("0")', "True"),
        ('bool(" ")', "True"),
        ("bool(0.0)", "False"),
        ('bool("False")', "True"),
    ]
    expr, expected = rng.choice(pool)
    return make_exercise(
        "types_conversion",
        seed,
        "choice",
        "Truthy or falsy?",
        "What is printed?",
        'Only "empty" values are falsy: 0, 0.0, and "". Any non-empty string is truthy, even "0".',
        code=f"print({expr})",
        choices=["True", "False"],
        expected=expected,
        solution=expected,
        solution_note=f"{expr} is {expected}: numbers are falsy only at exactly zero, and strings are falsy only when completely empty.",
        hints=[
            "Ask: is the value zero or completely empty?",
            'A string with anything inside it, even "0" or a space, counts as True.',
        ],
        wrong_answer="True" if expected == "False" else "False",
    )


@archetype("types_conversion", "str_build_message")
def str_build_message(rng: random.Random, seed: int):
    n = rng.randint(3, 99)
    label = rng.choice(["Score", "Level", "Round", "Points"])
    expected = f"{label}: {n}"
    return make_exercise(
        "types_conversion",
        seed,
        "code",
        "Join text and a number",
        f'Using the n variable, create a variable named message holding exactly "{label}: {n}". You cannot glue text to a number directly, so convert with str(n) or use an f-string.',
        "Mixing str and int with + causes a TypeError; convert first, or let an f-string do it.",
        starter=f"n = {n}\n# create message below\n",
        tests=[CodeTest("message", expected), CodeTest("n", n)],
        solution=f'n = {n}\nmessage = f"{label}: {{n}}"\n',
        solution_note=f'An f-string converts n automatically: f"{label}: {{n}}". The str() version works too: "{label}: " + str(n).',
        hints=[
            f'"{label}: " + n crashes because Python refuses to add text and a number.',
            "Either wrap the number: str(n), or use an f-string with {n} inside.",
            f'message = f"{label}: {{n}}"',
        ],
        wrong_answer=f'n = {n}\nmessage = "{label}: n"\n',
    )


# --------------------------------------------------------------------------
# strings
# --------------------------------------------------------------------------


@archetype("strings", "slice_upper")
def slice_upper(rng: random.Random, seed: int):
    word = rng.choice(WORDS)
    start = rng.randint(0, len(word) - 3)
    stop = rng.randint(start + 2, len(word))
    expected = word[start:stop].upper()
    return make_exercise(
        "strings",
        seed,
        "short",
        "Slice and uppercase",
        "What exact text is printed?",
        "Slices include the start index and stop before the end index.",
        code=f'word = "{word}"\nprint(word[{start}:{stop}].upper())',
        expected=expected,
        solution=expected,
        solution_note=f'word[{start}:{stop}] takes characters {start} up to (not including) {stop}: "{word[start:stop]}". .upper() capitalizes it.',
        hints=[
            "Count characters starting from 0; the stop index itself is not included.",
            f'Characters {start} through {stop - 1} of "{word}" spell "{word[start:stop]}".',
        ],
        wrong_answer=word[start:stop],
    )


@archetype("strings", "index_char")
def index_char(rng: random.Random, seed: int):
    word = rng.choice(WORDS)
    if rng.random() < 0.5:
        index = rng.randint(0, len(word) - 1)
    else:
        index = -rng.randint(1, len(word))
    expected = word[index]
    return make_exercise(
        "strings",
        seed,
        "short",
        "Pick one character",
        "What single character is printed?",
        "Indexing starts at 0 from the front. Negative indexes count from the end: -1 is the last character.",
        code=f'word = "{word}"\nprint(word[{index}])',
        expected=expected,
        solution=expected,
        solution_note=(
            f"Index {index} counts from the end: -1 is the last character, so word[{index}] is '{expected}'."
            if index < 0
            else f"Counting from 0, position {index} of \"{word}\" is '{expected}'."
        ),
        hints=[
            "The first character is index 0, not 1.",
            "A negative index counts backward from the end: -1 is the last character.",
        ],
        wrong_answer=word[(index + 1) % len(word)],
    )


@archetype("strings", "fstring_build")
def fstring_build(rng: random.Random, seed: int):
    name = rng.choice(NAMES)
    age = rng.randint(18, 70)
    expected = f"{name} is {age}"
    return make_exercise(
        "strings",
        seed,
        "code",
        "Build text with an f-string",
        f'Using the name and age variables, create a variable named message holding exactly "{expected}". Use an f-string.',
        'An f-string is quoted text with a leading f; anything inside {braces} is filled in from variables: f"{name} is {age}".',
        starter=f'name = "{name}"\nage = {age}\n# create message below\n',
        tests=[CodeTest("message", expected), CodeTest("name", name), CodeTest("age", age)],
        solution=f'name = "{name}"\nage = {age}\nmessage = f"{{name}} is {{age}}"\n',
        solution_note='The f before the opening quote activates the braces: each {variable} is replaced with its current value.',
        hints=[
            "Start the string with the letter f before the opening quote.",
            "Inside the quotes, wrap each variable name in curly braces.",
            'message = f"{name} is {age}"',
        ],
        wrong_answer=f'name = "{name}"\nage = {age}\nmessage = "name is age"\n',
    )


@archetype("strings", "len_predict")
def len_predict(rng: random.Random, seed: int):
    first, second = rng.sample(SIMPLE_WORDS, 2)
    phrase = f"{first} {second}"
    expected = len(phrase)
    return make_exercise(
        "strings",
        seed,
        "short",
        "Measure the string",
        "What number is printed?",
        "len(text) counts every character, including spaces.",
        code=f'phrase = "{phrase}"\nprint(len(phrase))',
        expected=expected,
        solution=str(expected),
        solution_note=f'"{first}" has {len(first)} characters, "{second}" has {len(second)}, plus 1 for the space: {expected}.',
        hints=[
            "Count every character between the quotes, one by one.",
            "Do not forget the space in the middle; it counts too.",
        ],
        wrong_answer=str(expected - 1),
    )


@archetype("strings", "concat_repeat")
def concat_repeat(rng: random.Random, seed: int):
    if rng.random() < 0.5:
        word = rng.choice(["ha", "no", "go", "hey", "yo"])
        n = rng.randint(2, 4)
        code = f'print("{word}" * {n})'
        expected = word * n
        note = f'Multiplying a string repeats it: "{word}" * {n} chains {n} copies together with nothing between them.'
        wrong = " ".join([word] * n)
    else:
        first, second = rng.sample(SIMPLE_WORDS, 2)
        code = f'print("{first}" + "-" + "{second}")'
        expected = f"{first}-{second}"
        note = "The + operator glues strings end to end, exactly as written, adding nothing extra."
        wrong = f"{first} - {second}"
    return make_exercise(
        "strings",
        seed,
        "short",
        "String math",
        "What exact text is printed?",
        "+ joins strings together; * repeats a string a whole number of times.",
        code=code,
        expected=expected,
        solution=expected,
        solution_note=note,
        hints=[
            "+ and * work on strings too: joining and repeating.",
            "Neither operator adds spaces on its own; write out the result character by character.",
        ],
        wrong_answer=wrong,
    )
