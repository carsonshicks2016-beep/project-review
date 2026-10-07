#!/usr/bin/env python3
"""
Offline Lecture Audio Processor & Word Frequency Generator
Companion script for LectureCanvas.

Features:
- Transcribes audio files (WAV, MP3, WebM, M4A) locally using OpenAI Whisper or faster-whisper.
- Filters out stopwords and spoken filler phrases.
- Computes word frequencies and outputs JSON formatted for LectureCanvas.
"""

import sys
import os
import json
import re
from collections import Counter
from typing import List, Tuple

DEFAULT_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most',
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such',
    'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were',
    'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'also', 'will', 'say', 'said', 'get', 'go', 'going', 'went',
    'make', 'made', 'take', 'see', 'one', 'two', 'thing', 'things', 'way', 'much', 'many', 'well'
}

LECTURE_FILLERS = {
    'um', 'uh', 'like', 'actually', 'basically', 'literally', 'sort', 'kind', 'right',
    'yeah', 'okay', 'ok', 'alright', 'know', 'mean', 'guess', 'gonna', 'wanna', 'gotta'
}

# Words ending in "s" that are not plurals. Kept in sync with the same list in
# src/services/textProcessor.ts so both paths rank a lecture identically.
NON_PLURAL_S = {
    'series', 'species', 'physics', 'mathematics', 'maths', 'statistics', 'economics',
    'ethics', 'politics', 'mechanics', 'dynamics', 'thermodynamics', 'genetics', 'optics',
    'acoustics', 'linguistics', 'semantics', 'robotics', 'graphics', 'logistics',
    'aesthetics', 'heuristics', 'bias', 'atlas', 'canvas', 'gas', 'lens', 'news', 'means',
    'plus', 'minus', 'always', 'perhaps', 'towards', 'unless', 'yes',
}

# Plurals whose singular ends in "-ie", not "-y".
IE_PLURALS = {'movies', 'calories', 'cookies', 'series', 'genies', 'zombies', 'rookies'}


def clean_word(word: str) -> str:
    """
    Lowercases and conservatively de-pluralizes a token.

    The "-es" rule only fires after a sibilant (boxes, matches, processes). Applying
    it to every "-es" word mangles the vocabulary a lecture actually cares about:
    "series" becomes "seri" and "species" becomes "speci".
    """
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', word.lower())

    if len(cleaned) <= 3 or cleaned in NON_PLURAL_S:
        return cleaned

    if cleaned.endswith('ies') and len(cleaned) > 4 and not re.search(r'[aeiou]ies$', cleaned):
        return cleaned[:-1] if cleaned in IE_PLURALS else cleaned[:-3] + 'y'
    if re.search(r'(?:ss|sh|ch|x|z)es$', cleaned):
        return cleaned[:-2]
    if cleaned.endswith('s') and not re.search(r'(?:ss|us|is)$', cleaned):
        return cleaned[:-1]
    return cleaned

def process_text(text: str, min_length: int = 3) -> List[Tuple[str, int]]:
    raw_tokens = re.findall(r"\b[a-zA-Z0-9'-]+\b", text)
    filtered = []
    display_map = {}

    for tok in raw_tokens:
        norm = clean_word(tok)
        if len(norm) < min_length:
            continue
        if norm in DEFAULT_STOPWORDS or norm in LECTURE_FILLERS:
            continue
        if norm.isdigit():
            continue

        display = tok.capitalize() if len(tok) >= 3 else tok.upper()
        if norm not in display_map:
            display_map[norm] = display
        filtered.append(norm)

    counts = Counter(filtered)
    results = [(display_map[word], count) for word, count in counts.most_common(50)]
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 offline_processor.py <path-to-audio-or-txt-file>")
        print("\nNote: To transcribe audio files directly, install whisper: pip install openai-whisper")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        sys.exit(1)

    transcript_text = ""

    # If it's a text file
    if input_path.endswith('.txt'):
        with open(input_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
    else:
        # Audio file: attempt whisper import
        try:
            import whisper
            print('Loading Whisper model (base)...')
            model = whisper.load_model("base")
            print(f"Transcribing {input_path}...")
            result = model.transcribe(input_path)
            transcript_text = result["text"]
            
            # Save raw transcript
            output_txt = input_path + ".transcript.txt"
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            print(f"Transcript saved to {output_txt}")
        except ImportError:
            print("whisper is not installed. Run: pip install openai-whisper to process audio files offline.")
            sys.exit(1)

    print("\nProcessing word frequencies...")
    ranked_words = process_text(transcript_text)

    print("\n--- TOP REPEATED KEYWORDS ---")
    for rank, (word, count) in enumerate(ranked_words[:20], 1):
        bar = "█" * min(count, 30)
        print(f"{rank:2d}. {word:15s} ({count:3d}x) {bar}")

    # Output JSON
    output_json = input_path + ".keywords.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump([{"text": w, "count": c} for w, c in ranked_words], f, indent=2)
    print(f"\nKeyword data saved to {output_json}")

if __name__ == '__main__':
    main()
