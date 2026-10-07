"""
Comprehensive Preset Patterns, Track Categories, and Presets for PULSE-16.
Features 26 instruments across Drums, Latin Percussion, Orchestral Section,
FM Bass, and Melodic Lead Synths.
"""

import random

# All 26 tracks in definitive order
TRACK_NAMES = [
    # 1. DRUMS (11)
    "KICK",
    "SNARE",
    "CLAP",
    "RIMSHOT",
    "HIHAT_CL",
    "HIHAT_OP",
    "SHAKER",
    "CRASH",
    "CONGA",
    "TOM",
    "COWBELL",
    # 2. ORCHESTRA (7)
    "TIMPANI",
    "ORCH_HIT",
    "BRASS_STAB",
    "PIZZ_C",
    "PIZZ_EB",
    "PIZZ_G",
    "TUBULAR_BELL",
    # 3. SYNTH & BASS (8)
    "BASS_C",
    "BASS_EB",
    "BASS_F",
    "BASS_G",
    "LEAD_C3",
    "LEAD_EB3",
    "LEAD_G3",
    "LEAD_BB3",
]

TRACK_CATEGORIES = {
    # Drums
    "KICK": "DRUMS",
    "SNARE": "DRUMS",
    "CLAP": "DRUMS",
    "RIMSHOT": "DRUMS",
    "HIHAT_CL": "DRUMS",
    "HIHAT_OP": "DRUMS",
    "SHAKER": "DRUMS",
    "CRASH": "DRUMS",
    "CONGA": "DRUMS",
    "TOM": "DRUMS",
    "COWBELL": "DRUMS",
    # Orchestra
    "TIMPANI": "ORCHESTRA",
    "ORCH_HIT": "ORCHESTRA",
    "BRASS_STAB": "ORCHESTRA",
    "PIZZ_C": "ORCHESTRA",
    "PIZZ_EB": "ORCHESTRA",
    "PIZZ_G": "ORCHESTRA",
    "TUBULAR_BELL": "ORCHESTRA",
    # Synth
    "BASS_C": "SYNTH",
    "BASS_EB": "SYNTH",
    "BASS_F": "SYNTH",
    "BASS_G": "SYNTH",
    "LEAD_C3": "SYNTH",
    "LEAD_EB3": "SYNTH",
    "LEAD_G3": "SYNTH",
    "LEAD_BB3": "SYNTH",
}

TRACK_LABELS = {
    "KICK": "808 Kick",
    "SNARE": "Snare",
    "CLAP": "Clap",
    "RIMSHOT": "Rimshot",
    "HIHAT_CL": "Closed Hat",
    "HIHAT_OP": "Open Hat",
    "SHAKER": "Shaker",
    "CRASH": "Crash",
    "CONGA": "Hi Conga",
    "TOM": "Mid Tom",
    "COWBELL": "Cowbell",
    # Orchestra
    "TIMPANI": "Timpani",
    "ORCH_HIT": "Orch Hit",
    "BRASS_STAB": "Brass Stab",
    "PIZZ_C": "Pizz (C4)",
    "PIZZ_EB": "Pizz (Eb4)",
    "PIZZ_G": "Pizz (G4)",
    "TUBULAR_BELL": "Tubular Bell",
    # Synth
    "BASS_C": "Bass (C2)",
    "BASS_EB": "Bass (Eb2)",
    "BASS_F": "Bass (F2)",
    "BASS_G": "Bass (G2)",
    "LEAD_C3": "Lead (C3)",
    "LEAD_EB3": "Lead (Eb3)",
    "LEAD_G3": "Lead (G3)",
    "LEAD_BB3": "Lead (Bb3)",
}

TRACK_KEYBINDS = {
    "KICK": "1",
    "SNARE": "2",
    "CLAP": "3",
    "RIMSHOT": "4",
    "HIHAT_CL": "5",
    "HIHAT_OP": "6",
    "SHAKER": "7",
    "CRASH": "8",
    "CONGA": "9",
    "TOM": "0",
    "COWBELL": "-",
    # Orchestra
    "TIMPANI": "Z",
    "ORCH_HIT": "X",
    "BRASS_STAB": "C",
    "PIZZ_C": "V",
    "PIZZ_EB": "B",
    "PIZZ_G": "N",
    "TUBULAR_BELL": "M",
    # Synth
    "BASS_C": "Q",
    "BASS_EB": "W",
    "BASS_F": "E",
    "BASS_G": "R",
    "LEAD_C3": "A",
    "LEAD_EB3": "S",
    "LEAD_G3": "D",
    "LEAD_BB3": "F",
}


def empty_grid() -> dict[str, list[bool]]:
    """Creates a blank 16-step grid for all 26 tracks."""
    return {track: [False] * 16 for track in TRACK_NAMES}


def get_preset_cinematic_epic() -> dict:
    """Hans Zimmer / Metro Boomin style epic orchestral hybrid beat."""
    grid = empty_grid()
    # Thunderous Timpani hits
    grid["TIMPANI"][0] = True
    grid["TIMPANI"][6] = True
    grid["TIMPANI"][10] = True
    # Orchestra Hits
    grid["ORCH_HIT"][0] = True
    grid["ORCH_HIT"][8] = True
    # Brass Stabs
    grid["BRASS_STAB"][4] = True
    grid["BRASS_STAB"][12] = True
    # Tubular Bells
    grid["TUBULAR_BELL"][14] = True
    # Pizzicato String arpeggios
    for s in [2, 6, 10, 14]:
        grid["PIZZ_C"][s] = True
    grid["PIZZ_EB"][4] = True
    grid["PIZZ_G"][8] = True
    grid["PIZZ_EB"][12] = True
    # 808 Trap Drums
    for s in [0, 7, 10, 14]:
        grid["KICK"][s] = True
    grid["SNARE"][4] = True
    grid["SNARE"][12] = True
    for s in range(16):
        grid["HIHAT_CL"][s] = True
    grid["HIHAT_OP"][2] = True
    grid["HIHAT_OP"][10] = True
    grid["CRASH"][0] = True
    # Sub Bass
    grid["BASS_C"][0] = True
    grid["BASS_C"][7] = True
    grid["BASS_EB"][10] = True
    grid["BASS_G"][14] = True
    return {"bpm": 136, "swing": 0.03, "grid": grid, "name": "Cinematic Epic"}


def get_preset_vivaldi_trap() -> dict:
    """Classical baroque strings meets modern hip-hop drill bounce."""
    grid = empty_grid()
    # Fast Pizzicato 16th pattern
    for s in [0, 2, 4, 6]:
        grid["PIZZ_C"][s] = True
    for s in [8, 10]:
        grid["PIZZ_EB"][s] = True
    for s in [12, 14]:
        grid["PIZZ_G"][s] = True
    # Brass stabs
    grid["BRASS_STAB"][4] = True
    grid["BRASS_STAB"][12] = True
    grid["TIMPANI"][0] = True
    grid["TIMPANI"][8] = True
    # Drums
    grid["KICK"][0] = True
    grid["KICK"][6] = True
    grid["KICK"][10] = True
    grid["RIMSHOT"][4] = True
    grid["SNARE"][12] = True
    for s in range(16):
        grid["HIHAT_CL"][s] = True
    grid["HIHAT_OP"][14] = True
    grid["BASS_C"][0] = True
    grid["BASS_EB"][6] = True
    grid["BASS_G"][10] = True
    return {"bpm": 140, "swing": 0.06, "grid": grid, "name": "Vivaldi Drill"}


def get_preset_trap() -> dict:
    grid = empty_grid()
    for s in [0, 7, 10, 14]:
        grid["KICK"][s] = True
    for s in [4, 12]:
        grid["SNARE"][s] = True
        grid["CLAP"][s] = True
    for s in range(16):
        grid["HIHAT_CL"][s] = True
    grid["HIHAT_OP"][2] = True
    grid["HIHAT_OP"][10] = True
    grid["RIMSHOT"][15] = True
    grid["CRASH"][0] = True
    grid["BASS_C"][0] = True
    grid["BASS_C"][7] = True
    grid["BASS_EB"][10] = True
    grid["BASS_G"][14] = True
    grid["LEAD_C3"][4] = True
    grid["LEAD_EB3"][6] = True
    grid["LEAD_G3"][12] = True
    grid["LEAD_BB3"][14] = True
    return {"bpm": 132, "swing": 0.04, "grid": grid, "name": "808 Trap Bounce"}


def get_preset_house() -> dict:
    grid = empty_grid()
    for s in [0, 4, 8, 12]:
        grid["KICK"][s] = True
    for s in [4, 12]:
        grid["CLAP"][s] = True
    for s in [2, 6, 10, 14]:
        grid["HIHAT_OP"][s] = True
    for s in range(16):
        grid["SHAKER"][s] = True
    grid["CRASH"][0] = True
    grid["CONGA"][7] = True
    grid["CONGA"][11] = True
    grid["COWBELL"][14] = True
    for s in [2, 6, 10, 14]:
        grid["BASS_C"][s] = True
    grid["BASS_EB"][7] = True
    grid["BASS_G"][15] = True
    grid["LEAD_C3"][2] = True
    grid["LEAD_EB3"][2] = True
    grid["LEAD_C3"][10] = True
    grid["LEAD_G3"][10] = True
    return {"bpm": 125, "swing": 0.12, "grid": grid, "name": "Classic House"}


def get_preset_synthwave() -> dict:
    grid = empty_grid()
    for s in [0, 4, 8, 12]:
        grid["KICK"][s] = True
    grid["SNARE"][4] = True
    grid["SNARE"][12] = True
    for s in range(16):
        grid["HIHAT_CL"][s] = True
    grid["HIHAT_OP"][2] = True
    grid["HIHAT_OP"][10] = True
    grid["CRASH"][0] = True
    grid["TOM"][14] = True
    grid["TOM"][15] = True
    for s in [0, 2, 4, 6]:
        grid["BASS_C"][s] = True
    for s in [8, 10]:
        grid["BASS_EB"][s] = True
    for s in [12, 14]:
        grid["BASS_G"][s] = True
    grid["LEAD_C3"][0] = True
    grid["LEAD_EB3"][4] = True
    grid["LEAD_G3"][8] = True
    grid["LEAD_BB3"][12] = True
    return {"bpm": 128, "swing": 0.0, "grid": grid, "name": "Synthwave 80s"}


def generate_algorithmic_groove() -> dict:
    grid = empty_grid()
    bpm = random.choice([88, 96, 110, 124, 132, 138])
    swing = round(random.uniform(0.0, 0.25), 2)
    
    grid["KICK"][0] = True
    if random.random() < 0.8:
        grid["KICK"][8] = True
    for s in [3, 6, 10, 14]:
        if random.random() < 0.4:
            grid["KICK"][s] = True
            
    snare_tr = random.choice(["SNARE", "CLAP", "RIMSHOT"])
    grid[snare_tr][4] = True
    grid[snare_tr][12] = True
    
    for s in range(16):
        if random.random() < 0.7:
            grid["HIHAT_CL"][s] = True
    for s in [2, 6, 10, 14]:
        if random.random() < 0.35:
            grid["HIHAT_OP"][s] = True
            
    # Include Orchestral accents randomly
    if random.random() < 0.5:
        grid["TIMPANI"][0] = True
    if random.random() < 0.4:
        grid["ORCH_HIT"][0] = True
    if random.random() < 0.5:
        grid["BRASS_STAB"][random.choice([4, 12])] = True
    if random.random() < 0.4:
        for s in [2, 6, 10, 14]:
            if random.random() < 0.5:
                grid[random.choice(["PIZZ_C", "PIZZ_EB", "PIZZ_G"])][s] = True
                
    # Bass
    bass_notes = ["BASS_C", "BASS_EB", "BASS_F", "BASS_G"]
    grid["BASS_C"][0] = True
    for step in [2, 4, 6, 8, 10, 12, 14]:
        if random.random() < 0.35:
            grid[random.choice(bass_notes)][step] = True
            
    return {"bpm": bpm, "swing": swing, "grid": grid, "name": "Algorithmic Hybrid"}


PRESETS = {
    "Cinematic Epic": get_preset_cinematic_epic,
    "Vivaldi Drill": get_preset_vivaldi_trap,
    "808 Trap": get_preset_trap,
    "Classic House": get_preset_house,
    "Synthwave": get_preset_synthwave,
}
