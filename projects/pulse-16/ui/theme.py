"""
Visual Design System & Styling Constants for PULSE-16.
High-contrast dark hardware console design with neon instrument accents.
"""

# Base Chassis Palette
BG_MAIN = (14, 16, 22)          # Deep chassis graphite
PANEL_BG = (22, 26, 34)         # Matte dark metal surface
PANEL_BORDER = (40, 46, 60)     # Outline bevel
PANEL_HEADER = (30, 36, 46)     # Accent header bar

TEXT_PRIMARY = (245, 248, 255)  # Crisp white
TEXT_SECONDARY = (150, 162, 182)# Subdued cool gray
TEXT_MUTED = (90, 102, 122)     # Tooltip & badge text

# Step Button Surfaces
STEP_INACTIVE_A = (32, 38, 50)  # Beat 1 & 3 steps (0-3, 8-11)
STEP_INACTIVE_B = (24, 28, 38)  # Beat 2 & 4 steps (4-7, 12-15)
STEP_BORDER = (48, 56, 72)
STEP_PLAYHEAD_BG = (65, 80, 105)# Backlight under active playhead

# All 26 Instrument Accent Colors
TRACK_COLORS = {
    # 1. Core Drums
    "KICK": (255, 65, 65),       # Crimson Red
    "SNARE": (255, 160, 20),     # Amber Gold
    "CLAP": (255, 215, 0),       # Pure Gold
    "RIMSHOT": (255, 105, 180),  # Hot Pink
    "HIHAT_CL": (0, 230, 255),   # Electric Cyan
    "HIHAT_OP": (0, 180, 240),   # Sky Blue
    "SHAKER": (70, 240, 190),    # Mint Turquoise
    "CRASH": (210, 235, 255),    # Ice White
    "CONGA": (255, 125, 45),     # Tangerine
    "TOM": (255, 0, 128),        # Vivid Magenta
    "COWBELL": (195, 115, 255),  # Electric Violet
    # 2. Orchestral Section
    "TIMPANI": (235, 45, 85),     # Regal Burgundy
    "ORCH_HIT": (255, 40, 180),   # Imperial Neon Magenta
    "BRASS_STAB": (255, 200, 30), # Horn Brass Gold
    "PIZZ_C": (255, 175, 55),     # Amber Violin Pluck
    "PIZZ_EB": (255, 150, 45),    # Warm Cello Pluck
    "PIZZ_G": (255, 120, 35),     # Deep Viola Pluck
    "TUBULAR_BELL": (130, 220, 255), # Shimmering Chimes
    # 3. Synth & Bass
    "BASS_C": (0, 245, 212),     # Bright Turquoise
    "BASS_EB": (40, 215, 80),    # Emerald
    "BASS_F": (120, 235, 20),    # Lime
    "BASS_G": (190, 245, 40),    # Chartreuse
    "LEAD_C3": (255, 225, 40),   # Solar Yellow
    "LEAD_EB3": (255, 155, 40),  # Electric Orange
    "LEAD_G3": (255, 75, 160),   # Neon Pink
    "LEAD_BB3": (185, 145, 255), # Bright Lavender
}

# Transport & Status Accents
COLOR_PLAY = (0, 235, 155)
COLOR_PAUSE = (255, 185, 20)
COLOR_RECORD = (255, 60, 60)
COLOR_SOLO = (255, 195, 30)
COLOR_MUTE = (245, 65, 65)
COLOR_BANK_ACTIVE = (0, 235, 190)

# Window & Geometry Dimensions
WINDOW_WIDTH = 1260
WINDOW_HEIGHT = 890

ROW_HEIGHT = 44
PAD_WIDTH = 124
MUTE_WIDTH = 25
SOLO_WIDTH = 25
STEP_WIDTH = 48
STEP_HEIGHT = 37
STEP_GAP = 5
GROUP_GAP = 12
