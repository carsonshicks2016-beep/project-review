"""Stitches all 5 curriculum level videos into a single master showcase MP4 with title cards and overlay banners.
"""
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ART_DIR = Path('/Users/REVIEW_USER/.gemini/antigravity/brain/fda2524c-2d80-46b0-b019-a21ff7938032')
OUT_PATH = ART_DIR / 'stage4_curriculum_showcase.mp4'

TIER_INFO = {
    1: {
        "title": "LEVEL 1: NOVICE",
        "subtitle": "Gaps: 0.20-0.35m | Hurdles: 0.10-0.18m | 75% Clearance, 60% Full Clears",
        "color": (46, 204, 113), # Green
    },
    2: {
        "title": "LEVEL 2: INTERMEDIATE",
        "subtitle": "Gaps: 0.34-0.49m | Hurdles: 0.18-0.26m | Track: 3.1m",
        "color": (52, 152, 219), # Blue
    },
    3: {
        "title": "LEVEL 3: ADVANCED",
        "subtitle": "Gaps: 0.48-0.62m | Hurdles: 0.26-0.34m | High Hurdle Flight",
        "color": (155, 89, 182), # Purple
    },
    4: {
        "title": "LEVEL 4: EXPERT",
        "subtitle": "Gaps: 0.61-0.76m | Hurdles: 0.34-0.42m | Track: 2.5m",
        "color": (230, 126, 34), # Orange
    },
    5: {
        "title": "LEVEL 5: EXTREME PARKOUR",
        "subtitle": "Gaps: 0.75-0.90m | Hurdles: 0.42-0.50m | Track: 2.2m",
        "color": (231, 76, 60), # Red
    },
}

def create_title_card(level: int, fps: int = 67, duration_sec: float = 1.2) -> list:
    info = TIER_INFO[level]
    w, h = 640, 480
    img = Image.new('RGB', (w, h), (18, 22, 28))
    draw = ImageDraw.Draw(img)
    
    # Border
    draw.rectangle([(20, 20), (w - 20, h - 20)], outline=info["color"], width=3)
    
    # Text
    draw.text((w // 2 - 130, h // 2 - 60), "STAGE 4 CURRICULUM", fill=(180, 190, 200))
    draw.text((w // 2 - 110, h // 2 - 20), info["title"], fill=info["color"])
    draw.text((w // 2 - 210, h // 2 + 30), info["subtitle"], fill=(220, 225, 230))
    draw.text((w // 2 - 140, h // 2 + 70), "Heading Alignment: f_x = 0.96 (Forward)", fill=(150, 160, 170))
    
    card_arr = np.array(img)
    n_frames = int(round(fps * duration_sec))
    return [card_arr] * n_frames

def overlay_banner(frame_arr: np.ndarray, level: int) -> np.ndarray:
    info = TIER_INFO[level]
    img = Image.fromarray(frame_arr)
    w, h = img.size
    
    # Semi-transparent dark banner at top
    overlay = Image.new('RGBA', (w, 42), (10, 14, 20, 210))
    draw = ImageDraw.Draw(overlay)
    
    # Level indicator box
    draw.rectangle([(8, 6), (150, 36)], fill=info["color"])
    draw.text((16, 12), info["title"], fill=(255, 255, 255))
    
    # Subtitle details
    draw.text((165, 12), info["subtitle"], fill=(235, 240, 245))
    
    # Heading tag
    draw.text((w - 100, 12), "f_x = 0.96", fill=(46, 204, 113))
    
    img.paste(overlay, (0, 0), overlay)
    return np.array(img)

def main():
    fps = 67
    writer = imageio.get_writer(str(OUT_PATH), fps=fps, codec='libx264', pixelformat='yuv420p')
    
    print(f"Creating combined showcase video: {OUT_PATH}")
    for lvl in range(1, 6):
        video_file = ART_DIR / f'level_{lvl}_eval.mp4'
        if not video_file.exists():
            print(f"Warning: {video_file} not found, skipping.")
            continue
        
        print(f"Processing Level {lvl}...")
        # 1. Title card
        title_frames = create_title_card(lvl, fps=fps, duration_sec=1.2)
        for tf in title_frames:
            writer.append_data(tf)
            
        # 2. Level frames with banner overlay
        reader = imageio.get_reader(video_file)
        n_frames = reader.count_frames()
        for idx in range(n_frames):
            frame = reader.get_data(idx)
            overlaid = overlay_banner(frame, lvl)
            writer.append_data(overlaid)
        reader.close()
        
    writer.close()
    print(f"Successfully generated showcase video at: {OUT_PATH}")

if __name__ == '__main__':
    main()
