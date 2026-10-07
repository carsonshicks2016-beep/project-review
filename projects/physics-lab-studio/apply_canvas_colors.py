import os
import re
import glob

workspace = "/Users/REVIEW_USER/Downloads/untitled folder 2"

# Map of old hex codes to new hex codes based on the new palette
color_map = {
    # Backgrounds
    '#020617': '#0b1217', # slate-950
    '#0f172a': '#111a22', # slate-900
    '#1e293b': '#1d2b38', # slate-800
    
    # Sky / Cyan
    '#38bdf8': '#00f0ff', # sky-400
    '#0ea5e9': '#00d4e0', # sky-500
    
    # Emerald / Lime
    '#34d399': '#ccff00', # emerald-400
    '#10b981': '#b3e600', # emerald-500
    
    # Rose / Neon Pink
    '#fb7185': '#ff0055', # rose-400
    '#f43f5e': '#e6004c', # rose-500
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Case-insensitive replacement for hex codes
    for old_hex, new_hex in color_map.items():
        # Replace lowercase and uppercase hex
        content = re.sub(old_hex, new_hex, content, flags=re.IGNORECASE)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated JS colors in {os.path.basename(filepath)}")

html_files = glob.glob(os.path.join(workspace, "*.html"))
for f in html_files:
    process_file(f)

print("Canvas color overhaul complete.")
