import os
import re
import glob

workspace = "/Users/REVIEW_USER/Downloads/untitled folder 2"

# Map of old RGBA prefixes to new RGBA prefixes
rgba_map = {
    # sky-400 / 500
    'rgba(56, 189, 248': 'rgba(0, 240, 255',
    'rgba(14, 165, 233': 'rgba(0, 212, 224',
    
    # slate-900
    'rgba(15, 23, 42': 'rgba(17, 26, 34',
    
    # slate-800
    'rgba(30, 41, 59': 'rgba(29, 43, 56',
    
    # slate-700
    'rgba(51, 65, 85': 'rgba(51, 65, 85', # keep as is
    
    # emerald-400 / 500
    'rgba(52, 211, 153': 'rgba(204, 255, 0',
    'rgba(16, 185, 129': 'rgba(179, 230, 0',
    
    # rose-400 / 500
    'rgba(251, 113, 133': 'rgba(255, 0, 85',
    'rgba(244, 63, 94': 'rgba(230, 0, 76',
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    for old_rgba, new_rgba in rgba_map.items():
        content = content.replace(old_rgba, new_rgba)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated RGBA in {os.path.basename(filepath)}")

html_files = glob.glob(os.path.join(workspace, "*.html"))
for f in html_files:
    process_file(f)

print("RGBA overhaul complete.")
