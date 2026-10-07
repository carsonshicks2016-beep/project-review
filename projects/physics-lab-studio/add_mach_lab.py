import os
import re

filepath = "/Users/REVIEW_USER/Downloads/untitled folder 2/index.html"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Add the new lab to the array
new_lab = """      },
      {
        num: 43,
        id: "mach_lab",
        title: "Mach Lab: Sound in Motion",
        file: "mach-lab/index.html",
        cat: "waves",
        catName: "Sound & Waves",
        badge: "Mach 0.3–8.0",
        formula: "t = τ + √((vτ)² + h²) / c",
        threshold: "Mach 1.0 sound barrier",
        desc: "A highly-polished physics-based auralization of a supersonic jet flyby. Visualizes the Mach cone and synthesizes the exact N-wave shock profile on the ground."
      }"""

content = content.replace("      }\n    ];", new_lab + "\n    ];")

# Update counts
content = content.replace("Labs 00–42", "Labs 00–43")
content = content.replace("Search 43 labs", "Search 44 labs")
content = content.replace("All Disciplines (43 Labs)", "All Disciplines (44 Labs)")
content = content.replace("43 / 43", "44 / 44")
content = content.replace("43 standalone interactive", "44 standalone interactive")
content = content.replace('text-sm">43</span>', 'text-sm">44</span>')
content = content.replace("All (43)", "All (44)")
content = content.replace("Registry of all 43 Programs", "Registry of all 44 Programs")

# Update Sound & Waves count from 6 to 7
content = content.replace("Sound & Waves (6)", "Sound & Waves (7)")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Added mach-lab to index.html successfully.")
