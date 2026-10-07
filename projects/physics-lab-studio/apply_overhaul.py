import os
import re
import glob

workspace = "/Users/REVIEW_USER/Downloads/untitled folder 2"

font_links = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Space+Grotesk:wght@300..700&display=swap" rel="stylesheet">
"""

tailwind_config = """
<script>
  tailwind.config = {
    darkMode: 'class',
    theme: {
      extend: {
        fontFamily: {
          sans: ['"DM Sans"', 'sans-serif'],
          display: ['"Space Grotesk"', 'sans-serif'],
          mono: ['"Space Grotesk"', 'monospace'],
        },
        colors: {
          slate: {
            50: '#f8fafc',
            100: '#f1f5f9',
            200: '#e2e8f0',
            300: '#cbd5e1',
            400: '#8ba3b8',
            500: '#64748b',
            600: '#475569',
            700: '#334155',
            800: '#1d2b38',
            900: '#111a22',
            950: '#0b1217',
          },
          sky: {
            400: '#00f0ff',
            500: '#00d4e0',
          },
          emerald: {
            400: '#ccff00',
            500: '#b3e600',
          },
          rose: {
            400: '#ff0055',
            500: '#e6004c',
          },
          amber: {
            400: '#ffb300',
            500: '#e6a100',
          }
        },
        boxShadow: {
          'neon': '0 0 10px rgba(0, 240, 255, 0.2), 0 0 20px rgba(0, 240, 255, 0.1)',
          'neon-lime': '0 0 10px rgba(204, 255, 0, 0.2), 0 0 20px rgba(204, 255, 0, 0.1)',
        }
      }
    }
  }
</script>
"""

custom_styles = """
<style>
  body {
    background-color: #0b1217 !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
  }
  .font-mono, .data-readout {
    font-family: 'Space Grotesk', monospace !important;
    font-variant-numeric: tabular-nums;
  }
  /* Sharper aesthetics */
  .rounded-2xl { border-radius: 0.375rem !important; }
  .rounded-xl { border-radius: 0.375rem !important; }
  .rounded-lg { border-radius: 0.25rem !important; }
  .shadow-xl, .shadow-2xl { box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important; }
  
  /* Technical UI elements */
  button { text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.85em; font-weight: 600; }
  input[type=range]::-webkit-slider-thumb { border-radius: 0 !important; width: 8px !important; height: 16px !important; background: #00f0ff !important; }
</style>
"""

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already processed to avoid duplicate injections
    if "DM+Sans" in content and "tailwind.config" in content:
        return

    # 1. Inject fonts and tailwind config right after tailwindcss script
    tailwind_script = '<script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>'
    
    if tailwind_script in content:
        injection = tailwind_script + "\n" + font_links + "\n" + tailwind_config + "\n" + custom_styles
        content = content.replace(tailwind_script, injection)
    elif "</head>" in content:
        # Fallback if tailwind script not found
        content = content.replace("</head>", font_links + "\n" + tailwind_config + "\n" + custom_styles + "\n</head>")

    # 2. Refine classes for a more "technical" look (border thickness, bg opacity)
    # Decrease blur intensity, make backgrounds slightly more opaque
    content = content.replace("backdrop-blur-xl", "backdrop-blur-md")
    content = content.replace("bg-slate-900/50", "bg-slate-900/80")
    content = content.replace("bg-slate-950/50", "bg-slate-950/80")
    
    # Optional: replace the generic body background color if it was hardcoded in style tags
    content = re.sub(r'background-color:\s*#020617;', 'background-color: #0b1217;', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Processed {os.path.basename(filepath)}")

html_files = glob.glob(os.path.join(workspace, "*.html"))
for f in html_files:
    process_file(f)

print("Overhaul complete.")
