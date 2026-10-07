import sys
import time
from backend.indexer import get_indexer

def test_fast():
    t0 = time.time()
    indexer = get_indexer()
    index = indexer.index
    print(f"Loaded index in {time.time()-t0:.2f}s")
    
    t1 = time.time()
    titles = sorted([t for t in index.all_titles if len(t) > 4], key=len, reverse=True)
    note_items = sorted(list(index.notes.items()), key=lambda x: x[0])
    
    found_count = 0
    all_patches = []
    skip = 0
    limit = 50
    import re
    
    for note_title, note in note_items:
        content = note.content
        content_lower = content.lower()
        possible_titles = [t for t in titles if t.lower() in content_lower and t not in note.outgoing_links and t != note_title]
        
        for t in possible_titles:
            pattern = re.compile(r'(?<!\[\[)' + re.escape(t) + r'(?!\]\])', re.IGNORECASE)
            lines = content.split("\n")
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith(">") or stripped.startswith("#") or stripped == "---":
                    continue
                match = pattern.search(line)
                if match:
                    before_match = line[:match.start()]
                    if before_match.count("[[") > before_match.count("]]"):
                        continue
                    
                    found_count += 1
                    if found_count > skip and len(all_patches) < limit:
                        original = match.group(0)
                        rep = f"[[{t}]]" if original == t else f"[[{t}|{original}]]"
                        all_patches.append({
                            "target_note_title": note_title,
                            "target_note_path": note.filepath,
                            "proposed_replacement": rep,
                            "context_line": line.strip(),
                            "line_number": line_num
                        })
                    
                    if len(all_patches) >= limit:
                        break
            if len(all_patches) >= limit:
                break
        if len(all_patches) >= limit:
            break

    print(f"Fast scan found {len(all_patches)} patches in {time.time()-t1:.2f}s")

if __name__ == "__main__":
    test_fast()
