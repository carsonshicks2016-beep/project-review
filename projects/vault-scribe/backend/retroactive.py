"""
Vault Scribe — Retroactive Innervation Engine
When a new note is created, this module finds all existing notes that mention
the new note's title but don't link to it — and offers to inject [[wikilinks]]
into those existing notes. Your vault grows bidirectionally.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from backend.config import VAULT_PATH
from backend.indexer import get_indexer


@dataclass
class RetroactivePatch:
    """A proposed wikilink injection into an existing note."""
    target_note_title: str      # The existing note that will be patched
    target_note_path: str       # Relative path to the existing note
    match_text: str             # The text that matched the new note's title
    context_line: str           # The line containing the match
    line_number: int            # Line number in the existing note (1-indexed)
    proposed_replacement: str   # The line with [[wikilink]] inserted


@dataclass
class RetroactiveResult:
    """Result of retroactive innervation scan."""
    patches: list[RetroactivePatch]
    scanned_notes: int
    total_patches: int


def find_retroactive_links(
    new_note_title: str,
    aliases: list[str] | None = None,
) -> RetroactiveResult:
    """
    Scan the vault for existing notes that mention the new note's title
    but don't wikilink to it.

    Args:
        new_note_title: The title of the newly created note.
        aliases: Additional terms to search for (e.g., abbreviations).

    Returns:
        RetroactiveResult with proposed patches.
    """
    indexer = get_indexer()
    index = indexer.index

    search_terms = [new_note_title]
    if aliases:
        search_terms.extend(aliases)

    patches: list[RetroactivePatch] = []
    scanned = 0

    for note_title, note in index.notes.items():
        # Don't patch the note itself
        if note_title == new_note_title:
            continue

        scanned += 1

        # Check if this note already links to the new note
        if new_note_title in note.outgoing_links:
            continue

        # Search for each term in the note's content
        for term in search_terms:
            if len(term) < 3:
                continue

            lines = note.content.split("\n")
            for line_num, line in enumerate(lines, 1):
                # Skip metadata lines, headings, and existing wikilinks
                stripped = line.strip()
                if stripped.startswith(">") or stripped.startswith("#") or stripped == "---":
                    continue

                # Case-insensitive search for the term
                pattern = re.compile(
                    r'(?<!\[\[)' + re.escape(term) + r'(?!\]\])',
                    re.IGNORECASE,
                )
                match = pattern.search(line)
                if match:
                    # Check we're not inside an existing wikilink
                    before_match = line[:match.start()]
                    if before_match.count("[[") > before_match.count("]]"):
                        continue

                    # Build the replacement line
                    original = match.group(0)
                    if original == new_note_title:
                        replacement_text = f"[[{new_note_title}]]"
                    else:
                        replacement_text = f"[[{new_note_title}|{original}]]"

                    proposed = line[:match.start()] + replacement_text + line[match.end():]

                    patches.append(RetroactivePatch(
                        target_note_title=note_title,
                        target_note_path=note.relative_path,
                        match_text=original,
                        context_line=line.strip(),
                        line_number=line_num,
                        proposed_replacement=proposed.strip(),
                    ))

                    # Only find the first match per note per term
                    break

    # Deduplicate patches (one per note)
    seen_notes: set[str] = set()
    unique_patches: list[RetroactivePatch] = []
    for patch in patches:
        if patch.target_note_title not in seen_notes:
            unique_patches.append(patch)
            seen_notes.add(patch.target_note_title)

    return RetroactiveResult(
        patches=unique_patches,
        scanned_notes=scanned,
        total_patches=len(unique_patches),
    )


def apply_retroactive_patches(
    patches: list[dict],
    vault_path: Path | None = None,
) -> dict[str, str]:
    """
    Apply retroactive patches to existing vault notes.

    Args:
        patches: List of patch dicts with keys:
            - target_note_path: relative path to the note
            - line_number: the line to patch
            - context_line: the original line (for verification)
            - proposed_replacement: the replacement line

        vault_path: Path to the vault root.

    Returns:
        Dict of { note_path: status } for each patched note.
    """
    vault = vault_path or VAULT_PATH
    results: dict[str, str] = {}

    for patch in patches:
        note_path = vault / patch["target_note_path"]
        line_num = patch["line_number"]
        expected_line = patch["context_line"]
        replacement = patch["proposed_replacement"]

        try:
            content = note_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Verify the line matches what we expect (safety check)
            if line_num <= len(lines) and lines[line_num - 1].strip() == expected_line:
                # Preserve original indentation
                indent = len(lines[line_num - 1]) - len(lines[line_num - 1].lstrip())
                lines[line_num - 1] = " " * indent + replacement

                note_path.write_text("\n".join(lines), encoding="utf-8")
                results[patch["target_note_path"]] = "patched"
            else:
                results[patch["target_note_path"]] = "skipped (line changed)"

        except Exception as e:
            results[patch["target_note_path"]] = f"error: {str(e)}"

    return results


def scan_note_for_innervation(note_title: str) -> RetroactiveResult:
    """
    Standalone innervation scan for any existing note.
    Find all notes that mention this title but don't link to it.

    Useful for the "Innervate" tab in the UI.
    """
    return find_retroactive_links(note_title)
