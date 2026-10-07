"""
Vault Scribe — Vault Indexer
Scans the entire Obsidian vault and builds an in-memory index of all notes,
their metadata, wikilinks, titles, and content for fast classification, linking,
and analysis.
"""

import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from backend.config import VAULT_PATH


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class NoteMetadata:
    """Parsed metadata from a vault note's blockquote header."""
    parent: str = ""
    note_type: str = ""
    tags: list[str] = field(default_factory=list)
    genotype: str = ""
    impact: str = ""
    trait: str = ""
    evidence_score: Optional[int] = None
    relevance: str = ""
    genres: str = ""
    artists: str = ""
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class VaultNote:
    """Represents a single note in the vault."""
    title: str                          # Filename without .md
    filepath: Path                      # Absolute path
    relative_path: str                  # Relative to vault root
    folder: str                         # Parent folder relative to vault
    content: str                        # Raw markdown content
    word_count: int                     # Word count
    metadata: NoteMetadata              # Parsed blockquote metadata
    outgoing_links: set[str] = field(default_factory=set)   # [[wikilinks]] this note contains
    incoming_links: set[str] = field(default_factory=set)   # Notes that link TO this note
    headings: list[str] = field(default_factory=list)       # H2+ headings in the note
    first_paragraph: str = ""           # First meaningful paragraph (for previews)


@dataclass
class VaultIndex:
    """The complete in-memory index of the vault."""
    notes: dict[str, VaultNote] = field(default_factory=dict)           # title → VaultNote
    filepath_map: dict[str, str] = field(default_factory=dict)          # relative_path → title
    link_graph: dict[str, set[str]] = field(default_factory=dict)       # title → outgoing links
    backlink_graph: dict[str, set[str]] = field(default_factory=dict)   # title → incoming links
    ghost_links: dict[str, list[str]] = field(default_factory=dict)     # ghost_title → [notes that reference it]
    all_titles: list[str] = field(default_factory=list)                 # sorted list of all titles
    tag_index: dict[str, list[str]] = field(default_factory=dict)       # tag → [note titles]
    folder_index: dict[str, list[str]] = field(default_factory=dict)    # folder → [note titles]

    # Stats
    total_notes: int = 0
    total_links: int = 0
    total_ghosts: int = 0
    total_words: int = 0
    index_time_ms: int = 0
    last_indexed: str = ""


# ── Parsing Helpers ─────────────────────────────────────────────────────────

# Match [[wikilink]] and [[wikilink|display text]]
WIKILINK_PATTERN = re.compile(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]')

# Match blockquote metadata lines: > Key: Value
BLOCKQUOTE_META_PATTERN = re.compile(r'^>\s*(.+?):\s*(.+)$', re.MULTILINE)

# Match inline tags: #tag-name (but not inside code blocks or URLs)
TAG_PATTERN = re.compile(r'(?<!\S)#([a-zA-Z][a-zA-Z0-9_-]+)')

# Match markdown headings
HEADING_PATTERN = re.compile(r'^(#{2,6})\s+(.+)$', re.MULTILINE)


def parse_metadata(content: str) -> NoteMetadata:
    """Extract blockquote metadata from a note's content."""
    meta = NoteMetadata()

    # Only look in the first ~30 lines for metadata
    header_region = "\n".join(content.split("\n")[:30])

    for match in BLOCKQUOTE_META_PATTERN.finditer(header_region):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()

        if key == "parent":
            meta.parent = value
        elif key == "type":
            meta.note_type = value
        elif key == "tags":
            meta.tags = TAG_PATTERN.findall(value)
        elif key == "genotype":
            meta.genotype = value.strip("*")
        elif key == "impact":
            meta.impact = value
        elif key == "trait":
            meta.trait = value
        elif key == "evidence score":
            try:
                meta.evidence_score = int(value.split("/")[0])
            except (ValueError, IndexError):
                pass
        elif key == "relevance":
            meta.relevance = value
        elif key == "relevant genres":
            meta.genres = value
        elif key == "artists":
            meta.artists = value
        else:
            meta.extra[key] = value

    # Also find inline tags not in blockquote
    if not meta.tags:
        # Look for tags in the first few lines (e.g., #Elite-Session after title)
        first_lines = "\n".join(content.split("\n")[:5])
        meta.tags = TAG_PATTERN.findall(first_lines)

    return meta


def extract_wikilinks(content: str) -> set[str]:
    """Extract all [[wikilink]] targets from content."""
    return set(WIKILINK_PATTERN.findall(content))


def extract_headings(content: str) -> list[str]:
    """Extract all H2+ headings from content."""
    return [match.group(2) for match in HEADING_PATTERN.finditer(content)]


def extract_first_paragraph(content: str) -> str:
    """Extract the first meaningful paragraph (not a heading, metadata, or divider)."""
    lines = content.split("\n")
    paragraph_lines = []
    in_paragraph = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines, headings, metadata, dividers
        if not stripped:
            if in_paragraph:
                break
            continue
        if stripped.startswith("#") or stripped.startswith(">") or stripped == "---":
            if in_paragraph:
                break
            continue

        # Skip table lines
        if stripped.startswith("|"):
            if in_paragraph:
                break
            continue

        in_paragraph = True
        paragraph_lines.append(stripped)

        # Limit to ~200 chars
        if sum(len(l) for l in paragraph_lines) > 200:
            break

    return " ".join(paragraph_lines)[:250]


# ── Indexer ─────────────────────────────────────────────────────────────────

class VaultIndexer:
    """Scans and indexes the entire Obsidian vault."""

    def __init__(self, vault_path: Path | None = None):
        self.vault_path = vault_path or VAULT_PATH
        self.index = VaultIndex()

    def build(self) -> VaultIndex:
        """Perform a full vault scan and build the index."""
        start = time.time()
        self.index = VaultIndex()

        # 1. Scan all markdown files
        md_files = sorted(self.vault_path.rglob("*.md"))

        # Skip .obsidian directory and node_modules
        md_files = [
            f for f in md_files
            if ".obsidian" not in f.parts
            and "node_modules" not in f.parts
            and ".git" not in f.parts
        ]

        # 2. Parse each file
        for filepath in md_files:
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                continue

            title = filepath.stem  # filename without .md
            relative = str(filepath.relative_to(self.vault_path))
            folder = str(filepath.parent.relative_to(self.vault_path))

            # Parse components
            metadata = parse_metadata(content)
            outgoing = extract_wikilinks(content)
            headings = extract_headings(content)
            first_para = extract_first_paragraph(content)
            words = len(content.split())

            note = VaultNote(
                title=title,
                filepath=filepath,
                relative_path=relative,
                folder=folder,
                content=content,
                word_count=words,
                metadata=metadata,
                outgoing_links=outgoing,
                headings=headings,
                first_paragraph=first_para,
            )

            self.index.notes[title] = note
            self.index.filepath_map[relative] = title
            self.index.link_graph[title] = outgoing
            self.index.total_words += words

            # Index by folder
            if folder not in self.index.folder_index:
                self.index.folder_index[folder] = []
            self.index.folder_index[folder].append(title)

            # Index by tags
            for tag in metadata.tags:
                tag_key = f"#{tag}" if not tag.startswith("#") else tag
                if tag_key not in self.index.tag_index:
                    self.index.tag_index[tag_key] = []
                self.index.tag_index[tag_key].append(title)

        # 3. Build backlink graph and identify ghosts
        all_title_set = set(self.index.notes.keys())

        for title, outgoing in self.index.link_graph.items():
            for link_target in outgoing:
                self.index.total_links += 1

                if link_target in all_title_set:
                    # Valid link — add to backlinks
                    if link_target not in self.index.backlink_graph:
                        self.index.backlink_graph[link_target] = set()
                    self.index.backlink_graph[link_target].add(title)

                    # Also set incoming_links on the target note
                    self.index.notes[link_target].incoming_links.add(title)
                else:
                    # Ghost link — target doesn't exist
                    if link_target not in self.index.ghost_links:
                        self.index.ghost_links[link_target] = []
                    self.index.ghost_links[link_target].append(title)
                    self.index.total_ghosts += 1

        # 4. Build sorted title list
        self.index.all_titles = sorted(all_title_set)
        self.index.total_notes = len(self.index.notes)

        # 5. Record timing
        elapsed = time.time() - start
        self.index.index_time_ms = int(elapsed * 1000)
        self.index.last_indexed = time.strftime("%Y-%m-%dT%H:%M:%S")

        return self.index

    def get_stats(self) -> dict:
        """Return summary statistics about the indexed vault."""
        return {
            "total_notes": self.index.total_notes,
            "total_links": self.index.total_links,
            "total_ghosts": self.index.total_ghosts,
            "total_words": self.index.total_words,
            "index_time_ms": self.index.index_time_ms,
            "last_indexed": self.index.last_indexed,
            "folders": len(self.index.folder_index),
            "tags": len(self.index.tag_index),
            "orphan_notes": sum(
                1 for title in self.index.notes
                if title not in self.index.backlink_graph
                and not self.index.notes[title].outgoing_links
            ),
        }

    def get_note(self, title: str) -> Optional[VaultNote]:
        """Look up a note by title."""
        return self.index.notes.get(title)

    def search_titles(self, query: str) -> list[tuple[str, float]]:
        """Fuzzy search note titles. Returns [(title, score), ...] sorted by score."""
        from rapidfuzz import fuzz, process

        if not query.strip():
            return []

        results = process.extract(
            query,
            self.index.all_titles,
            scorer=fuzz.token_sort_ratio,
            limit=20,
        )
        return [(title, score) for title, score, _ in results if score > 50]

    def find_notes_mentioning(self, term: str) -> list[tuple[str, str]]:
        """Find all notes that mention a term in their content but don't wikilink it.
        Returns [(title, context_line), ...]."""
        results = []
        term_lower = term.lower()

        for title, note in self.index.notes.items():
            # Skip if the note already links to this term
            if term in note.outgoing_links:
                continue

            # Search content for the term
            content_lower = note.content.lower()
            pos = content_lower.find(term_lower)
            if pos != -1:
                # Extract context line
                start = max(0, note.content.rfind("\n", 0, pos) + 1)
                end = note.content.find("\n", pos)
                if end == -1:
                    end = len(note.content)
                context = note.content[start:end].strip()
                results.append((title, context))

        return results


# ── Module-level singleton ──────────────────────────────────────────────────

_indexer: Optional[VaultIndexer] = None


def get_indexer() -> VaultIndexer:
    """Get or create the global vault indexer."""
    global _indexer
    if _indexer is None:
        _indexer = VaultIndexer()
        _indexer.build()
    return _indexer


def rebuild_index() -> VaultIndex:
    """Force a complete index rebuild."""
    global _indexer
    _indexer = VaultIndexer()
    return _indexer.build()
