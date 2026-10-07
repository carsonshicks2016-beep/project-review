"""
Vault Scribe — Ghost Checker
Validates that all [[wikilinks]] in generated content resolve to existing notes.
Prevents the creation of ghost nodes in the vault graph.
"""

from dataclasses import dataclass

from backend.indexer import get_indexer, WIKILINK_PATTERN


@dataclass
class GhostLink:
    """A wikilink that doesn't resolve to any existing note."""
    target: str             # The link target (note title that doesn't exist)
    context: str            # The line where the ghost link appears
    line_number: int        # Line number in the content (1-indexed)


@dataclass
class GhostCheckResult:
    """Result of ghost link validation."""
    valid_links: list[str]      # Links that resolve to existing notes
    ghost_links: list[GhostLink]  # Links that don't resolve
    total_links: int
    is_clean: bool              # True if no ghost links found


def check_ghosts(content: str) -> GhostCheckResult:
    """
    Check all [[wikilinks]] in content against the vault index.

    Args:
        content: Markdown content to validate.

    Returns:
        GhostCheckResult with valid links and ghost links.
    """
    indexer = get_indexer()
    index = indexer.index
    existing_titles = set(index.notes.keys())

    valid: list[str] = []
    ghosts: list[GhostLink] = []
    seen: set[str] = set()

    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        for match in WIKILINK_PATTERN.finditer(line):
            target = match.group(1).strip()
            if target in seen:
                continue
            seen.add(target)

            if target in existing_titles:
                valid.append(target)
            else:
                ghosts.append(GhostLink(
                    target=target,
                    context=line.strip(),
                    line_number=line_num,
                ))

    return GhostCheckResult(
        valid_links=valid,
        ghost_links=ghosts,
        total_links=len(seen),
        is_clean=len(ghosts) == 0,
    )


def remove_ghost_links(content: str, ghosts_to_remove: list[str]) -> str:
    """
    Remove specific ghost [[wikilinks]] from content, replacing them with plain text.

    Args:
        content: The markdown content.
        ghosts_to_remove: List of ghost link targets to remove.

    Returns:
        Content with ghost links replaced by plain text.
    """
    import re
    result = content
    for ghost in ghosts_to_remove:
        # Replace [[ghost]] with just "ghost"
        pattern = re.compile(
            r'\[\[' + re.escape(ghost) + r'(?:\|([^\]]+))?\]\]'
        )
        # If there's display text, use that; otherwise use the link target
        result = pattern.sub(
            lambda m: m.group(1) if m.group(1) else ghost,
            result,
        )
    return result


def get_vault_ghosts() -> dict[str, list[str]]:
    """
    Get all ghost links currently in the vault.

    Returns:
        Dict of ghost_title → [notes that reference it].
    """
    indexer = get_indexer()
    return dict(indexer.index.ghost_links)
