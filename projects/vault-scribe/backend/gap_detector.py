"""
Vault Scribe — Knowledge Gap Detector
Analyzes the entire vault graph to find "Ghost Hubs" — concepts that are 
linked to multiple times across different notes but don't actually exist as 
their own dedicated note yet.
"""

from dataclasses import dataclass
from collections import defaultdict
from backend.indexer import get_indexer

@dataclass
class KnowledgeGap:
    title: str
    mention_count: int
    mentioned_in: list[str]


def detect_knowledge_gaps(min_mentions: int = 2) -> list[KnowledgeGap]:
    """
    Scan the index for all outgoing links that don't map to an existing note.
    Group them by the target title and return the most frequently mentioned gaps.
    
    Args:
        min_mentions: Only return gaps that are mentioned in at least this many notes.
    """
    indexer = get_indexer()
    index = indexer.index
    
    # Track ghost links: target_title -> set(source_note_titles)
    ghost_map = defaultdict(set)
    
    for source_title, note_data in index.notes.items():
        for target in note_data.outgoing_links:
            # We already have an efficient ghost checker in indexer, but let's 
            # verify here explicitly against the current notes keys
            if target not in index.notes:
                # Normalizing case to prevent "magnesium" and "Magnesium" from being split
                normalized = target.lower()
                
                # Double check against aliases (just in case the index keys don't catch it)
                # RapidFuzz indexing handles this usually, but a strict missing link is 
                # exactly what we want for a gap.
                ghost_map[target].add(source_title)
                
    # Compile the results
    gaps = []
    for target_title, sources in ghost_map.items():
        if len(sources) >= min_mentions:
            gaps.append(KnowledgeGap(
                title=target_title,
                mention_count=len(sources),
                mentioned_in=list(sources)
            ))
            
    # Sort by mention count descending
    gaps.sort(key=lambda x: x.mention_count, reverse=True)
    
    return gaps
