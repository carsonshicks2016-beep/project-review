"""
Vault Scribe — Duplicate Detector
Detects potential duplicate notes using fuzzy title matching and
TF-IDF content similarity to prevent redundant entries in the vault.
"""

from dataclasses import dataclass

from rapidfuzz import fuzz, process

from backend.config import DUPLICATE_TITLE_THRESHOLD, DUPLICATE_CONTENT_THRESHOLD
from backend.indexer import get_indexer


@dataclass
class DuplicateCandidate:
    """A potential duplicate note found in the vault."""
    title: str              # Existing note title
    filepath: str           # Relative path in vault
    title_similarity: float # Fuzzy title match score (0-100)
    content_similarity: float  # TF-IDF cosine similarity (0-1)
    preview: str            # First paragraph of the existing note
    folder: str             # Folder the note is in


@dataclass
class DuplicateCheckResult:
    """Result of duplicate detection."""
    duplicates: list[DuplicateCandidate]
    has_duplicates: bool
    highest_similarity: float  # Maximum similarity score found


def check_duplicates(
    proposed_title: str,
    content: str = "",
    target_folder: str = "",
) -> DuplicateCheckResult:
    """
    Check if a proposed note might be a duplicate of an existing note.

    Uses two strategies:
    1. Fuzzy title matching against all note titles
    2. TF-IDF content similarity against notes in the target folder

    Args:
        proposed_title: The title of the note being created.
        content: The content of the note being created (for content similarity).
        target_folder: The target folder (prioritize same-folder matches).

    Returns:
        DuplicateCheckResult with potential duplicates.
    """
    indexer = get_indexer()
    index = indexer.index
    candidates: list[DuplicateCandidate] = []
    seen_titles: set[str] = set()

    # ── Strategy 1: Fuzzy Title Matching ────────────────────────────────
    results = process.extract(
        proposed_title,
        index.all_titles,
        scorer=fuzz.token_sort_ratio,
        limit=10,
    )

    for title, score, _ in results:
        if score < DUPLICATE_TITLE_THRESHOLD:
            continue
        if title == proposed_title:
            # Exact title match — definitely a duplicate
            score = 100.0

        note = index.notes.get(title)
        if note:
            candidates.append(DuplicateCandidate(
                title=title,
                filepath=note.relative_path,
                title_similarity=score,
                content_similarity=0.0,  # Will be filled in by Strategy 2
                preview=note.first_paragraph,
                folder=note.folder,
            ))
            seen_titles.add(title)

    # ── Strategy 2: Content Similarity (TF-IDF) ────────────────────────
    if content and target_folder:
        # Get all notes in the target folder
        folder_notes = index.folder_index.get(target_folder, [])

        if folder_notes:
            content_scores = _compute_content_similarity(
                content, folder_notes, index
            )

            for title, similarity in content_scores:
                if similarity < DUPLICATE_CONTENT_THRESHOLD:
                    continue

                if title in seen_titles:
                    # Update the existing candidate with content similarity
                    for cand in candidates:
                        if cand.title == title:
                            cand.content_similarity = similarity
                            break
                else:
                    note = index.notes.get(title)
                    if note:
                        candidates.append(DuplicateCandidate(
                            title=title,
                            filepath=note.relative_path,
                            title_similarity=0.0,
                            content_similarity=similarity,
                            preview=note.first_paragraph,
                            folder=note.folder,
                        ))
                        seen_titles.add(title)

    # ── Sort by combined similarity ─────────────────────────────────────
    candidates.sort(
        key=lambda c: c.title_similarity + (c.content_similarity * 100),
        reverse=True,
    )

    # Limit to top 5
    candidates = candidates[:5]

    highest = 0.0
    if candidates:
        highest = max(
            c.title_similarity + (c.content_similarity * 100)
            for c in candidates
        )

    return DuplicateCheckResult(
        duplicates=candidates,
        has_duplicates=len(candidates) > 0,
        highest_similarity=highest,
    )


def _compute_content_similarity(
    content: str,
    note_titles: list[str],
    index,
) -> list[tuple[str, float]]:
    """
    Compute TF-IDF cosine similarity between new content and existing notes.

    Returns list of (title, similarity_score) sorted by score descending.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        # scikit-learn not available — skip content similarity
        return []

    # Collect documents
    documents = [content]
    titles = ["__new__"]

    for title in note_titles:
        note = index.notes.get(title)
        if note and note.content:
            documents.append(note.content)
            titles.append(title)

    if len(documents) < 2:
        return []

    # Compute TF-IDF matrix
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=1,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Compute cosine similarity of new content against all others
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        # Pair with titles and sort
        results = [(titles[i + 1], float(sim)) for i, sim in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:10]

    except Exception:
        return []
