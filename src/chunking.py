from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on ". ", "! ", "? " or ".\n", keeping punctuation with each sentence
        sentences = re.split(r'(?<=[\.\!\?]) |(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separators left or empty string separator: fall back to fixed-size split
        if not remaining_separators or remaining_separators[0] == "":
            chunks: list[str] = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        splits = current_text.split(sep)
        # Separator not found in this text; try the next one
        if len(splits) == 1:
            return self._split(current_text, next_seps)

        result: list[str] = []
        current_chunk = ""

        for i, piece in enumerate(splits):
            # Re-attach separator to all pieces except the last
            piece_with_sep = piece + (sep if i < len(splits) - 1 else "")
            if not piece_with_sep:
                continue

            if len(piece_with_sep) > self.chunk_size:
                # Flush the accumulated chunk first, then recurse on the oversized piece
                if current_chunk:
                    result.append(current_chunk)
                    current_chunk = ""
                result.extend(self._split(piece_with_sep, next_seps))
            elif len(current_chunk) + len(piece_with_sep) <= self.chunk_size:
                current_chunk += piece_with_sep
            else:
                if current_chunk:
                    result.append(current_chunk)
                current_chunk = piece_with_sep

        if current_chunk:
            result.append(current_chunk)

        return result


class CustomChunker:
    """
    Optimized for Vietnamese legal documents (Quy chế / Thông tư).

    These documents have a 4-level hierarchy:
        ## Chương  →  ### Điều X.  →  1. 2. 3.  →  a) b) c)

    The previous implementation split on "." as a sentence boundary, which
    incorrectly fragmented numbered clauses like "1. Quy chế này quy định..."
    and alpha sub-items. Average clause length in these docs is 140–444 chars
    with individual clauses reaching 2 600+ chars — far too long for sentence
    splitting to be useful.

    Strategy:
        1. Split on the legal hierarchy in priority order:
               ### heading  →  ## heading  →  numbered clause (\\n1.)
               →  alpha sub-clause (\\na))  →  paragraph (\\n\\n)  →  line (\\n)
        2. Greedily pack units into chunks up to ``max_size`` chars.
        3. Carry the last complete clause/sub-item of each chunk as overlap
           into the next chunk, so cross-boundary references are preserved.
    """

    # Lookaheads keep the delimiter attached to the following unit.
    # Order matters: most specific first.
    _SPLIT_RE = re.compile(
        r'(?=\n#{3}\s)'           # ### Điều heading
        r'|(?=\n#{2}\s)'          # ## Chương heading
        r'|(?=\n\d{1,2}\.\s)'     # numbered clause  \n1.  \n12.  (not \n1.23)
        r'|(?=\n[a-zđ]\)\s)'      # alpha sub-clause \na)  \nđ)
        r'|\n\n'                   # paragraph break
    )

    def __init__(self, max_size: int = 500, overlap: int = 100) -> None:
        self.max_size = max(1, max_size)
        self.overlap = min(overlap, max_size // 2)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        units = self._split_into_units(text)
        if not units:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        current = ""
        overlap_seed = ""

        for unit in units:
            probe = (overlap_seed + unit) if not current else (current + unit)

            if len(probe) <= self.max_size:
                current = probe
            else:
                if current.strip():
                    chunks.append(current.strip())
                    overlap_seed = self._tail_clause(current, self.overlap)

                if len(unit) > self.max_size:
                    # Unit too big even alone: split on single newlines, then hard-split
                    for sub in self._split_lines(unit):
                        if sub.strip():
                            chunks.append(sub.strip())
                    overlap_seed = self._tail_clause(chunks[-1], self.overlap) if chunks else ""
                    current = ""
                else:
                    current = overlap_seed + unit

        if current.strip():
            chunks.append(current.strip())

        return chunks

    # ── helpers ──────────────────────────────────────────────────────────────

    def _split_into_units(self, text: str) -> list[str]:
        """Split on the legal hierarchy boundaries."""
        return [u for u in self._SPLIT_RE.split(text) if u and u.strip()]

    def _tail_clause(self, text: str, n: int) -> str:
        """
        Return the last clause/sub-item of ``text`` up to ``n`` chars.

        Tries (in order):
          1. Last alpha sub-clause boundary \\na) or \\nb)
          2. Last numbered clause boundary \\n1. \\n2.
          3. Last paragraph break \\n\\n
          4. Last line break \\n
          5. Last word boundary
        """
        if len(text) <= n:
            return text
        tail = text[-n:]
        for pattern in (r'\n[a-zđ]\)\s', r'\n\d{1,2}\.\s', r'\n\n', r'\n'):
            match = None
            for m in re.finditer(pattern, tail):
                match = m
            if match:
                return tail[match.start():]
        space = tail.rfind(" ")
        return tail[space + 1:] if space != -1 else tail

    def _split_lines(self, text: str) -> list[str]:
        """Split an oversized unit on single newlines, hard-split anything still too big."""
        result: list[str] = []
        for line in text.split("\n"):
            if not line.strip():
                continue
            if len(line) <= self.max_size:
                result.append(line)
            else:
                result.extend(line[i: i + self.max_size] for i in range(0, len(line), self.max_size))
        return result


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
            "hybrid": CustomChunker(max_size=chunk_size, overlap=chunk_size // 5),
        }
        result = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
