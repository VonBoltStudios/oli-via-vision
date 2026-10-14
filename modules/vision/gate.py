"""Vision gate: decide from the transcript alone whether a question needs the camera.

Hard regex patterns, no LLM call. Patterns are matched against a normalized transcript
(lowercase, apostrophes removed, punctuation -> spaces), so write them without
apostrophes: "whats on" not "what's on". Every decision is logged so misses can be
mined to grow the list.
"""

import re
from typing import List, Optional

DEFAULT_PATTERNS = [
    r"\bwhat (do|can) you see\b",
    r"\bwhat you see\b",
    r"\b(do|can) you see\b",
    r"\b(look|looking) (at|around)\b",
    r"\bwhat( is|s) (on|in front of|around|behind|next to)\b",
    r"\bis there (a|an|any)\b.*\b(in front of|around|near|nearby|here|on the|in the room|you see)\b",
    r"\bhow many\b.*\b(do you see|can you see|are there|in front of|around)\b",
    r"\bdescribe (the room|your surroundings|what is around you)\b",
]


def normalize(text: str) -> str:
    text = text.lower().replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


class VisionGate:
    def __init__(self, patterns: Optional[List[str]] = None):
        patterns = patterns or DEFAULT_PATTERNS
        self._compiled = []
        for i, p in enumerate(patterns):
            try:
                self._compiled.append((p, re.compile(p)))
            except re.error as e:
                raise ValueError(f"vision.patterns[{i}] is not a valid regex ({p!r}): {e}")

    def match(self, transcript: str) -> Optional[str]:
        """Return the first matching pattern, or None."""
        norm = normalize(transcript)
        for pattern, rx in self._compiled:
            if rx.search(norm):
                return pattern
        return None
