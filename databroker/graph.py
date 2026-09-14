"""
Phase 5: Knowledge graph — entity resolution helpers.

Deterministic, zero-token, consistent with heuristics.py's approach elsewhere
in this codebase: matching "NVIDIA Corporation" and "Nvidia Corp." to the
same graph entity doesn't need an LLM, just name normalization. The LLM's
job (in agent.py's extraction call) is only to identify which entities and
relationships are mentioned in a claim — resolving those mentions to a
canonical entity is handled here.
"""

from __future__ import annotations
import re

# Common corporate suffixes stripped before comparing names, so "NVIDIA
# Corporation", "Nvidia Corp.", and "NVIDIA" all normalize to "nvidia".
CORPORATE_SUFFIXES = [
    "corporation", "corp", "incorporated", "inc", "limited", "ltd",
    "llc", "l.l.c", "company", "co", "plc", "ag", "sa", "nv", "group", "holdings",
]

_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in CORPORATE_SUFFIXES) + r")\.?\b", re.IGNORECASE
)


def normalize_entity_name(name: str) -> str:
    """Canonical form used for entity dedup matching — not for display."""
    n = name.strip().lower()
    n = _SUFFIX_PATTERN.sub("", n)
    n = re.sub(r"[^\w\s]", "", n)  # drop punctuation
    n = re.sub(r"\s+", " ", n).strip()
    return n or name.strip().lower()  # never return empty — fall back to the raw lowercased name


def normalize_predicate(predicate: str) -> str:
    """Relationship labels get light normalization too, so 'acquired' and
    'acquires' (tense variation across different claims/sources) collapse to
    one edge type instead of fragmenting the graph. Deliberately simple —
    a handful of common variants, not a full lemmatizer."""
    p = predicate.strip().lower().replace(" ", "_")
    variants = {
        "acquires": "acquired", "acquiring": "acquired", "buys": "acquired", "bought": "acquired",
        "partners_with": "partnered_with", "partnering_with": "partnered_with",
        "invests_in": "invested_in", "investing_in": "invested_in",
        "competes_with": "competitor_of", "competing_with": "competitor_of", "rival_of": "competitor_of",
        "sues": "sued", "suing": "sued",
        "ceo_of": "ceo_of", "is_ceo_of": "ceo_of",
        "supplies": "supplier_of", "supplying": "supplier_of",
    }
    return variants.get(p, p)
