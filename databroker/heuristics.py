"""
Deterministic, zero-token helpers.

The design goal after the "don't make the LLM do every step" change: the LLM
is reserved for things only it can do — reading fetched text and extracting
claims, resolving genuinely ambiguous judgment calls, and comparing evidence
against the user's own thesis wording. Everything mechanical — building
search queries, spotting obvious duplicates, scoring importance from source
reliability + keyword signals, and deciding whether two claims are even
about the same topic before bothering to ask the LLM to check for a
contradiction — is handled here in plain Python.

Every function returns either a final, confident answer (LLM is skipped
entirely) or a signal that the case is ambiguous and needs an LLM call —
plus, where relevant, a trimmed-down candidate list so that LLM call is as
small as possible.
"""

from __future__ import annotations
import difflib
import re

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "will", "with", "at", "by", "its", "their", "this", "that",
    "has", "have", "had", "be", "as", "it", "from", "new", "over", "into",
}

# Keyword tiers used as a fast first pass on importance. Not exhaustive —
# just enough to confidently resolve the obvious cases without an LLM call.
CRITICAL_KEYWORDS = [
    "bankruptcy", "files for chapter 11", "fraud investigation", "sec investigation",
    "criminal charges", "accounting scandal", "ceo resigns", "ceo fired", "ceo ousted",
    "delisted", "data breach", "hacked", "recall of all", "safety recall",
]
HIGH_KEYWORDS = [
    "acquisition", "acquires", "acquired", "merger", "lawsuit", "sues", "sued",
    "regulatory investigation", "antitrust", "guidance cut", "cuts guidance",
    "earnings miss", "missed estimates", "downgrade", "upgrade", "recall",
    "layoffs", "plant closure", "strike", "patent lawsuit", "ban", "banned",
    "sanctions", "tariff",
]
MEDIUM_KEYWORDS = [
    "partnership", "partners with", "expansion", "expands", "new product",
    "launches", "price increase", "price cut", "hiring", "new factory",
    "joint venture", "earnings beat", "beat estimates", "buyback", "dividend",
    "contract win", "new customer",
]

SOURCE_TIER_FLOOR = {
    # source_type -> minimum importance level this source type alone can justify
    # without keyword hints (a filing is rarely "low", a social post rarely "high")
    "filing": "medium",
    "regulatory": "medium",
    "investor_relations": "medium",
}

LEVEL_ORDER = ["low", "medium", "high", "critical"]


def _max_level(a: str, b: str) -> str:
    return a if LEVEL_ORDER.index(a) >= LEVEL_ORDER.index(b) else b


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    result = set()
    for w in words:
        if w in STOPWORDS:
            continue
        # Keep short alphanumeric tokens like "q3" or "5b" — they carry real
        # signal for spotting numeric contradictions even though they'd
        # normally fail a plain length filter.
        if len(w) > 2 or any(ch.isdigit() for ch in w):
            result.add(w)
    return result


def text_similarity(a: str, b: str) -> float:
    """0..1 similarity, cheap stdlib-only (difflib), good enough to separate
    'clearly the same sentence' from 'clearly unrelated' without an LLM call."""
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def overlap_coefficient(a: str, b: str) -> float:
    """|A∩B| / min(|A|,|B|) — unlike Jaccard, doesn't get diluted when one
    claim is much longer/wordier than the other, which matters here because
    a genuine numeric contradiction (e.g. differing revenue figures) is often
    a single token buried in otherwise differently-worded sentences."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# ---------------------------------------------------------------------------
# Search query construction — no LLM call needed for this at all.
# ---------------------------------------------------------------------------
def build_search_queries(company_name: str, ticker: str, sub_question: str, max_queries: int = 3) -> list[str]:
    # Strip generic question phrasing so the query reads like something a
    # person would actually type into a search box. Applied twice since
    # compound leads ("what" + "is") are common and a single pass only
    # strips the outermost word.
    cleaned = sub_question.strip()
    lead_pattern = re.compile(
        r"^(what is|what are|what's|how is|how are|why is|why are|is there|are there|"
        r"does|do|is|are|check|investigate|find out|look into|any update on|"
        r"anything new (about|on|with))\b[:,]?\s*",
        re.IGNORECASE,
    )
    for _ in range(2):
        new_cleaned = lead_pattern.sub("", cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned
    cleaned = cleaned.rstrip("?").strip() or sub_question.strip()

    queries = [
        f"{company_name} {cleaned}",
        f"{ticker} {cleaned} news",
        f"{company_name} investor relations {cleaned}",
    ]
    # de-dupe while preserving order
    seen = set()
    unique = []
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    return unique[:max_queries]


# ---------------------------------------------------------------------------
# Research planning short-circuit — skip the decomposition LLM call for
# objectives that are already a single narrow question.
# ---------------------------------------------------------------------------
def is_simple_objective(objective: str, word_threshold: int = 9) -> bool:
    words = objective.strip().split()
    has_conjunction = bool(re.search(r"\b(and|or|,|;)\b", objective, re.IGNORECASE))
    return len(words) <= word_threshold and not has_conjunction


# ---------------------------------------------------------------------------
# Change detection pre-filter — resolves the obvious ends of the similarity
# spectrum without an LLM call; only the ambiguous middle band needs one,
# and even then only the top few candidates are sent.
# ---------------------------------------------------------------------------
def classify_against_history(
    candidate_text: str, recent_events: list, duplicate_threshold: float = 0.75,
    new_threshold: float = 0.2, top_k_for_llm: int = 3,
):
    """
    Returns one of:
      ("duplicate", event_id, None)          -- confident, no LLM needed
      ("new", None, None)                    -- confident, no LLM needed
      ("ambiguous", None, [candidate_rows])  -- needs an LLM call, list trimmed to top_k

    Uses word-level Jaccard rather than character-level similarity: two
    claims that both happen to start with "Company ..." but are about
    unrelated topics score low here, whereas char-level similarity on short
    strings tends to over-credit shared common words and falsely looks close.
    """
    if not recent_events:
        return "new", None, None

    scored = sorted(
        ((jaccard(candidate_text, r["description"] or r["title"]), r) for r in recent_events),
        key=lambda t: t[0], reverse=True,
    )
    best_score, best_row = scored[0]

    if best_score >= duplicate_threshold:
        return "duplicate", best_row["id"], None
    if best_score < new_threshold:
        return "new", None, None

    top_candidates = [row for _, row in scored[:top_k_for_llm]]
    return "ambiguous", None, top_candidates


# ---------------------------------------------------------------------------
# Importance scoring — keyword + source-reliability baseline. Only genuinely
# unclear cases (no keyword hit, mid-tier source) are escalated to an LLM.
# ---------------------------------------------------------------------------
def baseline_importance(text: str, source_type: str | None, source_confidence: str | None):
    """Returns (level, rationale, confident: bool, worth_llm_review_regardless: bool).

    The 4th field distinguishes two different kinds of "not confident": a
    routine ambiguous case (only worth an LLM call if there's a thesis to
    check relevance against) versus a potentially serious claim from an
    unverified source (worth a second look even with no thesis on file,
    because "possible bankruptcy" matters to a watcher regardless of
    whether they've written down a thesis yet)."""
    t = text.lower()

    keyword_level = None
    if any(k in t for k in CRITICAL_KEYWORDS):
        keyword_level = "critical"
    elif any(k in t for k in HIGH_KEYWORDS):
        keyword_level = "high"
    elif any(k in t for k in MEDIUM_KEYWORDS):
        keyword_level = "medium"

    tier_floor = SOURCE_TIER_FLOOR.get(source_type or "")
    is_unverified_social = source_type in ("community", "social")

    if keyword_level and tier_floor:
        level = _max_level(keyword_level, tier_floor)
        return level, f"Keyword match ('{keyword_level}' signal) from a {source_type} source.", True, False

    if keyword_level in ("critical", "high"):
        if is_unverified_social:
            return keyword_level, (
                f"Keyword match ('{keyword_level}') from an unverified {source_type} source — "
                "not auto-confirmed, needs review."
            ), False, True  # worth a look even with no thesis on file
        return keyword_level, f"Keyword match indicates a {keyword_level}-impact development.", True, False

    if keyword_level == "medium" and source_confidence in ("medium", "high"):
        return "medium", "Keyword match indicates a moderate development.", True, False

    if not keyword_level and source_type == "social":
        return "low", "No strong keyword signal; social-media-only source.", True, False

    if not keyword_level and is_unverified_social:
        return "low", f"No strong keyword signal; unverified {source_type} source.", True, False

    if not keyword_level and tier_floor:
        return tier_floor, f"No strong keyword signal, but sourced from a {source_type} (baseline importance).", True, False

    return "low", "No confident heuristic match.", False, False


# ---------------------------------------------------------------------------
# Conflict pre-filter — only claims that plausibly discuss the same fact
# (meaningful token overlap) are worth an LLM contradiction check.
# ---------------------------------------------------------------------------
def find_overlapping_claim_pairs(claims: list[dict], overlap_threshold: float = 0.25) -> list[tuple[int, int]]:
    """Note: this is a cheap recall-over-precision filter, not a semantic check —
    it will occasionally miss a real contradiction phrased very differently, and
    occasionally send an unrelated-but-topically-similar pair to the LLM for
    nothing. Both failure modes are intentionally the safe direction: a missed
    heuristic conflict is caught later if it recurs, and a wasted LLM call here
    is one tiny call, not a systemic risk."""
    pairs = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if overlap_coefficient(claims[i]["text"], claims[j]["text"]) >= overlap_threshold:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Thesis-impact aggregation — once per-point assessments come back from the
# (single, already-necessary) LLM call, the overall event-level thesis_impact
# is just a deterministic rollup, not a separate LLM judgment.
# ---------------------------------------------------------------------------
def aggregate_thesis_impact(assessments: list[dict]) -> str:
    statuses = {a.get("status") for a in assessments}
    if "contradicted" in statuses:
        return "contradicts"
    if "weakened" in statuses:
        return "weakens"
    if "new_risk" in statuses:
        return "new_risk"
    if "supported" in statuses:
        return "supports"
    if "new_opportunity" in statuses:
        return "new_opportunity"
    return "neutral"
