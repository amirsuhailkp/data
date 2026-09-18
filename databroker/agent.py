"""
The Agent Orchestrator: implements the investigation loop from spec section 10.

    Understand objective -> plan -> gather evidence -> evaluate ->
    (follow leads if needed) -> cross-check -> update knowledge ->
    score importance -> notify / store / ignore

Token-conservation design: the LLM is only called for the things only it can
do — reading/understanding fetched text, extracting claims from it, and
resolving genuinely ambiguous judgment calls (is this development an update
to something we already knew? does it support or weaken a specific thesis
point?). Every mechanical decision — building search queries, recognizing an
obvious duplicate or an obviously unrelated new item, scoring importance from
source reliability + keyword signals, deciding whether two claims are even
about the same topic before checking for a contradiction, rolling up an
overall thesis-impact label from per-point results — is handled by
`heuristics.py` with zero LLM calls. Each step below is either a full
heuristic short-circuit (no LLM call at all) or a "heuristic first, LLM only
if genuinely ambiguous, with the smallest possible prompt" pattern.
"""

from __future__ import annotations
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from .db import DB
from .llm import LLMProvider, HybridProvider
from .search import SearchProvider, MockSearch
from .social import SocialProvider, FinancialProvider
from .fetcher import PageFetcher, MockFetcher
from . import heuristics

SOURCE_RELIABILITY = {
    "filing": 1,
    "regulatory": 2,
    "investor_relations": 3,
    "primary_statement": 4,
    "news": 5,
    "industry_pub": 6,
    "community": 7,
    "social": 8,
}

# Fallback framing for a financial/trading-platform hit that doesn't supply
# its own context_label (kept for backward compatibility with any provider
# written before that field existed — StockTwits, Finnhub, and SEC all set
# their own label now, see social.py).
DEFAULT_FINANCIAL_LABEL = (
    "[TRADING PLATFORM CHATTER — real-time trader sentiment, not verified fact; "
    "only extract a claim from this if it describes something concrete (e.g. a "
    "specific event being discussed), and keep confidence low unless corroborated "
    "elsewhere in this content]"
)

RESEARCH_SYSTEM_PROMPT = """You are DataBroker, a disciplined investment research analyst.
You investigate companies the way a careful human analyst would: you weigh source
reliability, distinguish genuinely new information from repeats, flag contradictions
instead of silently picking a side, and separate evidence from opinion.
You are fluent in common trading strategy frameworks (momentum/breakout, swing,
mean-reversion, catalyst/event-driven, value) well enough to note which kind of
trader a development is most relevant to and why — but this is framing, not advice:
you never issue a buy/sell recommendation, a price target, or a predicted price
movement. When evidence mentions a concrete upcoming date (earnings call, court/
regulatory ruling, product launch, conference), surface it as a scheduled event to
watch, never as a prediction of what will happen at it.
You never recommend buying or selling. You never invent facts, sources, or numbers.
If you are not confident in something, say so explicitly and lower your confidence rating
rather than presenting a guess as fact. You will sometimes be given raw web content that has
already been retrieved for you — only use claims that are directly supported by that content,
and only cite URLs that appear verbatim in it. Never invent or guess a URL. Answer only the
specific question asked, as briefly as the required JSON shape allows — you are one step in a
larger pipeline, not the whole report."""

# Env-tunable limits (kept as module constants with sane defaults so a
# resource-constrained local model gets a smaller prompt out of the box).
import os
MAX_SEARCH_HITS_PER_QUESTION = int(os.environ.get("MAX_SEARCH_HITS_PER_QUESTION", "6"))
MAX_FETCH_CHARS = int(os.environ.get("MAX_FETCH_CHARS", "1500"))
MAX_SUBQUESTIONS = int(os.environ.get("RESEARCH_MAX_SUBQUESTIONS", "6"))
MAX_SOCIAL_HITS_PER_QUESTION = int(os.environ.get("MAX_SOCIAL_HITS_PER_QUESTION", "3"))
MAX_FINANCIAL_HITS = int(os.environ.get("MAX_FINANCIAL_HITS", "5"))


@dataclass
class ResearchResult:
    objective: str
    plan: list[str]
    claims: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    report: str = ""
    llm_calls_made: int = 0  # rough token-usage visibility, see _call()
    activity_log: list[str] = field(default_factory=list)  # what the agent did + which model, see _log()


class ResearchAgent:
    def __init__(self, db: DB, llm: LLMProvider, search: SearchProvider | None = None,
                 fetcher: PageFetcher | None = None, social: SocialProvider | None = None,
                 financial: FinancialProvider | None = None, notifier=None,
                 confirmer: FinancialProvider | None = None, technicals=None):
        self.db = db
        self.llm = llm
        self.search = search or MockSearch()
        self.fetcher = fetcher or MockFetcher()
        self.social = social  # None = Phase 4 social intelligence disabled (the default)
        self.financial = financial  # None = financial/trading platforms disabled (the default)
        self.notifier = notifier  # None = alerts disabled (see notify.py); set by build_agent() from NOTIFY_BACKEND
        # Quota-limited source used ONLY to corroborate an already-notable
        # finding, never polled every tick — see _confirm_event(). Currently
        # Alpha Vantage (25 req/day total on the free tier), which this
        # spends at most once per notable event instead of once per sweep.
        self.confirmer = confirmer
        # Deterministic chart/price data (see technicals.py) — computed with
        # zero LLM calls, handed to build_report() as read-only CONTEXT.
        # None = disabled (the default).
        self.technicals = technicals
        self._call_count = 0
        self.activity: list[str] = []  # human-readable trace of what this agent instance has done

    def _confirm_event(self, ticker: str, claim_text: str) -> str | None:
        """Second-opinion lookup against the quota-limited confirmation source,
        fired only after something else already found something notable.

        Returns a short human-readable corroboration note, or None if there's
        no confirmer configured, the lookup fails/quota is exhausted (the
        provider fails soft to []), or nothing came back. Deliberately does NOT
        call the LLM — this is a cheap "does an independent source also report
        this?" check whose result gets folded into the alert text."""
        if self.confirmer is None:
            return None
        self._log(f"Corroborating notable event against confirmation source ({type(self.confirmer).__name__})...")
        try:
            hits = self.confirmer.get_ticker_activity(ticker, max_results=5)
        except Exception as e:
            self._log(f"Confirmation lookup failed ({e}) — continuing without it.")
            return None
        if not hits:
            self._log("Confirmation source returned nothing (no coverage, or daily quota exhausted).")
            return "⚠️ Not corroborated — the confirmation source returned no matching coverage (it may also be out of daily quota)."
        # Surface what it independently reports, so the sentiment labels this
        # source attaches are visible alongside the original finding.
        lines = [f"- {h['title']}" for h in hits[:3]]
        self._log(f"Confirmation source returned {len(hits)} independent item(s).")
        return "✅ Independently reported by the confirmation source:\n" + "\n".join(lines)

    def _notify(self, text: str) -> None:
        """Push an alert if a notifier is configured. Two-stage flow (see investigate()):
        a quick alert the moment a notable event is found, then a follow-up with the
        fuller analysis once the report for that session is ready. Never raises —
        a broken notifier should never take down a research/sweep run."""
        if self.notifier is None:
            return
        try:
            if self.notifier.send(text):
                self._log("Alert sent.")
            else:
                self._log("Alert FAILED to send (notifier reported failure).")
        except Exception as e:  # a notifier bug is not worth crashing the research run over
            self._log(f"Alert FAILED to send (unexpected error: {e}).")

    def _log(self, msg: str) -> None:
        """Record one step of what the agent is doing. Kept both in-memory (so it can
        be persisted to research_sessions.activity_json for later review via the
        `history` command) and printed live to stderr, so a long `research`/`sweep`/
        `watch-loop` run shows real progress instead of going silent for minutes."""
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        self.activity.append(line)
        print(f"[databroker] {line}", file=sys.stderr)

    def _provider_for_task(self, task: str) -> LLMProvider:
        """Which underlying provider will actually serve this task. HybridProvider
        routes different tasks to different providers, so `self.llm` alone doesn't
        tell you which model is about to be used for a given step."""
        if isinstance(self.llm, HybridProvider):
            return self.llm.overrides.get(task, self.llm.default)
        return self.llm

    def _call(self, task: str, prompt: str, schema_hint: str) -> dict:
        """Every actual LLM call funnels through here so call counts, and which model
        actually handled each step, stay visible."""
        self._call_count += 1
        provider = self._provider_for_task(task)
        self._log(f"LLM call #{self._call_count} (task={task}) -> {provider.describe()}")
        result = self.llm.complete_json(RESEARCH_SYSTEM_PROMPT, prompt, schema_hint, task=task)
        # describe() can change as a result of the call itself — e.g. PooledProvider
        # records which underlying provider/model/key actually served this specific
        # request (see llm.py).
        after = provider.describe()
        self._log(f"LLM call #{self._call_count} (task={task}) done -> {after}")
        return result

    # ---------- Step 1: Understand + plan (heuristic short-circuit for simple asks) ----------
    def plan_research(self, objective: str, company_name: str) -> list[str]:
        self._log(f"Planning research for {company_name}: \"{objective}\"")
        if heuristics.is_simple_objective(objective):
            # "Anything new about Tesla?" doesn't need an LLM call to decompose —
            # it already IS the sub-question.
            self._log("Objective is simple enough to research directly — skipping planning LLM call.")
            return [objective]

        result = self._call(
            "plan",
            prompt=(
                f"Company: {company_name}\n"
                f"Research objective: {objective}\n\n"
                f"Break this into up to {MAX_SUBQUESTIONS} concrete sub-questions a research "
                "analyst would investigate to answer the objective well. Order them from most "
                "to least important."
            ),
            schema_hint='{"sub_questions": ["...", "..."]}',
        )
        subs = result.get("sub_questions") or [objective]
        return subs[:MAX_SUBQUESTIONS]

    # ---------- Step 2: search planning — fully deterministic, zero LLM calls ----------
    def plan_search_queries(self, sub_question: str, company_name: str, ticker: str) -> list[str]:
        return heuristics.build_search_queries(company_name, ticker, sub_question)

    # ---------- Step 3: retrieve + extract structured claims (the one step that must read text) ----------
    def gather_evidence(self, sub_question: str, company_name: str, ticker: str,
                         extra_hits: list[dict] | None = None) -> list[dict]:
        """`extra_hits` is pre-fetched "first line" content (ticker-keyed financial
        sources: SEC filings, Finnhub news, StockTwits chatter — see investigate())
        that gets merged in ahead of general web search results, rather than
        fetched fresh per sub-question the way social search hits are. Each hit
        may carry its own `context_label` describing how the LLM should treat
        that source type (falls back to DEFAULT_FINANCIAL_LABEL if absent)."""
        queries = self.plan_search_queries(sub_question, company_name, ticker)

        hits = []
        seen_urls = set()
        for q in queries:
            for h in self.search.search(q, max_results=5):
                url = h.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    hits.append(h)

        # Phase 4: social/community discussion, opt-in via SOCIAL_BACKEND. Tagged
        # and framed distinctly below — these are treated as sentiment/discussion,
        # not verified fact, and their source_type is trusted from the provider
        # (not re-guessed by the LLM) since a smaller model is prone to mistaking
        # a Reddit post for "news".
        social_hits = []
        if self.social is not None:
            for h in self.social.search(f"{company_name} {sub_question}", max_results=MAX_SOCIAL_HITS_PER_QUESTION):
                url = h.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    social_hits.append(h)

        financial_hits = []
        for h in (extra_hits or []):
            url = h.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                financial_hits.append(h)

        if not hits and not social_hits and not financial_hits:
            return []

        hit_lookup = {h["url"]: h for h in hits + social_hits + financial_hits}

        context_blocks = []
        # Financial/trading platform chatter goes first — "first line of
        # information" is reflected in prompt ordering, not just in being
        # included at all.
        for h in financial_hits:
            content = h.get("snippet", "")
            if not content:
                continue
            label = h.get("context_label", DEFAULT_FINANCIAL_LABEL)
            context_blocks.append(
                f"{label}\nURL: {h['url']}\nTitle: {h.get('title', '')}\n"
                f"Content: {content[:MAX_FETCH_CHARS]}"
            )
        for h in hits[:MAX_SEARCH_HITS_PER_QUESTION]:
            page_text = self.fetcher.fetch(h["url"], max_chars=MAX_FETCH_CHARS, sub_question=sub_question)
            content = page_text or h.get("snippet", "")
            if not content:
                continue
            context_blocks.append(
                f"URL: {h['url']}\nTitle: {h.get('title', '')}\nContent: {content[:MAX_FETCH_CHARS]}"
            )
        for h in social_hits:
            # No extra page fetch for social hits — the provider's own snippet
            # (post body / self-text) is already the actual content, not just
            # a search-result teaser, so fetching again would be redundant.
            content = h.get("snippet", "")
            if not content:
                continue
            context_blocks.append(
                "[COMMUNITY DISCUSSION — reflects public sentiment/speculation, not verified fact; "
                "only extract a claim from this if it describes something concrete, and keep confidence "
                f"low unless corroborated elsewhere in this content]\nURL: {h['url']}\n"
                f"Title: {h.get('title', '')}\nContent: {content[:MAX_FETCH_CHARS]}"
            )

        if not context_blocks:
            return []

        context = "\n\n---\n\n".join(context_blocks)
        result = self._call(
            "extract",
            prompt=(
                f"Company: {company_name} ({ticker})\nQuestion: {sub_question}\n\n"
                f"Web content retrieved for this question:\n\n{context}\n\n"
                "Extract only claims directly supported by the content above. Copy source_url "
                "EXACTLY from one of the URLs given above — never invent or modify a URL. If "
                "nothing above actually answers the question, return an empty claims list "
                "rather than guessing. For each claim, also note any named entities (other "
                "companies, people, products, regulators) it involves and any explicit "
                "relationship between them (e.g. 'acquired', 'partnered_with', 'invested_in', "
                "'competitor_of', 'ceo_of', 'sued') — skip this if the claim doesn't clearly "
                "state a relationship between two named things."
            ),
            schema_hint=(
                '{"claims": [{"text": "...", "source_url": "...", '
                '"source_type": "filing|regulatory|investor_relations|primary_statement|'
                'news|industry_pub|community|social", "publication_date": "YYYY-MM-DD or null", '
                '"confidence": "low|medium|high", '
                '"relationships": [{"subject": "...", "predicate": "...", "object": "..."}]}]}'
            ),
        )
        claims = result.get("claims", [])
        # Guard against a model hallucinating a URL not actually in the retrieved context —
        # local/smaller models are more prone to this than larger hosted ones.
        claims = [c for c in claims if c.get("source_url") in seen_urls]

        # Deterministic correction: for hits whose type we actually know (currently
        # just social/community ones — the plain web SearchProvider doesn't classify),
        # trust that over whatever the LLM guessed, and backfill publication_date
        # if the provider had it and the LLM didn't find one.
        for c in claims:
            hit = hit_lookup.get(c.get("source_url"))
            if hit and hit.get("source_type"):
                c["source_type"] = hit["source_type"]
            if hit and hit.get("publication_date") and not c.get("publication_date"):
                c["publication_date"] = hit["publication_date"]

        return claims

    # ---------- Step 4: Cross-check / conflict detection (LLM only for overlapping pairs) ----------
    def cross_check_and_store(self, company_id: int, claims: list[dict]) -> tuple[list[int], list[dict]]:
        """Store claims, mark cross-checked when 2+ independent sources agree (deterministic
        text-similarity grouping), and flag conflicts only for claim pairs that plausibly
        discuss the same fact (deterministic keyword-overlap pre-filter) — an LLM is only
        asked to adjudicate those specific pairs, not the whole claim set."""
        stored_ids = []
        conflicts = []
        groups: dict[str, list[int]] = {}

        for c in claims:
            tier = SOURCE_RELIABILITY.get(c.get("source_type", "news"), 5)
            source_id = self.db.add_source(
                url=c.get("source_url"),
                source_type=c.get("source_type"),
                publication_date=c.get("publication_date"),
                reliability_tier=tier,
            )
            claim_id = self.db.add_claim(
                company_id=company_id,
                text=c["text"],
                source_id=source_id,
                confidence=c.get("confidence", "medium"),
            )
            stored_ids.append(claim_id)
            key = heuristics.normalize(c["text"])[:60]
            groups.setdefault(key, []).append(claim_id)

            # Phase 5: knowledge graph — store any relationships this claim
            # mentioned (from the same extraction call, no extra LLM cost).
            # Entity resolution (matching "NVIDIA" / "NVIDIA Corp." / etc. to
            # one entity, and linking it to a tracked company if applicable)
            # is entirely deterministic — see graph.py / db.upsert_entity.
            for rel in c.get("relationships", []) or []:
                subject, predicate, obj = rel.get("subject"), rel.get("predicate"), rel.get("object")
                if not subject or not predicate or not obj:
                    continue
                subject_id = self.db.upsert_entity(subject)
                object_id = self.db.upsert_entity(obj)
                self.db.add_relationship(subject_id, predicate, object_id, claim_id, c.get("confidence"))

        for key, ids in groups.items():
            if len(ids) >= 2:
                for cid in ids:
                    self.db.update_claim_verification(cid, "cross_checked")

        overlapping_pairs = heuristics.find_overlapping_claim_pairs(claims)
        if overlapping_pairs:
            # Only the overlapping subset goes to the LLM, not every claim gathered this session.
            involved_indices = sorted({i for pair in overlapping_pairs for i in pair})
            index_map = {orig: pos for pos, orig in enumerate(involved_indices)}
            subset_text = "\n".join(f"- ({index_map[i]}) {claims[i]['text']}" for i in involved_indices)
            check = self._call(
                "conflict",
                prompt=(
                    "These claims share enough overlapping subject matter that they might be "
                    f"describing the same fact:\n{subset_text}\n\n"
                    "Do any of them contradict each other (e.g. different figures, different "
                    "people, different dates for the same event)? Only flag genuine "
                    "contradictions, not merely related-but-different details."
                ),
                schema_hint='{"conflicts": [{"a_index": 0, "b_index": 1, "description": "..."}]}',
            )
            for conf in check.get("conflicts", []):
                a_idx, b_idx = conf.get("a_index"), conf.get("b_index")
                if a_idx is None or b_idx is None or a_idx >= len(involved_indices) or b_idx >= len(involved_indices):
                    continue
                a_orig, b_orig = involved_indices[a_idx], involved_indices[b_idx]
                a_id, b_id = stored_ids[a_orig], stored_ids[b_orig]
                self.db.update_claim_verification(a_id, "conflicting")
                self.db.update_claim_verification(b_id, "conflicting")
                conflict_id = self.db.add_conflict(a_id, b_id, conf.get("description", ""))
                conflicts.append({"id": conflict_id, **conf})

        return stored_ids, conflicts

    # ---------- Step 5: change detection (heuristic resolves most cases) ----------
    def dedupe_against_history(self, company_id: int, candidate_title: str, candidate_desc: str):
        """Return ('new', None) / ('duplicate', event_id) / ('updated', event_id)."""
        recent = self.db.recent_events(company_id, days=60)
        classification, match_id, ambiguous_candidates = heuristics.classify_against_history(
            candidate_desc, recent
        )
        if classification != "ambiguous":
            return classification, match_id

        # Only the true middle-ground cases reach the LLM, and only with the
        # top few most-similar candidates rather than the full event history.
        candidates_text = "\n".join(
            f"- (id={r['id']}) {r['title']}: {r['description']}" for r in ambiguous_candidates
        )
        result = self._call(
            "dedupe",
            prompt=(
                f"New candidate development:\nTitle: {candidate_title}\nDescription: {candidate_desc}\n\n"
                f"Closest previously recorded developments for this company:\n{candidates_text}\n\n"
                "Is the new development: (a) genuinely new and unrelated to anything above, "
                "(b) an exact repeat/duplicate of something above, or (c) an update/progression "
                "of something above (e.g. 'planned' -> 'approved')? If (b) or (c), give the id "
                "of the matching event."
            ),
            schema_hint='{"classification": "new|duplicate|updated", "matching_event_id": null}',
        )
        return result.get("classification", "new"), result.get("matching_event_id")

    # ---------- Step 6: importance scoring (keyword+source heuristic first) ----------
    def score_importance(self, company_name: str, event_title: str, event_desc: str,
                          source_type: str | None, source_confidence: str | None, has_thesis: bool) -> dict:
        level, rationale, confident, worth_review_regardless = heuristics.baseline_importance(
            event_desc, source_type, source_confidence
        )
        if confident:
            return {"importance_level": level, "rationale": rationale,
                    "confidence": source_confidence or "medium", "source": "heuristic"}

        if not has_thesis and not worth_review_regardless:
            # No confident keyword/source signal, nothing to check thesis-relevance
            # against, and nothing serious-looking enough to warrant a second look
            # on its own — spending an LLM call here buys little, so default low.
            return {"importance_level": "low",
                    "rationale": "No confident heuristic signal and no thesis on file to weigh relevance against.",
                    "confidence": "low", "source": "heuristic_default"}

        result = self._call(
            "score",
            prompt=(
                f"Company: {company_name}\nDevelopment: {event_title} — {event_desc}\n"
                f"Source type: {source_type or 'unknown'}\n\n"
                "Score this development's importance to an investor holding/watching this "
                "company, considering magnitude, novelty, source reliability (an unverified "
                "forum/community post claiming something serious should usually be scored "
                "lower than the same claim from an official source, unless corroborated), "
                "and potential financial/competitive/regulatory/market impact.\n"
                "low = store for later, not worth surfacing now\n"
                "medium = include in a daily/weekly digest\n"
                "high = notify the user now\n"
                "critical = notify immediately and warrants deeper investigation"
            ),
            schema_hint=(
                '{"importance_level": "low|medium|high|critical", "rationale": "...", '
                '"confidence": "low|medium|high"}'
            ),
        )
        result.setdefault("source", "llm")
        return result

    # ---------- Step 7: thesis monitor (only run for events that clear the importance bar) ----------
    def assess_thesis_points(self, thesis_points: list[dict], event_title: str, event_desc: str) -> list[dict]:
        if not thesis_points:
            return []
        points_text = "\n".join(f"- (id={p['id']}) {p['point_text']}" for p in thesis_points)
        result = self._call(
            "thesis",
            prompt=(
                f"Investment thesis points:\n{points_text}\n\n"
                f"New development: {event_title} — {event_desc}\n\n"
                "For each thesis point this development actually bears on (skip points it "
                "doesn't affect), say whether it supports, weakens, contradicts, or creates a "
                "new risk/opportunity relative to that point, with a one-sentence reason. If it "
                "doesn't bear on any point, return an empty list."
            ),
            schema_hint=(
                '{"assessments": [{"point_id": 0, "status": '
                '"supported|weakened|contradicted|new_risk|new_opportunity", "note": "..."}]}'
            ),
        )
        return result.get("assessments", [])

    # ---------- Full loop for a research objective ----------
    def investigate(self, ticker: str, company_name: str, company_id: int, objective: str) -> ResearchResult:
        plan = self.plan_research(objective, company_name)
        thesis, thesis_points = self.db.get_latest_thesis(company_id)

        result = ResearchResult(objective=objective, plan=plan)

        # Financial/trading platform activity (StockTwits etc.) is ticker-scoped,
        # not free-text searchable — fetch it once per research session rather
        # than once per sub-question, and feed it into only the first (highest-
        # priority, per plan_research's ordering) sub-question's extraction.
        # Repeating identical trader chatter across every sub-question's prompt
        # would cost tokens without adding information.
        financial_hits = []
        if self.financial is not None:
            financial_hits = self.financial.get_ticker_activity(ticker, max_results=MAX_FINANCIAL_HITS)

        # Chart/price data, same story: fetched once per session (it's a
        # snapshot of current price action, not something that changes
        # per sub-question) and handed to build_report() as context, never
        # re-fetched or re-described per sub-question.
        technical_snapshot = None
        if self.technicals is not None:
            self._log(f"Fetching technical snapshot via {self.technicals.describe()}...")
            try:
                technical_snapshot = self.technicals.get_snapshot(ticker)
            except Exception as e:
                self._log(f"Technical snapshot fetch failed ({e}) — continuing without it.")
            if technical_snapshot:
                self._log("Technical snapshot obtained.")
            else:
                self._log("No technical snapshot available (insufficient history, or source unconfigured).")

        for i, sub_q in enumerate(plan):
            claims = self.gather_evidence(
                sub_q, company_name, ticker,
                extra_hits=financial_hits if i == 0 else None,
            )
            if not claims:
                continue
            claim_ids, conflicts = self.cross_check_and_store(company_id, claims)
            result.claims.extend(claims)
            result.conflicts.extend(conflicts)

            for c in claims:
                classification, match_id = self.dedupe_against_history(company_id, sub_q, c["text"])
                if classification == "duplicate":
                    continue  # already known — no further LLM work spent on it

                scoring = self.score_importance(
                    company_name, sub_q, c["text"],
                    source_type=c.get("source_type"),
                    source_confidence=c.get("confidence"),
                    has_thesis=bool(thesis_points),
                )
                importance = scoring.get("importance_level", "low")
                confidence = scoring.get("confidence", "medium")

                # Thesis assessment is the most expensive remaining step (it needs the
                # actual thesis wording), so it's gated: skip it for "low" importance
                # events, since those are unlikely to meaningfully move any thesis point.
                assessments = []
                if thesis_points and importance != "low":
                    assessments = self.assess_thesis_points([dict(p) for p in thesis_points], sub_q, c["text"])
                    for a in assessments:
                        pid = a.get("point_id")
                        if pid is not None:
                            self.db.update_thesis_point(pid, a.get("status", "unassessed"), a.get("note", ""))

                thesis_impact = heuristics.aggregate_thesis_impact(assessments) if assessments else "neutral"

                if classification == "updated" and match_id:
                    self.db.mark_event_updated(match_id, c["text"])
                    event_id = match_id
                else:
                    event_id = self.db.add_event(
                        company_id=company_id,
                        title=sub_q,
                        description=c["text"],
                        importance_level=importance,
                        importance_rationale=scoring.get("rationale", ""),
                        thesis_impact=thesis_impact,
                        confidence=confidence,
                        source_ids=[],
                        status="new",
                    )

                result.events.append({
                    "id": event_id, "title": sub_q, "description": c["text"],
                    "importance_level": importance, "thesis_impact": thesis_impact,
                    "confidence": confidence,
                })

                # Stage 1 of the alert flow: a quick "something happened" ping the
                # moment a non-low-importance event is stored, before the (slower,
                # more expensive) full report is synthesized below. Day trading is
                # latency-sensitive, so this doesn't wait for the whole session to finish.
                if importance != "low":
                    icon = {"critical": "🚨", "high": "🔔"}.get(importance, "ℹ️")
                    # Spend one confirmation-source call here — only now that
                    # something has actually cleared the importance bar.
                    confirmation = self._confirm_event(ticker, c["text"])
                    msg = (
                        f"{icon} {ticker} — {sub_q}\n{c['text'][:280]}\n"
                        f"Importance: {importance} | Thesis impact: {thesis_impact}"
                    )
                    if confirmation:
                        msg += f"\n\n{confirmation}"
                    self._notify(msg)

        result.report = self.build_report(company_name, objective, result, thesis["summary"] if thesis else None,
                                           technical_snapshot=technical_snapshot)
        result.llm_calls_made = self._call_count
        self.db.log_session(company_id, objective, plan, result.report)

        # Stage 2: the fuller analysis, sent once per session rather than once per
        # event above — this is the synthesis across everything found, not a repeat
        # of the quick pings. Only sent if something above actually cleared the
        # importance bar; a session with only "low" events already stayed quiet at
        # stage 1, so it stays quiet here too rather than pushing a report nobody
        # asked to see yet.
        if self.notifier is not None and any(e["importance_level"] != "low" for e in result.events):
            self._notify(f"📊 Analysis — {company_name} ({ticker}):\n\n{result.report}")

        return result

    # ---------- Market-wide discovery: proposing NEW watchlist candidates ----------
    def discover_candidates(self, scanners, exclude: set[str], min_score: int = 3,
                            limit: int = 5) -> list[dict]:
        """Scan the market for symbols you AREN'T watching that are showing
        unusual activity, and characterize the top few.

        Costs exactly ONE LLM call for the whole scan regardless of how many
        symbols the scanners return, because the deterministic ranking in
        discover.rank_candidates() does the filtering first (see that module
        for why corroboration across signal types is the ranking basis).

        Returns the ranked candidates, each with an added "note" explaining
        what the activity appears to be about and what to check. These are
        research leads, not recommendations — see the prompt below.
        """
        from .discover import rank_candidates

        raw: list[dict] = []
        for sc in scanners:
            self._log(f"Scanning market via {sc.describe()}...")
            try:
                hits = sc.scan()
            except Exception as e:
                self._log(f"Scanner {sc.describe()} failed ({e}) — continuing without it.")
                continue
            self._log(f"{sc.describe()} returned {len(hits)} signal(s).")
            raw.extend(hits)

        if not raw:
            self._log("No market signals returned by any scanner.")
            return []

        candidates = rank_candidates(raw, exclude=exclude, min_score=min_score, limit=limit)
        self._log(f"{len(candidates)} candidate(s) cleared the corroboration threshold "
                  f"(min_score={min_score}) out of {len({h['symbol'] for h in raw})} distinct symbols seen.")
        if not candidates:
            return []

        described = "\n".join(
            f"- {c['symbol']} (corroboration score {c['score']}): "
            + "; ".join(c["signals"].values())
            for c in candidates
        )
        result = self._call(
            "discover",
            prompt=(
                "These stock symbols were flagged by automated market scanners for unusual "
                "activity today. The scanners report only WHAT is unusual (volume, price move, "
                "chatter volume), never WHY.\n\n"
                f"{described}\n\n"
                "For each symbol, write one short note covering: what the company is, if you "
                "know it; what kinds of things typically cause this particular activity pattern; "
                "and what a researcher would need to check to find out which applies here. "
                "Be explicit that unusual activity is equally consistent with good news, bad "
                "news, and manipulation — a big move is not evidence of a good opportunity. "
                "If you do not recognize a symbol, say so plainly rather than guessing what "
                "the company is. Do NOT recommend buying or selling, do not predict direction, "
                "and do not assign a price target."
            ),
            schema_hint='{"notes": [{"symbol": "...", "note": "..."}]}',
        )
        notes = {
            (n.get("symbol") or "").upper(): n.get("note", "")
            for n in (result.get("notes") or []) if isinstance(n, dict)
        }
        for c in candidates:
            c["note"] = notes.get(c["symbol"], "")
        return candidates

    # ---------- Step 8: report generation (deterministic short-circuit when there's nothing to report) ----------
    def build_report(self, company_name: str, objective: str, result: ResearchResult,
                     thesis_summary: str | None, technical_snapshot: str | None = None) -> str:
        if not result.claims:
            # No LLM call here regardless of whether technical_snapshot is set —
            # a snapshot on its own, with no news/evidence to give it context,
            # isn't worth spending a call to have the LLM describe. The snapshot
            # itself is free (already computed, no LLM involved) so it's still
            # shown below for visibility.
            technical_note = f"\n\n**Technical snapshot:**\n```\n{technical_snapshot}\n```" if technical_snapshot else ""
            return (
                f"## Research report: {company_name}\n**Question:** {objective}\n\n"
                "No evidence was found for this question in the sources checked this run. "
                "No conclusions drawn — try again later or broaden the question."
                f"{technical_note}"
            )

        technical_block = (
            f"\nTechnical/price data for this session (computed deterministically from real "
            f"market data, NOT by you — treat these as facts about past price action, never "
            f"as a forecast):\n{technical_snapshot}\n"
            if technical_snapshot else ""
        )

        synthesis = self._call(
            "report",
            prompt=(
                f"Company: {company_name}\nOriginal question: {objective}\n"
                f"Investment thesis: {thesis_summary or 'None recorded'}\n"
                f"{technical_block}\n"
                f"Evidence gathered this session:\n{result.claims}\n\n"
                f"Any source conflicts found:\n{result.conflicts}\n\n"
                "Write the analyst report described below. Do not recommend buying or "
                "selling; describe what the evidence supports or challenges, and your "
                "overall confidence. If evidence conflicts, say so explicitly rather than "
                "picking a side. Separately, list any concrete upcoming catalysts explicitly "
                "mentioned in the evidence above (e.g. a scheduled earnings date, court/"
                "regulatory decision date, product launch, conference) — only ones actually "
                "stated in the evidence, never invented or guessed; return an empty list if "
                "none are mentioned. Do not predict what will happen at them.\n\n"
                "If technical/price data was provided above, you may use it to add context to "
                "the report — e.g. noting that a piece of news arrives while the stock sits "
                "near a notable technical level, or that RSI/volatility describe an unusually "
                "quiet or volatile stretch. This is framing only. You must NEVER use technical "
                "data to predict a future price, direction, or magnitude of movement, and "
                "NEVER combine it with news to produce something that reads as a forecast "
                "(e.g. 'this earnings beat plus the breakout above resistance suggests further "
                "upside' is forbidden — describing the earnings beat AND separately noting the "
                "stock is near its recent high is fine). If there is nothing in the evidence to "
                "connect to the technical data, leave the technical data undiscussed rather "
                "than forcing a connection."
            ),
            schema_hint=(
                '{"overall_assessment": "...", "strengths": ["..."], "weaknesses": ["..."], '
                '"improving": ["..."], "deteriorating": ["..."], "new_risks": ["..."], '
                '"upcoming_catalysts": ["..."], '
                '"confidence": "low|medium|high", "watch_next": ["..."]}'
            ),
        )
        lines = [f"## Research report: {company_name}", f"**Question:** {objective}", ""]
        if technical_snapshot:
            lines.append(f"**Technical snapshot:**\n```\n{technical_snapshot}\n```\n")
        lines.append(f"**Overall assessment:** {synthesis.get('overall_assessment', 'n/a')}")
        for label, key in [("Strengths", "strengths"), ("Weaknesses", "weaknesses"),
                            ("Improving", "improving"), ("Deteriorating", "deteriorating"),
                            ("New risks", "new_risks"), ("Upcoming catalysts", "upcoming_catalysts"),
                            ("What to watch next", "watch_next")]:
            items = synthesis.get(key) or []
            if items:
                lines.append(f"\n**{label}:**")
                lines.extend(f"- {i}" for i in items)
        lines.append(f"\n**Confidence:** {synthesis.get('confidence', 'n/a')}")
        if result.conflicts:
            lines.append("\n**Note:** conflicting information was found between sources for at least "
                          "one claim above — see the conflicts log rather than treating either as settled.")
        return "\n".join(lines)

    # ---------- Digest — no LLM calls at all, pure DB read + formatting ----------
    def daily_digest(self) -> str:
        sections = []
        for row in self.db.list_watchlist():
            events = self.db.unnotified_events(row["id"], min_level="medium")
            if not events:
                continue
            sections.append(f"### {row['name']} ({row['ticker']})")
            for e in sorted(events, key=lambda r: {"critical": 0, "high": 1, "medium": 2}.get(r["importance_level"], 3)):
                sections.append(
                    f"- **{e['title']}** — {e['description']}\n"
                    f"  Thesis impact: {e['thesis_impact']} | Confidence: {e['confidence']} | "
                    f"Level: {e['importance_level']}"
                )
                self.db.mark_notified(e["id"])
        if not sections:
            return "No new developments meeting the digest threshold today."
        return "Good morning. Here's what matters:\n\n" + "\n".join(sections)
