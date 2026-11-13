"""LLM extraction over policy text (minutes, agendas, news, filings, public posts) and its validation.

Rules: output that is not valid JSON is dropped; events with an unknown type or stage, or a malformed date, are dropped;
statements with an unknown stance are dropped; a quote that does not appear verbatim in the source drops its statement.
Nothing is repaired or guessed. sentiment_features() aggregates statements to jurisdiction-month features for model B
and keeps no personal data (no names, no handles, only counts and shares).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

STAGES = ['public_request', 'staff_study', 'first_reading', 'adopted', 'extended', 'repealed', 'rejected']
EVENT_TYPES = ['moratorium', 'rezoning', 'permit', 'ordinance', 'hearing']
STANCES = ['support', 'oppose', 'neutral']
ROLES = ['resident', 'official', 'developer', 'business', 'other']
CONCERNS = ['electricity_bills', 'water', 'noise', 'air_quality', 'traffic', 'land_use',
            'property_values', 'heritage', 'tax_breaks', 'developer_trust', 'other']

EXTRACT_PROMPT = """You read a local government or news document about land use.
Return JSON only:
{
 "relevant": true/false,                      # is it about a data center, large load, or on-site power plant?
 "events": [{"type": "moratorium|rezoning|permit|ordinance|hearing",
             "stage": one of %s,
             "date": "YYYY-MM-DD", "body": "...", "vote_for": int|null, "vote_against": int|null,
             "project": "...", "developer": "...", "mw": number|null, "location": "..."}],
 "statements": [{"speaker_role": "resident|official|developer|business|other",
                 "stance": "support|oppose|neutral",
                 "concerns": subset of %s,
                 "quote": "short exact quote"}]
}
Quote only text that appears in the document. If unsure, use null.""" % (STAGES, CONCERNS)

EMPTY = {'relevant': False, 'events': [], 'statements': [], 'dropped': []}
_DATE = re.compile(r'\d{4}-\d{2}-\d{2}')


def _norm(s: str) -> str:
    s = s.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', s).strip().lower()


def quote_in_source(quote: str, source: str) -> bool:
    """Verbatim up to whitespace, case and curly quotes. Paraphrases fail."""
    q = _norm(quote or '')
    return bool(q) and q in _norm(source)


def validate(raw: str, source_text: str) -> dict:
    """Parse and check one LLM answer. Returns the kept events and statements plus a list of what was dropped and why."""
    try:
        out = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return dict(EMPTY, dropped=['not valid JSON'])
    if not isinstance(out, dict):
        return dict(EMPTY, dropped=['not a JSON object'])
    dropped, events, statements = [], [], []
    for e in out.get('events') or []:
        if not isinstance(e, dict):
            dropped.append('event: not an object'); continue
        if e.get('type') not in EVENT_TYPES:
            dropped.append(f'event: type {e.get("type")!r}'); continue
        if e.get('stage') not in STAGES:
            dropped.append(f'event: stage {e.get("stage")!r}'); continue
        if e.get('date') is not None and not (isinstance(e['date'], str) and _DATE.fullmatch(e['date'])):
            dropped.append(f'event: date {e.get("date")!r}'); continue
        for k in ('vote_for', 'vote_against'):
            if e.get(k) is not None and not isinstance(e[k], int):
                e[k] = None
        if e.get('mw') is not None and not isinstance(e['mw'], (int, float)):
            e['mw'] = None
        events.append(e)
    for s in out.get('statements') or []:
        if not isinstance(s, dict):
            dropped.append('statement: not an object'); continue
        if s.get('stance') not in STANCES:
            dropped.append(f'statement: stance {s.get("stance")!r}'); continue
        if not quote_in_source(s.get('quote'), source_text):
            dropped.append('statement: quote not found in source'); continue
        s['speaker_role'] = s.get('speaker_role') if s.get('speaker_role') in ROLES else 'other'
        s['concerns'] = [c for c in (s.get('concerns') or []) if c in CONCERNS]
        statements.append(s)
    return {'relevant': out.get('relevant') is True, 'events': events, 'statements': statements, 'dropped': dropped}


def llm_extract(document_text: str, call_llm) -> dict:
    """call_llm(prompt, text) -> str. Validated JSON out; anything that fails validation is dropped, not guessed."""
    return validate(call_llm(EXTRACT_PROMPT, document_text), document_text)


def sentiment_features(statements: list[dict], posts_per_week: list[int] | None = None) -> dict:
    """Aggregate stance into features for model B. With no statements every feature is None (unknown, not zero)."""
    posts_per_week = posts_per_week or []
    if not statements:
        return {'share_oppose': None, 'net_stance': None, 'volume': 0, 'volume_trend': None, 'top_concern': None, 'concerns': {}}
    n = len(statements)
    oppose = sum(s['stance'] == 'oppose' for s in statements)
    support = sum(s['stance'] == 'support' for s in statements)
    counts = {c: sum(c in s.get('concerns', []) for s in statements) for c in CONCERNS}
    trend = (posts_per_week[-1] - posts_per_week[0]) / max(1, posts_per_week[0]) if len(posts_per_week) > 1 else None
    return {
        'share_oppose': oppose / n,                 # level of opposition
        'net_stance': (support - oppose) / n,       # -1 .. +1
        'volume': n,                                # how loud
        'volume_trend': trend,                      # getting louder?
        'top_concern': max(counts, key=counts.get) if any(counts.values()) else None,
        'concerns': {c: k for c, k in counts.items() if k},
    }


def jurisdiction_month_features(statements: list[dict]) -> dict[tuple[str, str], dict]:
    """statements carry 'jurisdiction' and 'date' (YYYY-MM-DD); returns {(jurisdiction, 'YYYY-MM'): features}."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in statements:
        if s.get('jurisdiction') and isinstance(s.get('date'), str) and len(s['date']) >= 7:
            groups[(s['jurisdiction'], s['date'][:7])].append(s)
    return {k: sentiment_features(v) for k, v in sorted(groups.items())}
