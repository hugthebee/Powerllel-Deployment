"""Refresh schedule per data-map row: next refresh = latest read of that row across sites + its 'How often to update'
(config/data_map.json). Rows nobody reads yet are due now; event-driven rows (per site, per deal, on demand) have no date.
Mock mode only reports."""
from __future__ import annotations

import json
from datetime import date, timedelta

from . import config as C, indicators as I


def schedule(today: date | None = None, db=I.DB) -> list[dict]:
    today = today or date.today()
    rows = I.load(I.connect(db))
    reg = json.loads((C.ROOT / 'data' / 'sources.json').read_text())
    by_row: dict[str, list[str]] = {}
    for s in reg['sources']:
        for i in s['data_map']:
            by_row.setdefault(i, []).append(s['name'])
    out = []
    for r in C.data_map():
        mine = [x for x in rows if x.data_map_id == r['id']]
        reads = [x.read_date for x in mine if x.status == 'Answered' and x.read_date]
        last = max(reads) if reads else None
        days = r['frequency_days']
        statuses = {x.status for x in mine}
        status = 'Answered' if 'Answered' in statuses else 'Example' if 'Example' in statuses else 'Not pulled' if 'Not pulled' in statuses else 'No source'
        if status == 'No source':
            nxt, due = None, False
        elif days is None:
            nxt, due = None, False
        elif last is None:
            nxt, due = today.isoformat(), True
        else:
            nd = date.fromisoformat(last) + timedelta(days=max(1, round(days)))
            nxt, due = nd.isoformat(), nd <= today
        out.append(dict(id=r['id'], category=r['category'], factor=r['factor'], sources=by_row.get(r['id'], []) or [r['data_api']], status=status,
                        last=last, frequency=r['frequency'], frequencyDays=days, next=nxt, due=due))
    return out
