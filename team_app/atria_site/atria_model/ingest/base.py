from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .. import config as C
from ..indicators import Indicator

MOCK = json.loads((C.ROOT / 'data' / 'examples' / 'mock_values.json').read_text())
MOCK_DATE = MOCK['read_date']
MODE = 'mock'   # no network calls in this build


@dataclass
class Source:
    id: str
    name: str
    url: str
    data_map: list[str]
    how: str
    kind: str = 'public'          # public | atria | customer | model | llm
    cost: str = 'Free'
    status: str = 'Not pulled'    # Answered | Not pulled | No source (for the source as a whole)
    fetch_date: str | None = None
    note: str = ''
    sites: dict = field(default_factory=dict)   # site_id -> status

    def as_dict(self) -> dict:
        d = asdict(self)
        rows = [C.data_map_row(i) for i in self.data_map]
        days = [r['frequency_days'] for r in rows if r['frequency_days']]
        d['frequency'] = '; '.join(sorted({r['frequency'] for r in rows}))
        d['frequency_days'] = min(days) if days else None
        return d


def row(site_id: str, data_map: str, name: str, value, *, unit: str = '', source: str, how: str, date: str | None,
        status: str = 'Answered', note: str = '', url: str = '') -> Indicator:
    cat = C.data_map_row(data_map)['category']
    return Indicator(site_id=site_id, category=cat, name=name, value=value, unit=unit, source=source, how=how, read_date=date,
                     status=status, data_map_id=data_map, note=note, url=url)


def example(site_id: str, src: Source, data_map: str, name: str, value, unit: str = '', note: str = '') -> Indicator:
    """A mock value standing in for `src` until it is pulled."""
    status = 'Example' if value is not None else 'Not pulled'
    return row(site_id, data_map, name, value, unit=unit, source=f'Example (would read {src.name})', how=src.how, date=MOCK_DATE if value is not None else None,
               status=status, note=note or 'Made-up value for the demo; replaced when the source is pulled.', url=src.url)


def mock_for(site_id: str, key: str) -> dict:
    return (MOCK['sites'].get(site_id) or {}).get(key) or {}
