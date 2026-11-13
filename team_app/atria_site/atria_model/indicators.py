"""The Indicator long table: every value for every site, one row each, with provenance.

Postgres-ready schema (portable DDL; SQLite for now). A value is numeric or text; status says whether it was answered:
  Answered    read from a source on read_date
  Example     a mock / sample value standing in until the source is pulled (shown with an "Example" tag, never as fact)
  Not pulled  a source exists but was not called for this site
  No source   nothing answers it yet (data-map row marked "Need to add")
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config as C

STATUSES = ('Answered', 'Example', 'Not pulled', 'No source')
SCORED = ('Answered', 'Example')   # values a score may use; the other two are skipped and flagged
DB = C.ROOT / 'data' / 'indicators.sqlite'

DDL = """
CREATE TABLE IF NOT EXISTS indicator (
    site_id      TEXT NOT NULL,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    data_map_id  TEXT NOT NULL,
    value_num    DOUBLE PRECISION,
    value_text   TEXT,
    unit         TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL,
    how          TEXT NOT NULL,
    read_date    TEXT,
    status       TEXT NOT NULL CHECK (status IN ('Answered', 'Example', 'Not pulled', 'No source')),
    note         TEXT NOT NULL DEFAULT '',
    url          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (site_id, name)
);
CREATE INDEX IF NOT EXISTS indicator_site_category ON indicator (site_id, category);
"""


@dataclass
class Indicator:
    site_id: str
    category: str          # Technical | Environmental | Cultural | Policy | Security | Economics
    name: str              # e.g. "dist_345kv_mi"
    value: float | str | None
    unit: str
    source: str            # e.g. "Atria substations", "ERCOT LLI status 2026-09", "Customer provided"
    how: str               # Atria module | API | Download | Crawled | Derived (model) | Customer provided
    read_date: str | None
    status: str = 'Answered'
    data_map_id: str = ''
    note: str = ''
    url: str = ''

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f'{self.name}: status {self.status!r} not in {STATUSES}')
        if self.status in ('Not pulled', 'No source'):
            self.value, self.read_date = None, None
        if self.data_map_id:
            C.data_map_row(self.data_map_id)   # raises on an unknown row

    @property
    def scored(self) -> bool:
        return self.status in SCORED and self.value is not None


def connect(path: Path | str = DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(DDL)
    return con


def upsert(con: sqlite3.Connection, rows: list[Indicator]):
    con.executemany(
        'INSERT OR REPLACE INTO indicator (site_id, name, category, data_map_id, value_num, value_text, unit, source, how, read_date, status, note, url) '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        [(r.site_id, r.name, r.category, r.data_map_id, r.value if isinstance(r.value, (int, float)) and not isinstance(r.value, bool) else None,
          None if r.value is None or (isinstance(r.value, (int, float)) and not isinstance(r.value, bool)) else str(r.value),
          r.unit, r.source, r.how, r.read_date, r.status, r.note, r.url) for r in rows])
    con.commit()


def load(con: sqlite3.Connection, site_id: str | None = None) -> list[Indicator]:
    q = 'SELECT site_id, category, name, value_num, value_text, unit, source, how, read_date, status, data_map_id, note, url FROM indicator'
    cur = con.execute(q + (' WHERE site_id = ?' if site_id else '') + ' ORDER BY site_id, category, name', (site_id,) if site_id else ())
    return [Indicator(site_id=a, category=b, name=c, value=d if d is not None else e, unit=f, source=g, how=h, read_date=i, status=j,
                      data_map_id=k, note=l, url=m) for a, b, c, d, e, f, g, h, i, j, k, l, m in cur.fetchall()]


def values(rows: list[Indicator]) -> dict[str, float | str]:
    """name -> value for the rows a score may use. Not pulled / No source rows are absent, so they are skipped, not zero."""
    return {r.name: r.value for r in rows if r.scored}


def as_dict(r: Indicator) -> dict:
    return asdict(r)
