"""The only way the pipeline reads configuration: config/*.json, generated from model/Atria_Data_Model_Map.xlsx by
scripts/build_config.py (plus the hand-edited config/scoring.json for value-function anchors and gates)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config'


@lru_cache(maxsize=None)
def _load(name: str):
    return json.loads((CONFIG / name).read_text())


def customer_inputs() -> list[dict]:
    return _load('customer_inputs.json')['rows']


def data_map() -> list[dict]:
    return _load('data_map.json')['rows']


def data_map_row(row_id: str) -> dict:
    for r in data_map():
        if r['id'] == row_id:
            return r
    raise KeyError(f'data map has no row {row_id!r}')


def policy_search() -> list[dict]:
    return _load('policy_search.json')['rows']


def formula() -> dict:
    return _load('formula.json')['formula']


def scoring() -> dict:
    return _load('scoring.json')


def customer_default(feeds_contains: str):
    """The parsed default of the customer input whose 'Feeds' cell mentions the text (e.g. 'time term r' -> 0.15)."""
    for r in customer_inputs():
        if feeds_contains.lower() in r['feeds'].lower():
            return r['default_value'] if r['default_value'] is not None else r['default']
    return None


def default_r() -> float:
    v = customer_default('time term r')
    return float(v) if isinstance(v, (int, float)) else 0.15


def reload():
    _load.cache_clear()
