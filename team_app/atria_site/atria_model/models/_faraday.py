"""Bridge to the Faraday engine vendored from atria-power-desk (faraday/atria_app, flat imports, unchanged)."""
import importlib
import json
import sys
from pathlib import Path

FARADAY = Path(__file__).resolve().parents[2] / 'faraday'
APP = FARADAY / 'atria_app'
ANALYSIS = FARADAY / 'data' / 'sites' / 'analysis'


def mod(name: str):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    return importlib.import_module(name)


def cached_analysis(site_id: str) -> dict | None:
    """The last default Faraday run for the site (saved by atria-power-desk), so pages open without a model run."""
    try:
        return json.loads((ANALYSIS / f'{site_id}.json').read_text())
    except (OSError, ValueError):
        return None
