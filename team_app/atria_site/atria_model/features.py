"""Value functions: raw indicator value -> 0..1 on FIXED anchors read from config/scoring.json, so adding a site never
re-scales another. Missing values are skipped and flagged, never counted as zero."""
from __future__ import annotations

from . import config as C


def linear_value(x: float | None, best: float, worst: float) -> float | None:
    """1 at `best`, 0 at `worst`, linear between, clamped; works whether lower or higher is better. None stays None."""
    if x is None:
        return None
    u = (float(x) - worst) / (best - worst)
    return max(0.0, min(1.0, u))


def value_functions() -> dict[str, dict]:
    return C.scoring()['value_functions']


def numeric(name: str, x):
    """Text values that have an ordinal meaning (moratorium stage) become numbers; anything else non-numeric is unusable."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if name == 'moratorium_stage':
        return C.scoring()['moratorium_stage_ordinal'].get(str(x))
    try:
        return float(x)
    except ValueError:
        return None


def score_indicator(name: str, x) -> float | None:
    vf = value_functions().get(name)
    if vf is None:
        return None
    return linear_value(numeric(name, x), vf['best'], vf['worst'])


def category_detail(values: dict) -> dict[str, dict]:
    """Per category: s_k (None if nothing answered), the indicators used with their u and weight, and the ones missing."""
    out = {k: {'score': None, 'used': [], 'missing': []} for k in C.scoring()['categories']}
    for name, vf in value_functions().items():
        u = score_indicator(name, values.get(name))
        slot = out.setdefault(vf['category'], {'score': None, 'used': [], 'missing': []})
        if u is None:
            slot['missing'].append(name)
        else:
            slot['used'].append({'name': name, 'u': u, 'weight': vf['weight'], 'value': values.get(name)})
    for slot in out.values():
        tw = sum(i['weight'] for i in slot['used'])
        slot['score'] = sum(i['weight'] * i['u'] for i in slot['used']) / tw if tw else None
        slot['coverage'] = tw / (tw + sum(value_functions()[n]['weight'] for n in slot['missing'])) if (tw or slot['missing']) else 0.0
    return out


def category_scores(values: dict) -> dict[str, float]:
    """s_k = weighted average of the answered indicators in category k; a category with nothing answered is left out."""
    return {k: d['score'] for k, d in category_detail(values).items() if d['score'] is not None}


def gates(flags: dict) -> dict[str, dict]:
    """Each fatal flaw as pass / fail / unknown. flags[gate] is True (flaw present), False (checked, absent) or missing (unknown)."""
    out = {}
    for key, g in C.scoring()['gates'].items():
        v = flags.get(key)
        out[key] = dict(g, state='fail' if v is True else 'pass' if v is False else 'unknown')
    return out
