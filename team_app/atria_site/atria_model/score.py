"""One formula (tab "4 Formula"):

    AtriaScore = 100 x G x sum_k w_k*s_k x (1 - P_stall) x (1 + r)^(-E[T_power]/12)

Gates come first: any fatal flaw sets G = 0. A gate that could not be checked is 'unknown': it does not zero the score,
but the result lists it and the verdict says so. Categories with no answered indicator are left out of the weighted
average (weights renormalised over the categories present), never scored as zero.
"""
from __future__ import annotations

from . import config as C


def fatal_flaws() -> list[str]:
    return list(C.scoring()['gates'])


def time_factor(months: float, r: float) -> float:
    return (1 + r) ** (-months / 12)


def from_terms(G: float, sum_ws: float, p_stall: float, months: float, r: float) -> float:
    return 100 * G * sum_ws * (1 - p_stall) * time_factor(months, r)


def verdict(score: float, G: int, unknown: list[str]) -> str:
    if G == 0:
        return 'Fails a gate'
    label = next(v['label'] for v in C.scoring()['verdicts'] if score >= v['min'])
    return label + (' (gate unknown)' if unknown else '')


def atria_score(flags: dict, s: dict[str, float], w: dict[str, float],
                p_stall: float, months_to_power: float, r: float) -> dict:
    flaws = fatal_flaws()
    G = 0 if any(flags.get(f) is True for f in flaws) else 1
    unknown = [f for f in flaws if flags.get(f) is None]
    cats = [k for k in w if k in s and s[k] is not None]
    wsum = sum(w[k] for k in cats)
    suit = sum(w[k] * s[k] for k in cats) / wsum if wsum else 0.0
    tf = time_factor(months_to_power, r)
    score = 100 * G * suit * (1 - p_stall) * tf
    return {'score': round(score, 1), 'G': G, 'suitability': round(suit, 3),
            'risk_factor': round(1 - p_stall, 3), 'time_factor': round(tf, 3), 'p_stall': round(p_stall, 3),
            'months_to_power': months_to_power, 'r': r,
            'contributions': {k: round(w[k] * s[k] / wsum, 4) for k in cats} if wsum else {},
            'categories_missing': [k for k in w if k not in cats],
            'fatal_flaws': [f for f in flaws if flags.get(f) is True], 'unknown_gates': unknown,
            'verdict': verdict(score, G, unknown)}


def toy_example() -> dict:
    """The tab-4 toy example, computed from the numbers parsed out of the spreadsheet."""
    t = C.formula()['toy_example']
    val = from_terms(t['G'], t['sum_ws'], t['p_stall'], t['months'], t['r'])
    return dict(t, score=round(val, 1), time_factor=round(time_factor(t['months'], t['r']), 4))


def skeleton_example() -> dict:
    """The skeleton's own __main__ example (category scores and weights, three stall layers, 36 months): 38.3."""
    from .models.a_time_to_power import STUB
    from .models.b_stall import model_b_stall
    s = {'Technical': 0.75, 'Environmental': 0.60, 'Cultural': 0.70, 'Policy': 0.65, 'Security': 0.70}
    w = {'Technical': 0.35, 'Environmental': 0.20, 'Cultural': 0.15, 'Policy': 0.20, 'Security': 0.10}
    p = model_b_stall(p_local=0.08, p_state=0.05, p_commercial=0.03)
    return atria_score({f: False for f in fatal_flaws()}, s, w, p, STUB['mean'], r=0.15) | {'s': s, 'w': w}
