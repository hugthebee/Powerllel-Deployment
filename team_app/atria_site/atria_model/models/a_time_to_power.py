"""Model A: months until the grid supplies the full load, as a distribution (P10 / P50 / P90, mean).

Today this is a heuristic (data map: "Partial (heuristic today)"): the P50 is atria-power-desk's staged grid plan
(substation voltage -> headroom, ISO large-load study time, queue time), and the spread is a lognormal expert prior
(P90 = 1.5 x P50). The fitted survival model (Weibull AFT or Bayesian hierarchical) replaces `arrival()` once past
project timelines arrive (customer input "Calibration")."""
from __future__ import annotations

import math

STUB = {'p10': 24, 'p50': 34, 'p90': 52, 'mean': 36}   # the skeleton's placeholder, used only by its toy example
P90_OVER_P50 = 1.5
Z90 = 1.2815515655446004
METHOD = 'Grid plan from substation voltage, distance and study time, with an expert range around it (late case = 1.5 × median)'


def _sigma(ratio: float = P90_OVER_P50) -> float:
    return math.log(ratio) / Z90


def arrival(p50_months: float, ratio: float = P90_OVER_P50) -> dict:
    """Distribution of the full-grid month from its median."""
    s = _sigma(ratio)
    return {'p10': round(p50_months * math.exp(-Z90 * s), 1), 'p50': round(float(p50_months), 1),
            'p90': round(p50_months * math.exp(Z90 * s), 1), 'mean': round(p50_months * math.exp(s * s / 2), 1),
            'sigma': round(s, 4), 'method': METHOD}


def cdf(month: float, p50_months: float, ratio: float = P90_OVER_P50) -> float:
    """P(grid has arrived by `month`)."""
    if month <= 0:
        return 0.0
    s = _sigma(ratio)
    return 0.5 * (1 + math.erf(math.log(month / p50_months) / (s * math.sqrt(2))))


def curve(p50_months: float, horizon: int = 120, step: int = 3) -> list[dict]:
    return [{'month': m, 'p': round(cdf(m, p50_months), 4)} for m in range(0, horizon + 1, step)]


def expected_months(grid: dict, bridge_full_month: float | None = None, bridge_allowed: bool = True) -> dict:
    """E[T_power] for the score: with bridge power allowed and sized by Faraday, the month on-site + grid first cover the
    full load; otherwise the mean grid arrival."""
    if bridge_allowed and bridge_full_month is not None:
        return {'months': float(bridge_full_month), 'basis': 'bridge', 'note': 'The month on-site bridge power plus the grid first cover the full load'}
    return {'months': float(grid['mean']), 'basis': 'grid', 'note': 'The average expected month the grid supplies the full load'}


def from_case(case: dict) -> dict:
    """Model A from atria-power-desk's derived case for a site (grid plan)."""
    g = case['grid']
    out = arrival(g['fullGridMonth'])
    out.update(curve=curve(g['fullGridMonth']), schedule=g.get('schedule'), substation=g.get('substation'), iso=g.get('iso'),
               first_power_month=next((m for m, mw in g.get('schedule') or [] if mw > 0), None))
    return out
