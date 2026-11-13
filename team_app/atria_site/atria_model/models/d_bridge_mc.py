"""Model D: bridge power need, mix, cost and CO2: the Faraday hourly dispatch plus a Monte Carlo over grid arrival.

Faraday (atria-power-desk, unchanged) sizes the on-site mix and dispatches it hour by hour for the site's grid plan.
The Monte Carlo draws the full-grid month from model A (500 draws by default) and carries the on-site output forward
at its last pre-grid level when the grid is late (the same rule the underwriting uses). Re-running the dispatch per
draw is out of scope for a screen; price and weather shapes stay ERCOT 2023."""
from __future__ import annotations

import math
import random

from ._faraday import cached_analysis, mod

METHOD = 'Hourly run of the recommended on-site plant, over 500 possible grid-arrival months'
N_DRAWS = 500


def run(site_id: str, inputs: dict | None = None) -> dict | None:
    """A live Faraday run (minutes on first call per input set; cached in memory and on disk for the default inputs)."""
    return mod('sites').analyze(site_id, inputs or {})


def load(site_id: str) -> dict | None:
    return cached_analysis(site_id)


def draws(arrival: dict, n: int = N_DRAWS, seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    mu, s = math.log(arrival['p50']), arrival['sigma']
    return [math.exp(rng.gauss(mu, s)) for _ in range(n)]


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def monte_carlo(analysis: dict, arrival: dict, n: int = N_DRAWS, seed: int = 7) -> dict:
    """Distribution of bridge MW-years and on-site fuel + variable cost over grid-arrival draws."""
    ui = analysis['uwInputs']
    opening, onsite, var = ui['opening'], ui['onsite'], ui['varCost']
    mw = analysis['onsiteMw']
    pre = [x for i, x in enumerate(onsite) if opening <= i < ui['fullGrid']]
    tail = pre[-1] if pre else 0.0
    ts = draws(arrival, n, seed)
    mwy, cost = [], []
    for t in ts:
        end = max(opening, int(round(t)))
        months = [onsite[m] if m < len(onsite) and m < ui['fullGrid'] else tail for m in range(opening, end)]
        mwy.append(mw * (end - opening) / 12)
        cost.append(sum(x * 730 * var for x in months) / 1e6)
    summ = lambda xs: {'p10': round(_pct(xs, .1), 1), 'p50': round(_pct(xs, .5), 1), 'p90': round(_pct(xs, .9), 1), 'mean': round(sum(xs) / len(xs), 1)}   # noqa: E731
    return {'n': n, 'seed': seed, 'gridMonth': summ(ts), 'bridgeMwYears': summ(mwy), 'onsiteCostM': summ(cost), 'method': METHOD}


def summary(analysis: dict | None) -> dict | None:
    """What the Technical page shows from Faraday: mix, KPIs and the monthly power-over-time series."""
    if not analysis or analysis.get('status') != 'ok':
        return None
    rec = (analysis.get('model') or {}).get('recommendation') or {}
    return {'targetMW': analysis['targetMW'], 'opening': analysis['opening'], 'fullDemand': rec.get('fullDemand'),
            'caps': analysis['caps'], 'onsiteMw': analysis['onsiteMw'], 'yearsNeeded': analysis['yearsNeeded'],
            'bridgeMwYears': analysis['bridgeMwYears'], 'varCostUsdPerMwh': analysis['varCostUsdPerMwh'],
            'detail': rec.get('detail'), 'kpis': rec.get('kpis'), 'monthly': analysis.get('monthly')}
