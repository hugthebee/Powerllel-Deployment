"""Model F: bridge contract cash flows, breakeven price, expected value, and NPV / IRR distributions.

The cash-flow engine is atria-power-desk's underwriting (screening grade: no debt, tax or inflation), used unchanged.
The distributions re-run its cash flows for each model D draw of the full-grid month; with probability P_stall the
project dies before power-on and loses capex less salvage."""
from __future__ import annotations

import random

from ._faraday import mod
from .d_bridge_mc import _pct, draws

METHOD = 'Monthly cash flows of the bridge power contract, over 500 possible grid-arrival months'


def underwrite(analysis: dict, price=None, price_basis='mwh', discount=None, p_stall=None, probs=None, post_grid=None) -> dict:
    ui = analysis['uwInputs']
    U = mod('underwriting')
    return U.underwrite(ui['caps'], ui['onsite'], ui['opening'], ui['fullGrid'], ui['varCost'],
                        ui['death'] if p_stall is None else p_stall, price=price, price_basis=price_basis,
                        discount=discount, probs=probs, post_grid=post_grid)


def distribution(analysis: dict, arrival: dict, p_stall: float, price=None, price_basis='mwh', discount=None,
                 n: int = 500, seed: int = 7) -> dict:
    U = mod('underwriting')
    ui = analysis['uwInputs']
    cfg = U.CFG
    price = cfg['contract_usd_per_mwh_default'] if price is None else float(price)
    discount = cfg['discount_rate_default'] if discount is None else float(discount)
    opening, onsite, full = ui['opening'], ui['onsite'], ui['fullGrid']
    pre = [x for i, x in enumerate(onsite) if opening <= i < full]
    tail = pre[-1] if pre else 0.0
    rng = random.Random(seed + 1)
    npvs, irrs, dead = [], [], 0
    capex = None
    for t in draws(arrival, n, seed):
        fg = max(opening, int(round(t)))
        series = list(onsite[:min(full, fg)]) + [tail] * max(0, fg - full)
        flows, info = U.case_flows(ui['caps'], series, opening, fg, price, price_basis, ui['varCost'], cfg['salvage_share_of_capex'],
                                   cfg['post_grid_usd_per_kw_month'], discount)
        capex = info['capexM'] * 1e6
        if rng.random() < p_stall:
            dead += 1
            flows = [-capex] + [0.0] * max(1, opening - 1) + [cfg['salvage_share_of_capex'] * capex]
        npvs.append(U.npv(discount, flows) / 1e6)
        i = U.irr(flows)
        if i is not None:
            irrs.append(i)
    summ = lambda xs: {'p10': round(_pct(xs, .1), 2), 'p50': round(_pct(xs, .5), 2), 'p90': round(_pct(xs, .9), 2), 'mean': round(sum(xs) / len(xs), 2)} if xs else None   # noqa: E731
    hist = _hist(npvs)
    return {'n': n, 'price': price, 'priceBasis': price_basis, 'discountRate': discount, 'pStall': p_stall,
            'npvM': summ(npvs), 'irr': summ(irrs), 'probLoss': round(sum(x < 0 for x in npvs) / n, 3), 'diedDraws': dead,
            'expectedValueM': round(sum(npvs) / n, 2), 'histogram': hist, 'method': METHOD}


def _hist(xs, bins=20):
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [{'x0': lo, 'x1': hi, 'n': len(xs)}]
    w = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        counts[min(bins - 1, int((x - lo) / w))] += 1
    return [{'x0': round(lo + i * w, 2), 'x1': round(lo + (i + 1) * w, 2), 'n': c} for i, c in enumerate(counts)]


def cost_of_delay(analysis: dict, price=None, discount=None, p_stall=None) -> dict:
    """Change in the probability-weighted contract NPV when the grid slips one month (base case moved by +1)."""
    a = underwrite(analysis, price=price, discount=discount, p_stall=p_stall)
    b = underwrite(dict(analysis, uwInputs=dict(analysis['uwInputs'], fullGrid=analysis['uwInputs']['fullGrid'] + 1)),
                   price=price, discount=discount, p_stall=p_stall)
    return {'contractNpvPerMonthM': round(b['weightedNpvM'] - a['weightedNpvM'], 3),
            'note': 'Bridge investor view: a later grid lengthens the bridge contract. The developer view is the time factor in the score.'}
