"""Model G: firm backup check, the GRIDTECH capacity-adequacy rules R0-R5 (GRIDTECH_Model_Explained.pdf, section 4.6).

Only on-site gas (turbine, engine, fuel cell) counts as firm: contracts arrive over the same wire as the grid and the battery
runs out of energy. For every month from opening, the rule is checked against the worst hour of that month over all 730 hours
(not a sample), using the packet's ERCOT-2023 hourly load and tiers scaled to the campus peak, the ten-month ramp, the site's
grid import steps and the stress events (import limit 60%, or 0 under Texas SB6).

    R0  gas >= max_h (D - flex - G_eff)                       always on
    R1  R0 with the largest gas unit out (N-1)
    R2  + grid lost: gas alone carries Tier 0
    R3  + grid lost: gas carries Tier 0 + Tier 1, less what can be relocated
    R4  + Texas SB6: imports 0 in stress hours, and gas >= 50% of peak
    R5  + grid lost: gas carries the whole campus

This checks a portfolio that is already sized; it does not size one (GRIDTECH's MILP does that)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ._faraday import FARADAY

METHOD = 'Backup rules R0-R5 on the worst hour of every month (ERCOT 2023 hourly load and stress events, scaled to the campus)'
LEVELS = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5']
LABELS = {'R0': 'Gas covers what the grid and flexible load cannot', 'R1': 'Same, with the largest gas unit broken (N-1)',
          'R2': 'Grid lost: gas alone carries protected load', 'R3': 'Grid lost: gas carries protected plus relocatable load',
          'R4': 'Texas SB6: no imports in stress events, gas at least half of peak', 'R5': 'Grid lost: gas carries the whole campus'}
SHORT = {'R0': 'basic cover', 'R1': 'N-1', 'R2': 'grid-loss (protected load)', 'R3': 'grid-loss (protected + relocatable)', 'R4': 'Texas SB6', 'R5': 'full islanding'}
GAS = ('gas_turbine', 'gas_engine', 'fuel_cell')
UNIT_MW = {'gas_turbine': 25.0, 'gas_engine': 5.0, 'fuel_cell': 5.0}
HOURS, MONTH_H, YEAR_H = 87_600, 730, 8_760


@lru_cache(maxsize=1)
def _packet():
    ref = pd.read_csv(FARADAY / 'data' / 'reference_hourly.csv', usecols=['load_mw', 'tier0_protected_mw', 'tier1_relocatable_mw',
                                                                         'tier2_interruptible_mw', 'stress_active', 'stress_import_fraction', 'timestamp_local'])
    opts = pd.read_csv(FARADAY / 'data' / 'resource_options.csv').set_index('option_id')
    resp = pd.read_json(FARADAY / 'data' / 'response_parameters.json', typ='series')
    return ref, {g: int(opts.loc[g, 'available_month']) for g in GAS}, float(resp['remote_capacity_mw']), float(resp['remote_energy_multiplier'])


def _hourly(peak: float, opening: int, schedule: list):
    ref, avail, remote, mult = _packet()
    idx = np.arange(HOURS) % YEAR_H
    m = np.arange(HOURS) // MONTH_H
    f = np.where(m < opening, 0.0, np.minimum(1.0, 1 / 3 + (m - opening) / 15))
    k = peak / 75.0 * f
    t0, t1, t2 = (ref[c].to_numpy()[idx] * k for c in ('tier0_protected_mw', 'tier1_relocatable_mw', 'tier2_interruptible_mw'))
    load = ref['load_mw'].to_numpy()[idx] * k
    stress = ref['stress_active'].to_numpy()[idx].astype(bool)
    frac = ref['stress_import_fraction'].to_numpy()[idx]
    g_ord = np.zeros(HOURS)
    for start, mw in sorted(schedule):
        g_ord[m >= int(start)] = float(mw)
    reloc = np.where(stress, np.minimum(t1, remote * peak / 75.0 / mult), 0.0)
    intr = np.where(stress, t2, 0.0)
    return dict(m=m, load=load, t0=t0, t1=t1, stress=stress, frac=frac, g_ord=g_ord, reloc=reloc, intr=intr, avail=avail,
                stamp=ref['timestamp_local'].to_numpy()[idx], reloc_max=remote * peak / 75.0 / mult)


def _firm(caps: dict, avail: dict, month: int) -> tuple[float, float]:
    on = {g: float(caps.get(g) or 0) for g in GAS if month >= avail[g]}
    firm = sum(on.values())
    largest = max([UNIT_MW[g] for g, mw in on.items() if mw > 0], default=0.0)
    return firm, largest


def check(caps: dict, peak: float, opening: int, schedule: list) -> dict:
    """Each level: the worst month (smallest margin), the MW it needs, the MW the portfolio has there, and the binding hour."""
    H = _hourly(peak, opening, schedule)
    m = H['m']
    live = m >= opening
    months = range(opening, HOURS // MONTH_H)
    firm_m = {mo: _firm(caps, H['avail'], mo) for mo in months}

    def g_eff(sb6: bool):
        return np.where(H['stress'], 0.0 if sb6 else H['g_ord'] * H['frac'], H['g_ord'])

    def worst(need: np.ndarray, n1: bool, rule: str) -> dict:
        best = None
        for mo in months:
            sel = np.where(m == mo)[0]
            i = sel[int(np.argmax(need[sel]))]
            firm, largest = firm_m[mo]
            have = firm - (largest if n1 else 0.0)
            margin = float(have - need[i])
            if best is None or margin < best['margin']:
                best = dict(rule=rule, month=mo, hour=int(i), needMw=round(float(need[i]), 1), haveMw=round(have, 1), margin=round(margin, 1),
                            firmMw=float(firm), largestMw=largest if n1 else None)
        return best

    out = {}
    for lv in LEVELS:
        n = LEVELS.index(lv)
        sb6 = n >= 4
        rules = [worst(H['load'] - H['reloc'] - H['intr'] - g_eff(sb6), n >= 1, 'gap')]   # R0 (N-1 from R1)
        if n >= 2:
            rules.append(worst(H['t0'], False, 'tier0'))
        if n >= 3:
            rules.append(worst(H['t0'] + H['t1'] - np.minimum(H['t1'], H['reloc_max']), False, 'tier01'))
        if n >= 4:
            rules.append(worst(np.where(live, 0.5 * peak, 0.0), False, 'sb6'))
        if n >= 5:
            rules.append(worst(H['load'], False, 'all'))
        b = min(rules, key=lambda r: r['margin'])
        i = b['hour']
        b.update(level=lv, label=LABELS[lv], ok=bool(b['margin'] >= -0.05), preGrid=bool(H['g_ord'][i] == 0), when=str(H['stamp'][i])[:16].replace('T', ' '), stress=bool(H['stress'][i]),
                 parts=dict(load=round(float(H['load'][i]), 1), reloc=round(float(H['reloc'][i]), 1), intr=round(float(H['intr'][i]), 1),
                            grid=round(float(g_eff(sb6)[i]), 1), gridOrd=round(float(H['g_ord'][i]), 1)))
        out[lv] = b
    highest = max((lv for lv in LEVELS if all(out[x]['ok'] for x in LEVELS[:LEVELS.index(lv) + 1])), key=LEVELS.index, default=None)
    return dict(levels=out, highestMet=highest, gasMw=sum(float(caps.get(g) or 0) for g in GAS), peakMw=peak, opening=opening,
                schedule=schedule, method=METHOD)


def from_analysis(a: dict | None) -> dict | None:
    if not a or a.get('status') != 'ok':
        return None
    return check(a['caps'], float(a['targetMW']), int(a['opening']), a['case']['grid']['schedule'])


def extra_gas(shortfall_mw: float) -> dict:
    """The cheapest way to close a shortfall in GRIDTECH's terms: 5 MW engine blocks (adding them does not change the largest unit)."""
    blocks = int(np.ceil(max(0.0, shortfall_mw) / 5.0 - 1e-9))
    mw = blocks * 5.0
    crf = 0.08 / (1 - 1.08 ** -20)                         # engine life 20 years, 8% real
    per_mw_yr = 1000 * (2400 * crf + 25.0)                 # CAPEX $2,400/kW, FOM $25/kW-yr
    return dict(mw=mw, blocks=blocks, costPerYearM=round(mw * per_mw_yr / 1e6, 2), basis='Gas engines in 5 MW blocks: $2,400/kW annualised at 8% over 20 years plus $25/kW-yr fixed O&M')
