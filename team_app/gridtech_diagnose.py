"""Why a GRIDTECH run has no feasible portfolio, and which nearby inputs have one.

The sizing problem in gridtech_model.LP can only be infeasible through its capacity rules. Everything else (the hourly
balance, the battery, Tier 1/2 response, offsite contracts) can always be met once onsite gas covers the hours the grid
and flexibility cannot. Those rules involve only the gas build-out (plus land and minimum commitments), so they are
rebuilt here exactly as LP._build writes them (same rows, same masks, same bounds) and solved as a tiny integer
problem in milliseconds:

    adequacy   every month from opening: gas online >= max over its hours of (demand - Tier 1/2 flexibility - grid import)
    n1         reliability R1+: the same with the largest gas unit out
    grid_n1    reliability R2+: gas alone carries the protected load if the grid connection is lost
    sb6        reliability R4: gas >= 50% of 75 MW (scaled) backup
    land       onsite equipment (gas + minimum commitments) fits on the site
    bounds     resource maxima, availability months, minimum commitments

The one rule this cannot decide is the carbon ceiling (it needs the hourly dispatch); results that still carry one are
marked as not fully checked.
"""
from __future__ import annotations
import contextlib
import copy
import io
import math

import numpy as np

import gridtech_model as G

HPM = G.HPM
GROUPS = ('adequacy', 'n1', 'grid_n1', 'sb6', 'land')
TOL = 1e-6


# ------------------------------------------------------------------ the capacity rules of LP._build, over gas block counts
class Capacity:
    def __init__(self, sc):
        self.sc = sc
        self.inp = I = G.Inputs(sc)
        rel = sc.rel
        self.blk = {g: I.blk[g] for g in G.GAS}
        maxc = {r: float((sc.max_caps or {}).get(r, I.opt.loc[r, 'max_mw'])) for r in G.RES}
        self.ub = {r: 0 if r in sc.forbid else math.floor(maxc[r] / I.blk[r] + 1e-9) for r in G.RES}
        self.lb = {r: math.ceil((sc.min_caps or {}).get(r, 0) / I.blk[r] - 1e-9) for r in G.RES}
        self.months = list(range(sc.start, 120))
        self.mask = {g: np.array([I.mask[g][m * HPM] for m in self.months]) for g in G.GAS}
        self.need, self.at = [], []
        for m in self.months:
            a, b = m * HPM, (m + 1) * HPM
            gap = I.load[a:b] - I.flex[a:b] - I.cap_eff[a:b]
            j = int(np.argmax(gap))
            self.need.append(float(gap[j]))
            self.at.append(a + j)
        self.need = np.array(self.need)
        self.grid_need = None
        if rel.grid_n1:
            gn = []
            for m in self.months:
                a, b = m * HPM, (m + 1) * HPM
                prot = {'tier0': I.t0, 'tier0+tier1': I.t0 + I.t1, 'all': I.load}[rel.c2_protect][a:b]
                if rel.c2_allow_relocation and rel.c2_protect != 'tier0':
                    prot = prot - np.minimum(I.t1[a:b], I.remote / 1.05)
                gn.append(float(np.max(prot - rel.grid_feed_survival * I.cap_ord[a:b])))
            self.grid_need = np.array(gn)
        self.sb6 = rel.sb6_backup_fraction * I.peak if rel.sb6_backup_fraction > 0 else None
        # staged: the SB6 backup row is written per model year with only the gas that may exist in that year
        self.sb6_years = []
        if self.sb6 is not None:
            years = range(sc.start // 12, 10) if sc.staged else [None]
            for yy in years:
                ok = [g for g in G.GAS if yy is None or I.mask[g][I.year == yy].sum() > 0]
                self.sb6_years.append((yy, ok))
        self.acres = {r: I.blk[r] * float(I.opt.loc[r, 'land_acres_per_mw']) for r in G.RES}
        self.fixed_land = sum(self.lb[r] * self.acres[r] for r in G.RES if r not in G.GAS)
        self.unit = dict(rel.unit_mw)
        self.optmax = {g: float(I.opt.loc[g, 'max_mw']) for g in G.GAS}

    def solve(self, groups=GROUPS, objective='mw'):
        """Tiny MIP over n_g (gas blocks), y_g (type built, for N-1) and L (largest unit). Returns (feasible, solution)."""
        import highspy
        rel = self.sc.rel
        gas = list(G.GAS)
        ng = len(gas)
        iy = {g: ng + i for i, g in enumerate(gas)}
        iL = 2 * ng
        nv = 2 * ng + 1
        lb = [float(self.lb[g]) for g in gas] + [0.0] * ng + [0.0]
        ub = [float(self.ub[g]) for g in gas] + [1.0] * ng + [float(max(self.unit.values()))]
        if 'n1' not in groups or not rel.onsite_n1:
            ub[iL] = 0.0
            for g in gas:
                ub[iy[g]] = 0.0
        cost = [self.blk[g] if objective == 'mw' else self.acres[g] for g in gas] + [0.0] * (ng + 1)
        rows = []                                                    # (cols, vals, lo, hi)
        for k, m in enumerate(self.months):
            gm = [self.blk[g] * self.mask[g][k] for g in gas]
            if 'adequacy' in groups and self.need[k] > 0:
                rows.append((list(range(ng)), gm, self.need[k], math.inf))
            if 'n1' in groups and rel.onsite_n1:
                rows.append((list(range(ng)) + [iL], gm + [-1.0], self.need[k], math.inf))
            if 'grid_n1' in groups and self.grid_need is not None and self.grid_need[k] > 0:
                rows.append((list(range(ng)), gm, self.grid_need[k], math.inf))
        if 'n1' in groups and rel.onsite_n1:
            for i, g in enumerate(gas):
                rows.append(([i, iy[g]], [self.blk[g], -self.optmax[g]], -math.inf, 0.0))
                rows.append(([iL, iy[g]], [1.0, -float(self.unit[g])], 0.0, math.inf))
        if 'sb6' in groups and self.sb6 is not None:
            for _, ok in self.sb6_years:
                rows.append(([gas.index(g) for g in ok], [self.blk[g] for g in ok], self.sb6, math.inf))
        if 'land' in groups:
            rows.append((list(range(ng)), [self.acres[g] for g in gas], -math.inf, self.inp.land - self.fixed_land))
        h = highspy.Highs()
        h.setOptionValue('output_flag', False)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = nv, len(rows)
        lp.col_cost_ = np.array(cost, float)
        lp.col_lower_ = np.array(lb, float)
        lp.col_upper_ = np.array(ub, float)
        lp.row_lower_ = np.array([r[2] for r in rows], float) if rows else np.zeros(0)
        lp.row_upper_ = np.array([r[3] for r in rows], float) if rows else np.zeros(0)
        starts, idx, val = [0], [], []
        cols = [[] for _ in range(nv)]
        for ri, (cs, vs, _, _) in enumerate(rows):
            for c, v in zip(cs, vs):
                if v != 0:
                    cols[c].append((ri, v))
        for c in cols:
            for ri, v in c:
                idx.append(ri)
                val.append(v)
            starts.append(len(idx))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.array(starts, np.int32)
        lp.a_matrix_.index_ = np.array(idx, np.int32)
        lp.a_matrix_.value_ = np.array(val, float)
        lp.integrality_ = [highspy.HighsVarType.kInteger] * nv
        h.passModel(lp)
        h.run()
        st = h.modelStatusToString(h.getModelStatus())
        if st != 'Optimal':
            return False, None
        x = np.array(h.getSolution().col_value)
        return True, {g: float(round(x[i]) * self.blk[g]) for i, g in enumerate(gas)}

    def feasible(self, groups=GROUPS):
        return self.solve(groups)[0]

    # numbers used in the explanations
    def max_gas(self, k):
        return sum(self.ub[g] * self.blk[g] * self.mask[g][k] for g in G.GAS)

    def best_n1(self, k):
        """Most gas left with the largest built unit out, over every choice of gas types to build."""
        best = -math.inf
        types = list(G.GAS)
        for bits in range(1, 1 << len(types)):
            s = [g for i, g in enumerate(types) if bits >> i & 1 and self.ub[g] > 0]
            if not s:
                continue
            best = max(best, sum(self.ub[g] * self.blk[g] * self.mask[g][k] for g in s) - max(self.unit[g] for g in s))
        return best


def structurally_feasible(sc):
    return Capacity(sc).feasible()


# ------------------------------------------------------------------ reasons
def _lab(m, cal):
    d = G.month_to_date(m, cal)
    return f'month {m}' + (f' ({d})' if cal else '')


def _gas_status(cap, k, m, cal):
    out = []
    for g in G.GAS:
        av = int(cap.inp.opt.loc[g, 'available_month'])
        name = G.BAR_LABELS[g].lower()
        if cap.ub[g] == 0:
            out.append(f'{name}: not allowed (maximum 0 MW)')
        elif cap.mask[g][k] == 0:
            out.append(f'{name}: not available until {_lab(av, cal)}')
        else:
            out.append(f'{name}: up to {cap.ub[g] * cap.blk[g]:g} MW')
    return '; '.join(out)


def reasons(cap, co2_cap=None, sizing_status=None):
    I, sc = cap.inp, cap.sc
    cal = I.cu.calendar_start
    out = []
    if not cap.feasible(('adequacy',)):
        short = [(k, m) for k, m in enumerate(cap.months) if cap.need[k] > cap.max_gas(k) + TOL]
        k, m = max(short, key=lambda km: cap.need[km[0]] - cap.max_gas(km[0])) if short else (0, cap.months[0])
        h = cap.at[k]
        load, flex, imp = float(I.load[h]), float(I.flex[h]), float(I.cap_eff[h])
        stress = bool(I.active[h])
        rng = f'{_lab(short[0][1], cal)} to {_lab(short[-1][1], cal)}' if len(short) > 1 else _lab(m, cal)
        grid_txt = (f'the grid can import {imp:g} MW' + (' (cut during a stress event)' if stress else '')) if imp > 0 else 'the grid is not connected yet' if I.cap_ord[h] == 0 else 'the grid import is cut to 0 MW in this stress event'
        out.append({'key': 'adequacy', 'rule': 'Power in every hour',
                    'title': f'Not enough onsite generation can be online in {len(short)} month{"s" if len(short) != 1 else ""} ({rng})',
                    'detail': f'Until the grid connection carries the whole campus, onsite gas has to cover the gap. The worst case is {_lab(m, cal)}: '
                              f'demand reaches {load:.1f} MW, {grid_txt}, and Tier 1/2 flexibility covers {flex:.1f} MW, which leaves {cap.need[k]:.1f} MW for onsite gas. '
                              f'At most {cap.max_gas(k):g} MW of gas can be online then ({_gas_status(cap, k, m, cal)}).',
                    'fixes': ['open later', 'connect more grid import earlier', 'allow more onsite gas, or make it available sooner', 'a smaller campus']})
    else:
        if sc.rel.onsite_n1 and not cap.feasible(('adequacy', 'n1')):
            short = [(k, m) for k, m in enumerate(cap.months) if cap.need[k] > cap.best_n1(k) + TOL]
            k, m = max(short, key=lambda km: cap.need[km[0]] - cap.best_n1(km[0])) if short else (0, cap.months[0])
            out.append({'key': 'n1', 'rule': 'Reliability: N-1 onsite',
                        'title': 'The onsite fleet cannot lose its largest unit and still cover demand',
                        'detail': f'Reliability {_rel_key(sc)} needs onsite gas to cover the gap even with its largest unit out '
                                  f'(units: gas turbine {cap.unit["gas_turbine"]:g} MW, gas engine {cap.unit["gas_engine"]:g} MW, fuel cell {cap.unit["fuel_cell"]:g} MW). '
                                  f'In {_lab(m, cal)} that gap is {cap.need[k]:.1f} MW, but the most gas left after losing one unit is {max(cap.best_n1(k), 0):.1f} MW.',
                        'fixes': ['a lower reliability level', 'more onsite gas', 'open later or connect the grid earlier']})
        if cap.grid_need is not None and not cap.feasible(('adequacy', 'grid_n1')):
            short = [(k, m) for k, m in enumerate(cap.months) if cap.grid_need[k] > cap.max_gas(k) + TOL]
            k, m = max(short, key=lambda km: cap.grid_need[km[0]] - cap.max_gas(km[0])) if short else (0, cap.months[0])
            prot = {'tier0': 'Tier 0', 'tier0+tier1': 'Tier 0 + Tier 1', 'all': 'the whole campus'}[sc.rel.c2_protect]
            out.append({'key': 'grid_n1', 'rule': 'Reliability: loss of the grid',
                        'title': f'Onsite gas cannot carry {prot} if the grid connection is lost',
                        'detail': f'Reliability {_rel_key(sc)} requires onsite gas alone to carry {prot} without the grid. In {_lab(m, cal)} that is {cap.grid_need[k]:.1f} MW, '
                                  f'but at most {cap.max_gas(k):g} MW of gas can be online ({_gas_status(cap, k, m, cal)}).',
                        'fixes': ['a lower reliability level', 'more onsite gas, or available sooner', 'a smaller campus']})
        if cap.sb6 is not None and not cap.feasible(('adequacy', 'sb6')):
            most = min(sum(cap.ub[g] * cap.blk[g] for g in ok) for _, ok in cap.sb6_years)
            out.append({'key': 'sb6', 'rule': 'Reliability: Texas SB6 backup',
                        'title': 'Not enough onsite backup for the SB6 rule',
                        'detail': f'Reliability {_rel_key(sc)} requires at least {cap.sb6:.1f} MW of onsite gas backup, but at most {most:g} MW of gas can be built.',
                        'fixes': ['a lower reliability level', 'more onsite gas']})
        if not cap.feasible(('adequacy', 'land')):
            ok, sol = cap.solve(('adequacy',), objective='land')
            need = sum(sol[g] / cap.blk[g] * cap.acres[g] for g in G.GAS) if ok else float('nan')
            mins_txt = f' plus {cap.fixed_land:.2f} acres for the minimum commitments' if cap.fixed_land > 0 else ''
            out.append({'key': 'land', 'rule': 'Land on the site',
                        'title': 'The onsite equipment does not fit on the site',
                        'detail': f'Covering the gap needs at least {need:.2f} acres of onsite gas{mins_txt}, but the site has {I.land:g} acres.',
                        'fixes': ['remove or reduce minimum commitments' if cap.fixed_land > 0 else 'connect more grid import earlier', 'open later', 'a smaller campus']})
        if not out and not cap.feasible():
            out.append({'key': 'combined', 'rule': 'Several rules together',
                        'title': 'Each rule can be met alone, but not all at once',
                        'detail': 'The reliability rules, the land limit and the resource maxima each work on their own, but together they need more onsite gas than can be built or fitted.',
                        'fixes': ['a lower reliability level', 'open later or connect the grid earlier', 'more onsite gas']})
    if not out:
        if co2_cap is not None:
            proven = sizing_status == 'Infeasible'
            out.append({'key': 'co2', 'rule': 'Carbon ceiling',
                        'title': f'The carbon ceiling of {co2_cap:g} kg CO2/MWh is {"too tight" if proven else "most likely too tight"}',
                        'detail': ('Every capacity rule can be met, so the ceiling is what rules out every portfolio' if proven else
                                   f'Every capacity rule can be met, so the ceiling is the only rule left that can block a portfolio; the solver found none before stopping ("{sizing_status}")')
                                  + ': the gas the campus needs until the grid carries it, plus grid power, emit more on average than the ceiling allows.',
                        'fixes': ['a higher carbon ceiling, or none', 'connect more grid import earlier (less gas)', 'open later']})
        else:
            out.append({'key': 'solver', 'rule': 'Solver',
                        'title': 'The solver stopped without a portfolio',
                        'detail': f'Every capacity rule can be met, so a portfolio exists, but the solver ended with status "{sizing_status}" before finding it. '
                                  'Running again usually finds it; upfront capacity solves faster than year by year.',
                        'fixes': ['run again', 'choose upfront capacity']})
    return out


def _rel_key(sc):
    for k, r in G.PRESETS.items():
        if r == sc.rel:
            return k
    return 'this level'


# ------------------------------------------------------------------ alternatives: nearby inputs that pass every capacity rule
def alternatives(payload, build, co2_cause=False, limit=6):
    """payload = the page's answers; build(payload) -> (Scenario, info) (the server's validator).
    Each alternative changes one kind of input, as little as possible, and passes every capacity rule."""
    co2 = payload.get('co2Cap') not in (None, '')
    out = []

    def check(p):
        try:
            sc, _ = build(p)
            with contextlib.redirect_stdout(io.StringIO()):          # the model prints notes for unusual schedules
                cap = Capacity(sc)
        except ValueError:
            return None
        ok, sol = cap.solve()
        return sol if ok else None

    def add(key, title, changes, p, sol, note=''):
        verified = not co2 or key == 'no_co2'
        gas = ', '.join(f'{G.BAR_LABELS[g].lower()} {mw:g} MW' for g, mw in sol.items() if mw > 0) if sol else ''
        out.append({'key': key, 'title': title, 'changes': changes, 'inputs': p, 'verified': verified,
                    'note': note or ('Passes every capacity rule' + (f'; needs at least {gas} of onsite gas' if gas else '; needs no onsite gas')
                                     + ('. The carbon ceiling still needs a run to confirm.' if not verified else '. A run finds the cheapest portfolio.'))})

    start = int(float(payload['start']))
    if co2_cause:
        p = copy.deepcopy(payload)
        p['co2Cap'] = None
        sol = check(p)
        if sol is not None:
            add('no_co2', 'Drop the carbon ceiling', [f'Carbon ceiling: {payload["co2Cap"]} kg/MWh → none'], p, sol,
                'Every capacity rule passes without the ceiling, so this has a feasible portfolio. Its carbon intensity shows how far the ceiling was from reachable.')
        return out                                                 # capacity is not the problem; other changes would not help for sure

    # 1. later opening
    for s in range(start + 1, 110):
        p = copy.deepcopy(payload)
        p['start'] = str(s)
        sol = check(p)
        if sol is not None:
            add('later_opening', f'Open in month {s} instead of {start}', [f'Opening month: {start} → {s} (full demand month {s + 10})'], p, sol)
            break
    # 2. lower reliability
    order = list(G.PRESETS)
    cur = str(payload.get('reliability', 'R0')).upper()
    if cur in order:
        for k in reversed(order[:order.index(cur)]):
            p = copy.deepcopy(payload)
            p['reliability'] = k
            sol = check(p)
            if sol is not None:
                add('lower_reliability', f'Use reliability {k} instead of {cur}', [f'Reliability: {cur} → {k} ({G.PRESET_TEXT[k]})'], p, sol)
                break
    # 3. grid interconnection: the least generous full-import schedule that works
    peak = float(payload['peakMw'])
    best = None
    for m in range(24, 17, -1):                                    # latest month first = the smallest change
        p = copy.deepcopy(payload)
        p['interconnection'] = {'mode': 'month', 'month': m}
        sol = check(p)
        if sol is not None:
            best = (m, p, sol)
            break
    if best and payload.get('interconnection') != best[1]['interconnection']:
        m, p, sol = best
        add('grid', f'Full grid import ({peak:g} MW) from month {m}', [f'Interconnection: {_ic_text(payload)} → full {peak:g} MW from month {m}'], p, sol)
    # 4. onsite gas: available from opening, then larger maxima
    gas_late = [g for g in G.GAS if _res_val(payload, g, 'available_month') > start]
    if gas_late:
        p = copy.deepcopy(payload)
        for g in gas_late:
            p.setdefault('resources', {}).setdefault(g, {})['available_month'] = str(start)
        sol = check(p)
        if sol is not None:
            add('gas_sooner', 'Onsite gas available from the opening month',
                [f'{G.BAR_LABELS[g]} available from month {_res_val(payload, g, "available_month"):g} → {start}' for g in gas_late], p, sol,
                'Passes every capacity rule if the equipment can be delivered by the opening month (a lead-time question for the developer).'
                + (' The carbon ceiling still needs a run to confirm.' if co2 else ''))
    for mult in (1.5, 2, 3, 4):
        p = copy.deepcopy(payload)
        ch = []
        for g in G.GAS:
            row = G.OPT.loc[g]
            cur_max = _res_val(payload, g, 'max_mw', peak)
            new = max(row.block_mw, math.floor(cur_max * mult / row.block_mw) * row.block_mw)
            if new > cur_max:
                p.setdefault('resources', {}).setdefault(g, {})['max_mw'] = str(new)
                ch.append(f'{G.BAR_LABELS[g]} maximum {cur_max:g} → {new:g} MW')
        sol = check(p)
        if sol is not None:
            add('more_gas', f'Allow {mult:g}× more onsite gas', ch, p, sol)
            break
    # 5. no minimum commitments
    if any(str(v).strip() not in ('', '0') for v in (payload.get('minCaps') or {}).values()):
        p = copy.deepcopy(payload)
        p['minCaps'] = {}
        sol = check(p)
        if sol is not None:
            add('no_mins', 'Remove the minimum commitments', ['Minimum commitments: removed'], p, sol)
    # 6. a smaller campus (grid schedule kept in MW)
    if payload.get('interconnection', {}).get('mode') != 'packet':
        lo, hi, found = 5, int(peak) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            p = copy.deepcopy(payload)
            p['peakMw'] = str(mid)
            if payload['interconnection'].get('mode') == 'month':          # 'one month' means full import = the peak; keep the MW instead
                p['interconnection'] = {'mode': 'steps', 'steps': [{'month': payload['interconnection']['month'], 'mw': str(peak)}]}
            sol = check(p)
            if sol is not None:
                found, lo = (mid, p, sol), mid + 1
            else:
                hi = mid - 1
        if found:
            mid, p, sol = found
            add('smaller', f'A {mid:g} MW campus instead of {peak:g} MW', [f'Peak load: {peak:g} → {mid:g} MW (grid import kept in MW)'], p, sol)
    return out[:limit]


def _res_val(payload, rid, col, peak=None):
    v = (payload.get('resources') or {}).get(rid, {}).get(col)
    if v not in (None, ''):
        return float(v)
    if col == 'max_mw':
        row = G.OPT.loc[rid]
        return float(max(row.block_mw, np.floor(row.max_mw * (peak or 75) / 75 / row.block_mw + 1e-9) * row.block_mw))
    return float(G.OPT.loc[rid, col])


def _ic_text(payload):
    ic = payload.get('interconnection') or {}
    if ic.get('mode') == 'packet':
        return 'packet schedule'
    if ic.get('mode') == 'month':
        return f'full import from month {ic["month"]}'
    return ', '.join(f'{s["mw"]} MW from month {s["month"]}' for s in ic.get('steps', []))


def diagnose(payload, build, sizing_status=None):
    """Reasons + alternatives for an infeasible run."""
    sc, info = build(payload)
    with contextlib.redirect_stdout(io.StringIO()):
        cap = Capacity(sc)
    rs = reasons(cap, info.get('co2'), sizing_status)
    co2_cause = rs and rs[0]['key'] == 'co2'
    alts = [] if rs and rs[0]['key'] == 'solver' else alternatives(payload, build, co2_cause=bool(co2_cause))
    return {'reasons': rs, 'alternatives': alts,
            'method': 'Reasons and alternatives come from the model\'s capacity rules (power in every hour, N-1, loss of the grid, SB6 backup, land, '
                      'resource maxima and dates), rebuilt exactly as the optimiser writes them and solved in milliseconds.'
                      + (' The carbon ceiling cannot be checked this way; alternatives that keep it are marked "needs a run".' if info.get('co2') is not None else '')}
