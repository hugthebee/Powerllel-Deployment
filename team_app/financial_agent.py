"""Financial agent for a Powerllel thesis: the Faraday pool method (GitHub hugthebee/Faraday-GridTech-Hackathon-, faraday_service.py).

    python team_app/financial_agent.py < thesis_plan.json > answers.json

Method: about 21,600 pre-scored portfolios of the packet case (openings month 14-26, so full demand month 24-36), each with its
10-year decision cost, emissions and energy mix from the hourly dispatch; a second pool of 18,700 portfolios that never interrupt
Tier 2. Hard limits filter the pool, the GridTech_Inputs.xlsx customer weights (speed / cost / carbon) rank what is left.
It answers the thesis questions by searching the pools, and gives the full Faraday recommendation for the thesis.
Runs in its own process so it cannot disturb the optimiser running in the server.
Scope: the 75 MW packet campus with the packet's ERCOT stress events and grid schedule, on packet v6.0.1 (the optimiser's packet).
The pools were rebuilt on v6.0.1 by rebuild_pools_v6.py; the v5.5.1 pools are in backup_v5.5.1/.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LABEL = {'utility_solar': 'solar', 'wind': 'wind', 'battery_4h': 'battery', 'gas_turbine': 'turbine', 'gas_engine': 'engines', 'fuel_cell': 'fuel cell', 'nuclear': 'nuclear'}
CAPS = list(LABEL)
FIRST, LAST = 14, 26


def row_view(r):
    if r is None:
        return None
    caps = {k: float(r[k]) for k in CAPS if float(r[k]) > 0}
    served = float(r['served_mwh']) or 1.0
    return {'opening': int(r['start']), 'fullDemand': int(r['full_month']), 'costM': float(r['total_cost']) / 1e6, 'kgPerMwh': float(r['intensity']),
            'cleanSharePct': float(r['clean_share']) * 100, 'merchantSharePct': float(r['merchant_mwh']) / served * 100, 'merchantGwh': float(r['merchant_mwh']) / 1e3,
            'interruptedMwh': float(r['interrupted_mwh']), 'relocatedMwh': float(r['relocated_mwh']), 'dollarsPerMwh': float(r['cost_per_mwh']),
            'caps': caps, 'detail': ' + '.join(f'{LABEL[k]} {v:g} MW' for k, v in caps.items()) or 'grid only'}


def cheapest(df):
    return None if df is None or df.empty else df.loc[df.total_cost.idxmin()]


def opening_for(full):
    """Latest opening month whose full demand (opening + 10) is no later than `full`."""
    return int(full) - 10


def answer(q, base, no_t2):
    k = q['key']
    if k == 'speed':
        t, c = opening_for(q['fullTarget']), opening_for(q['fullCompare'])
        a, b = cheapest(base[base.start <= t]), cheapest(base[base.start <= c])
        if a is None:
            return {'status': 'none', 'text': f"No portfolio in the pool reaches full demand by month {q['fullTarget']} (earliest is month {FIRST + 10})."}
        out = {'status': 'ok', 'target': row_view(a), 'compare': row_view(b)}
        if b is not None:
            d = (a.total_cost - b.total_cost) / 1e6
            out['extraM'] = d; out['extraPct'] = d / (b.total_cost / 1e6) * 100
            out['text'] = (f"Full demand by month {q['fullTarget']}: {row_view(a)['detail']}, ${a.total_cost / 1e6:,.1f}M. By month {q['fullCompare']}: "
                           f"{row_view(b)['detail']}, ${b.total_cost / 1e6:,.1f}M. Reaching it {q['fullCompare'] - q['fullTarget']} months sooner costs ${d:,.1f}M more ({out['extraPct']:.1f}%).")
        return out
    if k == 'cost':
        a = cheapest(base[base.start <= opening_for(q['fullLatest'])])
        if a is None:
            return {'status': 'none', 'text': 'No portfolio in the pool.'}
        v = row_view(a)
        return {'status': 'ok', 'best': v, 'text': f"Cheapest with full demand by month {q['fullLatest']}: {v['detail']}, ${v['costM']:,.1f}M; opens month {v['opening']}, "
                                                   f"full demand month {v['fullDemand']}, {v['kgPerMwh']:.0f} kg CO2/MWh, {v['merchantSharePct']:.0f}% of energy from the merchant grid."}
    if k == 'environment':
        ok = base[base.intensity < q['co2Max']]
        if ok.empty:
            return {'status': 'none', 'text': f"No portfolio in the pool averages below {q['co2Max']:g} kg CO2/MWh."}
        s = int(ok.start.min())
        a = cheapest(ok[ok.start == s])
        v = row_view(a)
        return {'status': 'ok', 'best': v, 'datesTested': sorted(int(x) for x in base.start.unique()),
                'text': f"Earliest full demand below {q['co2Max']:g} kg CO2/MWh: month {v['fullDemand']} (opening {v['opening']}), {v['detail']}, ${v['costM']:,.1f}M, {v['kgPerMwh']:.0f} kg/MWh."}
    if k == 'continuity':
        s = opening_for(q['fullTarget'])
        a, b = cheapest(no_t2[no_t2.start <= s]) if no_t2 is not None else None, cheapest(base[base.start <= s])
        if a is None or b is None:
            return {'status': 'none', 'text': 'No portfolio without Tier 2 interruption in the pool for that date.'}
        va, vb = row_view(a), row_view(b)
        d = va['costM'] - vb['costM']
        added = {LABEL[r]: va['caps'].get(r, 0) - vb['caps'].get(r, 0) for r in CAPS if abs(va['caps'].get(r, 0) - vb['caps'].get(r, 0)) > 1e-9}
        return {'status': 'ok', 'noTier2': va, 'withTier2': vb, 'extraM': d, 'changes': added,
                'text': f"Without any Tier 2 interruption: {va['detail']}, ${va['costM']:,.1f}M, against ${vb['costM']:,.1f}M when interruption is allowed ({d:+,.1f}M)."}
    if k == 'match':                              # the same opening (and emissions limit) as an optimiser run: the pool's cheapest
        o = int(q['opening'])
        if not FIRST <= o <= LAST:
            return {'status': 'not-covered', 'text': f'Not covered: the pools hold openings from month {FIRST} to {LAST}; this run opens in month {o}.'}
        pool = base[base.start == o]
        if q.get('co2Max'):
            pool = pool[pool.intensity <= q['co2Max']]
        a = cheapest(pool)
        if a is None:
            return {'status': 'none', 'text': f'No portfolio in the pool opens in month {o}' + (f" below {q['co2Max']:g} kg CO2/MWh." if q.get('co2Max') else '.')}
        v = row_view(a)
        return {'status': 'ok', 'best': v, 'text': f"Cheapest pool portfolio opening in month {o}: {v['detail']}, ${v['costM']:,.1f}M, {v['kgPerMwh']:.0f} kg CO2/MWh."}
    if k == 'slip':
        return {'status': 'not-covered', 'text': "Not covered: the pools were built on the packet's grid schedule, so a slipped connection needs the optimiser (or a new pool)."}
    if k == 'premium':
        t, c = opening_for(q['fullTarget']), opening_for(q['fullCompare'])
        a, b = cheapest(base[base.start <= t]), cheapest(base[base.start <= c])
        if a is None or b is None:
            return {'status': 'none', 'text': 'No portfolio for one of the dates.'}
        cap = b.total_cost * (1 + q['premiumPct'] / 100)
        inside = base[(base.start <= t) & (base.total_cost <= cap)]
        green = None if inside.empty else inside.loc[inside.intensity.idxmin()]
        prem = (a.total_cost / b.total_cost - 1) * 100
        out = {'status': 'ok', 'target': row_view(a), 'compare': row_view(b), 'premiumPct': prem, 'feasible': bool(not inside.empty), 'cleanestInside': row_view(green)}
        out['text'] = (f"The cheapest month-{q['fullTarget']} portfolio costs {prem:.1f}% more than the month-{q['fullCompare']} one: "
                       + (f"within your {q['premiumPct']:g}%. The cleanest inside the premium emits {green.intensity:.0f} kg/MWh against {b.intensity:.0f} for the month-{q['fullCompare']} portfolio."
                          if green is not None else f"above your {q['premiumPct']:g}%, so no portfolio fits the premium."))
        return out
    if k == 'staging':
        return {'status': 'not-covered', 'text': 'Not covered: every portfolio in the pools is built upfront; staging needs the optimiser.'}
    return {'status': 'not-covered', 'text': 'Unknown question.'}


def main():
    plan = json.load(sys.stdin)
    if plan.get('only') == 'recommend':           # re-rank for new weights (a template chosen on the results page)
        plan['questions'] = []
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        import faraday_service as S
        base, no_t2 = S._pool(False), S._pool(True)
        excl = [k for k in (plan.get('excluded') or []) if k in CAPS]     # resources switched off: drop every portfolio that uses them
        if excl:
            base = base[(base[excl] == 0).all(axis=1)].reset_index(drop=True)
            no_t2 = no_t2[(no_t2[excl] == 0).all(axis=1)].reset_index(drop=True) if no_t2 is not None else None
    out = {'method': 'Faraday pool method: pre-scored portfolios, customer limits as filters, GridTech_Inputs.xlsx weights to rank',
           'poolSize': int(len(base)), 'poolNoTier2': int(len(no_t2)) if no_t2 is not None else 0, 'openings': [FIRST, LAST], 'answers': {}}
    for q in plan.get('questions', []):
        try:
            out['answers'][q['key']] = answer(q, base, no_t2)
        except Exception as e:                      # one question failing must not lose the others
            out['answers'][q['key']] = {'status': 'error', 'text': f'{type(e).__name__}: {e}'}
    rec = plan.get('recommend') or {}
    try:
        with redirect_stdout(io.StringIO()):
            r = S.recommend(rec)
        keep = ('status', 'inputs', 'notes', 'search', 'recommendation', 'options', 'portfolioTable', 'costBreakdown', 'tradeoffs', 'powerOverTime', 'stress', 'reliability')
        out['recommendation'] = {k: r.get(k) for k in keep if k in r}
        if r.get('exportId') and plan.get('sensitivity', True):
            with redirect_stdout(io.StringIO()):       # bear / base / bull prices: every option re-dispatched under each scenario
                out['recommendation']['sensitivity'] = S.sensitivity(r['exportId'])
    except Exception as e:
        out['recommendation'] = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
    json.dump(out, sys.stdout, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


if __name__ == '__main__':
    main()
