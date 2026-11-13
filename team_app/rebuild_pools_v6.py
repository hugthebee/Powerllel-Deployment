"""Rebuild the financial agent's portfolio pools on packet v6.0.1 (the optimiser's packet), so both agents use the same assumptions.

    .venv/bin/python team_app/rebuild_pools_v6.py [--sample N]

Before running, the repository's data/ and starter/ must hold packet v6.0.1 (the v5.5.1 copies are in backup_v5.5.1/).
  phase A  re-score every portfolio of the old pools (pool3_base.json, pool3_no_t2.json) with the alternative evaluator on v6 prices and rules
  phase B  new searches per opening month 14-26 (cheapest, cleanest, and cheapest under 8 emission caps between them), as build_pareto.py,
           so the pool also holds the portfolios that are best under v6 (wind $45/MWh, battery up to 53 MW at $1,220/kW)
Writes pool3_base.json and pool3_no_t2.json (only portfolios that serve all demand and break no rule).
"""
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alt_adapter as A  # noqa: E402
import engine as E  # noqa: E402

OLD = HERE.parent / 'backup_v5.5.1' / 'team_app'
POLICIES = {'base': True, 'no_t2': False}           # name -> Tier 2 interruption allowed
STARTS = list(range(14, 27))
N_EPS = 8


def rescore(args):
    pol, start, caps = args
    return pol, A.evaluate_entry(start, caps, 0, True, POLICIES[pol], 0.0)


def search(args):
    pol, start, mode, eps = args
    intr = POLICIES[pol]
    ev = lambda caps, cpr: A.evaluate_entry(start, caps, 0, True, intr, 0.0, allow_infeasible=True)
    rec = []
    if mode == 'cost':
        E.search(start, 0.0, record=rec, evaluate=ev, max_iter=60)
    elif mode == 'carbon':
        E.search(start, 3000.0, record=rec, evaluate=ev, max_iter=60)
    else:
        E.search(start, 0.0, max_intensity=eps, record=rec, evaluate=ev, max_iter=60)
    return pol, start, mode, [e for e in rec if not e.get('unserved_mwh')]


def main():
    sample = int(sys.argv[sys.argv.index('--sample') + 1]) if '--sample' in sys.argv else None
    t0 = time.time()
    pools = {p: {} for p in POLICIES}
    jobs = []
    for pol in POLICIES:
        old = json.loads((OLD / f'pool3_{pol}.json').read_text())
        old = [e for e in old if 14 <= e['start'] <= 26]
        jobs += [(pol, e['start'], dict(zip(E.KEYS, e['caps']))) for e in old]
    if sample:
        jobs = jobs[::max(1, len(jobs) // sample)][:sample]
    ctx = get_context('fork')
    with ctx.Pool(8) as p:
        dropped = 0
        for i, (pol, e) in enumerate(p.imap_unordered(rescore, jobs, chunksize=32)):
            if e:
                pools[pol][(e['start'], tuple(e['caps']))] = e
            else:
                dropped += 1
            if i % 4000 == 0:
                print(f'phase A {i}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
        print(f'phase A done: {sum(len(v) for v in pools.values())} kept, {dropped} no longer feasible, {time.time() - t0:.0f}s', flush=True)
        if sample:
            return
        anchors = {}
        jobs1 = [(pol, s, m, None) for pol in POLICIES for s in STARTS for m in ('cost', 'carbon')]
        for pol, s, m, rec in p.imap_unordered(search, jobs1):
            for e in rec:
                pools[pol][(e['start'], tuple(e['caps']))] = e
            if rec:
                best = min(rec, key=(lambda e: e['total_cost']) if m == 'cost' else (lambda e: e['intensity']))
                anchors.setdefault(pol, {}).setdefault(s, {})[m] = best['intensity']
        print(f'phase B1 done {time.time() - t0:.0f}s', flush=True)
        jobs2 = []
        for pol in POLICIES:
            for s in STARTS:
                a = anchors.get(pol, {}).get(s, {})
                if 'carbon' in a and 'cost' in a and a['cost'] - a['carbon'] > 1.0:
                    jobs2 += [(pol, s, 'eps', float(x)) for x in np.linspace(a['carbon'], a['cost'], N_EPS + 2)[1:-1]]
        for i, (pol, s, m, rec) in enumerate(p.imap_unordered(search, jobs2)):
            for e in rec:
                pools[pol][(e['start'], tuple(e['caps']))] = e
            if i % 40 == 0:
                print(f'phase B2 {i}/{len(jobs2)} {time.time() - t0:.0f}s', flush=True)
    for pol, d in pools.items():
        (HERE / f'pool3_{pol}.json').write_text(json.dumps(list(d.values())))
        print(pol, len(d), 'portfolios written', flush=True)
    print(f'done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
