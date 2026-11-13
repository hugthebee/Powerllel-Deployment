"""Powerllel thesis: one customer thesis, answered by four agents running in the background.

    optimisation agent   the GRIDTECH optimiser (gridtech_model.py), one run per scenario the questions need, queued one at a time
    financial agent      the Faraday pool method (financial_agent.py, run in its own process): instant answers from pre-scored portfolios
    site sourcing agent  proposed sites, parcels and site facts    } Atria Site Decisions, vendored in team_app/atria_site and run in-process
    moratorium agent     moratorium, local-risk and policy screening } (no server on port 8780 needed; same numbers as that server)

API (wired in gridtech_server.py)
    POST /api/thesis          the thesis -> {id} (starts the agents) | 400 {error, errors}
    GET  /api/thesis/<id>     the thesis, each agent's status and every answer (optimiser answers are read from the runs each time)
    GET  /api/theses          saved theses, newest first
The questions follow the packet's illustrative requests 1-7: speed, cost, environment, continuity, slip, premium, staging.
"""
import json
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'outputs' / 'theses'
SRV = None                    # the gridtech_server module (set by init)
THESES = {}
LOCK = threading.Lock()
PACKET_NOTE = ('Same packet as the optimiser (v6.0.1: wind $45/MWh, battery up to 53 MW at $1,220/kW). Its battery charges only from surplus clean '
               'contract energy, a stricter rule than the packet allows.')
RESOURCE_IDS = ['gas_turbine', 'gas_engine', 'fuel_cell', 'battery_4h', 'wind', 'utility_solar', 'nuclear']
QUESTIONS = ['speed', 'cost', 'environment', 'continuity', 'slip', 'premium', 'staging']
FIRST_OPENING, LAST_OPENING = 14, 109
EAST = {'SPP', 'MISO', 'PJM', 'ISO-NE', 'NYISO', 'SERC'}
WEST = {'CAISO', 'WECC (non-ISO)'}


def init(server_module):
    global SRV
    SRV = server_module
    OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------------------------ the thesis
def _num(errors, key, v, label, lo, hi, integer=False, required=True):
    if v in (None, ''):
        if required:
            errors[key] = f'{label}: enter a value'
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        errors[key] = f'{label}: enter a number'
        return None
    if integer and x != int(x):
        errors[key] = f'{label}: must be a whole number'
        return None
    if not lo <= x <= hi:
        errors[key] = f'{label}: enter {lo:g} - {hi:g}'
        return None
    return int(x) if integer else x


def parse(t):
    """Check the thesis; return the cleaned thesis or raise ValueError(errors)."""
    e = {}
    qs = [q for q in (t.get('questions') or []) if q in QUESTIONS]
    if not qs:
        e['questions'] = 'Pick at least one question for the agents to answer'
    isos = [str(x) for x in (t.get('isos') or [])][:6] or ['ERCOT']
    peak = _num(e, 'peakMw', t.get('peakMw'), 'Target campus MW', 5, 2000)
    full = lambda k, lab, req=True: _num(e, k, t.get(k), lab, FIRST_OPENING + 10, LAST_OPENING + 10, integer=True, required=req)
    c = {
        'title': str(t.get('title') or '').strip()[:120], 'isos': isos, 'peakMw': peak, 'questions': qs,
        'fullTarget': full('fullTarget', 'Full demand by month', 'speed' in qs or 'premium' in qs or 'continuity' in qs or 'staging' in qs),
        'fullCompare': full('fullCompare', 'Compare with full demand in month', 'speed' in qs or 'premium' in qs),
        'fullLatest': full('fullLatest', 'Latest full-demand month', 'cost' in qs),
        'premiumPct': _num(e, 'premiumPct', t.get('premiumPct'), 'Extra you would pay (%)', 0, 200, required='premium' in qs),
        'co2Max': _num(e, 'co2Max', t.get('co2Max'), 'Average emissions limit (kg CO2/MWh)', 1, 1000, required='environment' in qs),
        'slipMonths': _num(e, 'slipMonths', t.get('slipMonths'), 'Grid connection slip (months)', 1, 36, integer=True, required='slip' in qs),
        'slipOpening': _num(e, 'slipOpening', t.get('slipOpening'), 'Opening month to keep', FIRST_OPENING, LAST_OPENING, integer=True, required='slip' in qs),
        'slipDelayed': _num(e, 'slipDelayed', t.get('slipDelayed'), 'Or delay opening to month', FIRST_OPENING, LAST_OPENING, integer=True, required='slip' in qs),
        'wholesaleGrowth': _num(e, 'wholesaleGrowth', t.get('wholesaleGrowth', 5), 'Wholesale price growth (%/yr)', -20, 30),
        'gasGrowth': _num(e, 'gasGrowth', t.get('gasGrowth', 3), 'Gas price growth (%/yr)', -20, 30),
        'reliability': str(t.get('reliability') or 'R0').upper(),
        'weights': {k: max(0.0, float((t.get('weights') or {}).get(k, d))) for k, d in (('speed', 34), ('cost', 33), ('environment', 33))},
        'ownership': None,
        'appetite': t.get('appetite') if t.get('appetite') in ('Low', 'Medium', 'High') else 'Medium',
        'localNotes': str(t.get('localNotes') or '')[:2000],
        'stress': t.get('stress') if isinstance(t.get('stress'), dict) and t['stress'].get('mode') == 'custom' else None,
        'excluded': sorted({str(r) for r in (t.get('excluded') or []) if r in RESOURCE_IDS}),
        'capexUpliftPct': _num(e, 'capexUpliftPct', t.get('capexUpliftPct', 0) or 0, 'Onsite equipment cost increase (%)', 0, 100),
        'gridAdderExtra': _num(e, 'gridAdderExtra', t.get('gridAdderExtra', 0) or 0, 'Extra grid cost ($/MWh)', 0, 200),
    }
    if c['reliability'] not in ('R0', 'R1', 'R2', 'R3', 'R4', 'R5'):
        e['reliability'] = 'Choose R0-R5'
    if c['fullTarget'] and c['fullCompare'] and c['fullCompare'] <= c['fullTarget'] and ('speed' in qs or 'premium' in qs):
        e['fullCompare'] = 'Compare with a later full-demand month than your target'
    if c['slipOpening'] and c['slipDelayed'] and c['slipDelayed'] <= c['slipOpening'] and 'slip' in qs:
        e['slipDelayed'] = 'The delayed opening must be later than the opening you want to keep'
    if e:
        raise ValueError(e)
    if not c['title']:
        c['title'] = f"{peak:g} MW in {', '.join(isos)}"
    return c


def _stress_for(c):
    """Stress events for the optimiser: the customer's file, else the events of every grid in the thesis (ERCOT alone = the packet's)."""
    if c['stress']:
        return c['stress'], c['stress'].get('source') or 'your stress events'
    areas = []
    if any(i in EAST for i in c['isos']):
        areas.append('Eastern Interconnection')
    if any(i in WEST for i in c['isos']):
        areas.append('Western Interconnection')
    if not areas:
        return {'mode': 'packet'}, "packet events (7 ERCOT, 2023)"
    sm = SRV.stress_meta()
    rows = [r for r in sm['packet']] if 'ERCOT' in c['isos'] else []
    for a in sm['areas']:
        if a['area'] in areas:
            rows += [r for r in a['events'] if r.get('include')]
    clean = [{k: (str(v) if k in ('hours', 'importPct', 'noticeMin') else v) for k, v in r.items() if k in ('id', 'start', 'hours', 'importPct', 'noticeMin', 'operator', 'type', 'confidence', 'notes')} for r in rows]
    label = ' + '.join((['ERCOT (packet)'] if 'ERCOT' in c['isos'] else []) + areas) + ' stress events, 2023'
    return {'mode': 'custom', 'source': label, 'events': clean, 'excluded': []}, label


ONSITE = ('gas_turbine', 'gas_engine', 'fuel_cell', 'battery_4h')


def _resources(c):
    """Onsite equipment CAPEX raised by the thesis's uplift (e.g. losing the state sales-tax exemption), as optimiser resource overrides."""
    up = float(c.get('capexUpliftPct') or 0)
    if not up:
        return {}
    return {r: {'capex_usd_per_kw': round(float(SRV.G.OPT.loc[r, 'capex_usd_per_kw']) * (1 + up / 100), 2)} for r in ONSITE}


def _payload(c, start, **over):
    p = {'peakMw': c['peakMw'], 'calendarStart': None, 'start': int(start),
         'interconnection': {'mode': 'packet'} if c['peakMw'] == 75 else {'mode': 'steps', 'steps': [{'month': 18, 'mw': c['peakMw'] / 2}, {'month': 24, 'mw': c['peakMw']}]},
         'reliability': c['reliability'], 'co2Cap': None, 'wholesaleGrowth': c['wholesaleGrowth'] / 100, 'gasGrowth': c['gasGrowth'] / 100,
         'resources': _resources(c), 'minCaps': {}, 'staged': False, 'stressEvents': c['_stress'], 'gridDelay': 0, 'noTier2': False, 'excluded': c.get('excluded') or [],
         'gridAdderExtra': c.get('gridAdderExtra') or 0}
    p.update(over)
    p['start'] = max(FIRST_OPENING, min(LAST_OPENING, int(p['start'])))
    return p


def plan(c):
    """The optimiser scenarios each question needs (environment adds a second, adaptive one once the first is done)."""
    sc = {}
    def add(key, label, start, **over):
        sc[key] = {'key': key, 'label': label, 'payload': _payload(c, start, **over), 'jobId': None, 'status': 'queued'}
    q = c['questions']
    ft, fc = c['fullTarget'], c['fullCompare']
    if ft and any(x in q for x in ('speed', 'premium', 'continuity', 'staging')):
        add('target', f'Full demand by month {ft} (opening {ft - 10})', ft - 10)
    if fc and any(x in q for x in ('speed', 'premium')):
        add('compare', f'Full demand by month {fc} (opening {fc - 10})', fc - 10)
    if 'cost' in q:
        add('latest', f"Lowest cost, full demand by month {c['fullLatest']} (opening {c['fullLatest'] - 10})", c['fullLatest'] - 10)
    if 'environment' in q:
        s0 = (ft - 10) if ft else 20
        add('co2_a', f"Below {c['co2Max']:g} kg CO2/MWh, opening {s0}", s0, co2Cap=c['co2Max'])
    if 'continuity' in q:
        add('no_t2', f'No Tier 2 interruption, opening {ft - 10}', ft - 10, noTier2=True)
    if 'slip' in q:
        o, o2, d = c['slipOpening'], c['slipDelayed'], c['slipMonths']
        add('slip_base', f'Opening {o}, grid on time', o)
        add('slip_keep', f'Opening {o}, grid {d} months late', o, gridDelay=d)
        add('slip_delay', f'Opening {o2}, grid {d} months late', o2, gridDelay=d)
    if 'staging' in q:
        add('staged', f'Capacity added in stages, opening {ft - 10}', ft - 10, staged=True)
    return sc


# ------------------------------------------------------------------------------------------------ persistence
def _save(th):
    try:
        (OUT / f"{th['id']}.json").write_text(json.dumps(th, default=str))
    except Exception:
        traceback.print_exc()


def _load(tid):
    with LOCK:
        if tid in THESES:
            return THESES[tid]
    f = OUT / f'{tid}.json'
    if not f.is_file():
        return None
    th = json.loads(f.read_text())
    for a in th['agents'].values():                 # the server restarted while it was working
        if a['status'] in ('running', 'queued'):
            a['status'], a['detail'] = 'stopped', 'Stopped when the server restarted. Run the thesis again to finish.'
    if th.get('mode') == 'run':
        v = SRV.job_view(th['runJobId']) or {}
        if v.get('status') == 'done':
            th['scenarios']['run']['status'] = 'done'
            th['agents']['optimisation'].update(status='done', detail='Found a feasible portfolio' if (v.get('result') or {}).get('status') == 'OK' else 'No feasible portfolio for these inputs')
    for s in th['scenarios'].values():
        if s['status'] in ('queued', 'running') and not (s.get('jobId') and (SRV.job_view(s['jobId']) or {}).get('status') == 'done'):
            s['status'] = 'stopped'
        elif s['status'] == 'running':
            s['status'] = 'done'
    with LOCK:
        THESES[tid] = th
    return th


def start(t):
    try:
        c = parse(t)
    except ValueError as e:
        errs = e.args[0] if e.args and isinstance(e.args[0], dict) else {'thesis': str(e)}
        return 400, {'error': 'Some thesis answers need fixing', 'errors': errs}
    c['_stress'], c['_stressLabel'] = _stress_for(c)
    tid = uuid.uuid4().hex[:10]
    th = {'id': tid, 'createdAt': time.time(), 'thesis': c, 'scenarios': plan(c),
          'agents': {
              'optimisation': {'status': 'queued', 'detail': 'Waiting to start', 'name': 'Optimisation agent'},
              'financial': {'status': 'queued', 'detail': 'Waiting to start', 'name': 'Financial agent'},
              'sourcing': {'status': 'queued', 'detail': 'Waiting to start', 'name': 'Site sourcing agent'},
              'moratorium': {'status': 'queued', 'detail': 'Waiting to start', 'name': 'Policy agent'}},
          'financial': None, 'moratorium': None}
    with LOCK:
        THESES[tid] = th
    _save(th)
    threading.Thread(target=_run_financial, args=(th,), daemon=True).start()
    threading.Thread(target=_run_moratorium, args=(th,), daemon=True).start()
    threading.Thread(target=_run_optimiser, args=(th,), daemon=True).start()
    return 200, {'id': tid}


# ------------------------------------------------------------------------------------------------ agents
def _questions_for_financial(c):
    if c.get('fullTarget') and not c['questions']:          # agents around one optimiser run
        return [{'key': 'match', 'opening': c['fullTarget'] - 10, 'co2Max': c.get('co2Max')}]
    out = []
    for k in c['questions']:
        q = {'key': k}
        q.update({x: c[x] for x in ('fullTarget', 'fullCompare', 'fullLatest', 'co2Max', 'premiumPct') if c.get(x) is not None})
        out.append(q)
    return out


def _run_financial(th, rerank=None):
    a, c = th['agents']['financial'], th['thesis']
    a['status'], a['detail'] = 'running', 'Searching the portfolio pools'
    w = {'speed': 0, 'cost': 100, 'environment': 0}       # cost-only: the pool's cheapest portfolio that opens in time and meets the carbon limit
    rerank = None
    go = (c['fullTarget'] - 10) if c.get('fullTarget') and 14 <= c['fullTarget'] - 10 <= 26 else None
    body = {'questions': _questions_for_financial(c), 'excluded': c.get('excluded') or [],
            'recommend': {'weights': [w['speed'], w['cost'], w['environment']], 'goLive': go, 'carbonMax': c.get('co2Max'), 'costPct': None, 'excluded': c.get('excluded') or []}}
    if rerank:
        body['only'] = 'recommend'
    try:
        p = subprocess.run([sys.executable, str(HERE / 'financial_agent.py')], input=json.dumps(body), capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or 'failed').strip().splitlines()[-1])
        got = json.loads(p.stdout)
        if rerank and th.get('financial'):          # keep the answers; replace the recommendation, trade-offs and sensitivity
            th['financial']['recommendation'] = got.get('recommendation')
            th['financial']['weights'] = rerank
            a['status'], a['detail'] = 'done', f"Re-ranked for speed {rerank['speed']:g} / cost {rerank['cost']:g} / environment {rerank['environment']:g}"
            _save(th)
            return
        th['financial'] = got
        th['financial']['weights'] = w
        th['financial']['weightless'] = True
        th['financial'].pop('packetNote', None)
        notes = []
        if c['peakMw'] != 75:
            notes.append(f"its pools are for the 75 MW packet campus, not {c['peakMw']:g} MW")
        if c['isos'] != ['ERCOT'] or c['stress']:
            notes.append("its pools use the packet's ERCOT stress events")
        if c.get('capexUpliftPct') or c.get('gridAdderExtra'):
            notes.append("its pools use packet costs, without your cost adders (the optimiser includes them)")
        th['financial']['scopeNote'] = ('Scope: ' + '; '.join(notes) + '.') if notes else None
        a['status'], a['detail'] = 'done', f"Searched {th['financial']['poolSize']:,} portfolios"
    except Exception as e:
        traceback.print_exc()
        a['status'], a['detail'] = 'failed', f'{type(e).__name__}: {e}'
    _save(th)


def _post(url, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


# ---- Atria Site Decisions, in-process (copied from ~/Downloads/Atria/atria-site-decisions: atria_model, config, data)
ATRIA_DIR = HERE / 'atria_site'


def _atria():
    if str(ATRIA_DIR) not in sys.path:
        sys.path.insert(0, str(ATRIA_DIR))
    from atria_model import evaluate as E
    return E


def _plain(x):
    return json.loads(json.dumps(x, default=str))       # the same plain JSON the Atria server would send


def _atria_map():
    """Parcel outline and county record per site (the Atria server's /api/map)."""
    E, out = _atria(), []
    for sid in E.site_ids():
        d = E.site_doc(sid)
        row = dict(id=sid, ring=None)
        f = ATRIA_DIR / 'data' / 'raw' / 'sites' / f'{sid}.json'
        if f.is_file():
            p = (json.loads(f.read_text()).get('parcel') or {}).get('data') or {}
            ring = p.get('outline')
            ring = json.loads(ring) if isinstance(ring, str) else ring
            lk = ATRIA_DIR / 'data' / 'raw' / 'sites' / 'atria' / sid / 'lookup.json'
            lr = (json.loads(lk.read_text()).get('parcel', {}).get('value', {}).get('geometry_rings') or [None])[0] if lk.is_file() else None
            if lr:
                ring = [lr]
            row.update(ring=(ring or [None])[0], apn=p.get('geoId'), address=p.get('situsAddress') if p.get('situsAddress') not in (None, 'None', ', , TX 79908') else None,
                       acres=float(p['gisAcres']) if p.get('gisAcres') else None, legalAcres=float(p['legalAcres']) if p.get('legalAcres') else None)
        out.append(row)
    return out


def _run_moratorium(th):
    """Site sourcing agent (proposed sites, parcels and site facts) and moratorium agent (moratoriums and local risk), both from the Atria API."""
    a, c = th['agents']['moratorium'], th['thesis']
    src = th['agents'].setdefault('sourcing', {'name': 'Site sourcing agent'})
    a['status'], a['detail'] = 'running', 'Checking moratoriums, state rules and local risk'
    src['status'], src['detail'] = 'running', 'Finding proposed sites in your grids (Atria API)'
    try:
        E = _atria()
        t = dict(E.default_thesis())
        t.update({'isos': c['isos'], 'targetMW': c['peakMw'], 'carbonMax': c.get('co2Max'), 'appetite': c['appetite'], 'backup': c['reliability'],
                  'answered': ['thesis.states_isos_load', 'thesis.target_campus_mw', 'thesis.risk_appetite']})
        sites = _plain(E.all_sites(t))
        pick = lambda cat: lambda s: {k: (s['categories'].get(cat) or {}).get(k) for k in ('headline', 'outcome', 's', 'reason')}
        def acres(s):
            for u in (s['categories'].get('Technical') or {}).get('used', []):
                if u.get('name') == 'usable_acres':
                    return u.get('value')
            return None
        try:
            parcels = {m['id']: m for m in _plain(_atria_map())}
        except Exception:
            parcels = {}
        rows = []
        for s in sites:
            rows.append({'id': s['id'], 'label': s['label'], 'county': s['county'], 'state': s['state'], 'iso': s['iso'], 'score': s['score'], 'verdict': s['verdict'],
                         'thesisFails': s.get('thesisFails', [])[:3], 'acres': acres(s), 'stall': s['stall'].get('p_stall'), 'stallMax': s['stall'].get('max'),
                         'stallLayers': {k: s['stall'].get(f'p_{k}') for k in ('local', 'state', 'commercial')}, 'stallInputs': s['stall'].get('local_inputs', []),
                         'moratorium': pick('Policy')(s), 'local': pick('Cultural')(s), 'environment': pick('Environmental')(s),
                         'lat': s.get('lat'), 'lng': s.get('lng'),
                         'parcel': {k: (parcels.get(s['id']) or {}).get(k) for k in ('ring', 'apn', 'address', 'acres', 'legalAcres')} if parcels.get(s['id']) else None,
                         'gridP50': (s.get('time') or {}).get('grid', {}).get('p50'), 'powerBy': (s.get('time') or {}).get('months'),
                         'risks': [r['text'] for r in s.get('risks', [])[:4]]})
        inside = [r for r in rows if r['iso'] in c['isos']] or rows
        for r in inside:                         # land, infrastructure, policy and local facts per site (Atria API indicators)
            try:
                det = _plain(E.evaluate(r['id'], t, detail=True))
                r['land'] = _land_facts(det)
                r['policy'] = _policy_facts(det)
            except Exception as e:
                r['land'] = {'error': f'{type(e).__name__}'}
            if r.get('acres') is None and (r.get('parcel') or {}).get('acres'):
                r['acres'], r['acresBasis'] = r['parcel']['acres'], 'parcel'     # usable acres unknown: the parcel's mapped size
            r.pop('url', None)
        inside = [r for r in inside if r.get('acres')]                           # no size at all: not proposed
        th['moratorium'] = {'source': 'policy agent', 'sites': inside, 'allSites': len(rows),
                            'note': None if len(inside) < len(rows) or all(r['iso'] in c['isos'] for r in rows) else 'No screened site lies in your grids; showing every screened site.'}
        found = sum(1 for r in inside if (r['moratorium'].get('outcome') or '') == 'fail')
        a['status'], a['detail'] = 'done', f"Checked moratoriums and local risk at {len(inside)} site{'s' if len(inside) != 1 else ''}: {found} with a local moratorium or approval problem"
        src['status'], src['detail'] = 'done', f"Proposed {len(inside)} site{'s' if len(inside) != 1 else ''} in {', '.join(c['isos'])} with parcels and site facts (Atria API)"
    except Exception as e:
        traceback.print_exc()
        a['status'], a['detail'] = 'failed', f'The site data could not be read ({type(e).__name__}: {e})'
        src['status'], src['detail'] = 'failed', a['detail']
    _save(th)


LAND_FIELDS = [   # (indicator, label, group)
    ('parcel_acres', 'Parcel size', 'Parcel'), ('usable_acres', 'Usable acres (outside wetlands)', 'Parcel'), ('adjoining_parcels', 'Adjoining parcels', 'Parcel'),
    ('same_owner_acres', 'Adjoining land, same owner', 'Parcel'), ('easements_crossing', 'Easements or lines crossing', 'Parcel'), ('slope_mean_pct', 'Mean slope', 'Parcel'),
    ('land_value_per_acre', 'Appraised land value', 'Parcel'),
    ('nearest_substation', 'Nearest substation', 'Power and gas'), ('dist_345kv_mi', 'Distance to a 345 kV substation', 'Power and gas'), ('line_345kv_mi', 'Distance to a 345 kV line', 'Power and gas'),
    ('gas_dist_mi', 'Distance to a gas pipeline', 'Power and gas'), ('grid_p50_months', 'Grid connection, median estimate', 'Power and gas'),
    ('nearest_ix', 'Nearest internet exchange', 'Fiber'), ('fiber_ix_dist_mi', 'Distance to that exchange', 'Fiber'), ('nearest_carrier_facility_mi', 'Distance to a carrier facility', 'Fiber'),
    ('water_system', 'Water system', 'Water and environment'), ('water_stress_0_5', 'Water stress (0-5)', 'Water and environment'), ('flood_zone', 'Flood zone', 'Water and environment'),
    ('wetland_share', 'Share in wetlands', 'Water and environment'), ('heat_days', 'Hot days a year (cooling)', 'Water and environment'), ('attainment', 'Air quality area', 'Water and environment'),
    ('air_path', 'Air permit path for onsite gas', 'Water and environment'), ('contamination_sites_nearby', 'Contamination sites nearby', 'Water and environment'),
    ('zoning', 'Zoning', 'Rules and hazards'), ('data_center_approval', 'Data-center approval', 'Rules and hazards'), ('hazard_top', 'Main natural hazards', 'Rules and hazards'),
    ('hazard_index', 'Natural hazard index (0-100)', 'Rules and hazards'),
]
POLICY_FIELDS = ['moratorium_stage', 'moratorium_history', 'data_center_approval', 'zoning', 'state_rules', 'dc_incentive', 'share_oppose', 'net_stance', 'noise_dba_home', 'burden_pct']


def _policy_facts(d):
    ind = {i['name']: i for i in d.get('indicators', [])}
    return {k: {'value': ind[k].get('value'), 'note': ind[k].get('note') or '', 'source': (ind[k].get('source') or '').replace('Atria: ', ''), 'date': ind[k].get('read_date')}
            for k in POLICY_FIELDS if k in ind and ind[k].get('value') not in (None, '')}


def _land_facts(d):
    ind = {i['name']: i for i in d.get('indicators', [])}
    out = {'lat': d.get('lat'), 'lng': d.get('lng'), 'facts': []}
    for k, lab, grp in LAND_FIELDS:
        i = ind.get(k)
        if not i or i.get('value') in (None, '') or i.get('status') not in ('Answered', 'Example'):
            continue
        out['facts'].append({'key': k, 'label': lab, 'group': grp, 'value': i['value'], 'unit': i.get('unit') or '',
                             'source': (i.get('source') or '').replace('Atria: ', ''), 'placeholder': i.get('status') == 'Example'})
    return out


def _same_inputs(p):
    """A finished run with exactly these inputs, so the agent can reuse it instead of running again."""
    k = SRV._case_keys(p)
    if not k:
        return None
    for v in SRV._all_views():
        if v.get('status') != 'done' or (v.get('result') or {}).get('status') not in ('OK', 'INFEASIBLE'):
            continue
        if SRV._case_keys(v.get('inputs') or {}) == k:
            return v.get('id')
    return None


def _wait(jid):
    while True:
        v = SRV.job_view(jid)
        if not v or v.get('status') != 'running':
            return v
        time.sleep(2)


def _run_one(th, s):
    s['status'] = 'running'
    old = _same_inputs(s['payload'])
    if old:
        s['jobId'], s['status'], s['reused'] = old, 'done', True
        return SRV.job_view(old)
    while True:
        code, body = SRV.start_job({**s['payload'], 'thesis': {'id': th['id'], 'title': th['thesis']['title'], 'scenario': s['label']}})
        if code == 200:
            break
        if code == 409:                             # another run (manual or another thesis) is using the optimiser
            th['agents']['optimisation']['detail'] = 'Waiting for the run in progress to finish'
            time.sleep(4)
            continue
        s['status'], s['error'] = 'failed', body.get('error') or 'could not start'
        s['errors'] = body.get('errors')
        return None
    s['jobId'] = body['jobId']
    _save(th)
    v = _wait(body['jobId'])
    s['status'] = 'done' if v and v.get('status') == 'done' else 'failed'
    return v


def _ok(v):
    return bool(v) and (v.get('result') or {}).get('status') == 'OK'


def _run_optimiser(th):
    a, c = th['agents']['optimisation'], th['thesis']
    a['status'] = 'running'
    try:
        order = list(th['scenarios'].values())
        i = 0
        while i < len(order):
            s = order[i]
            if th.get('stop'):
                for x in order[i:]:
                    if x['status'] == 'queued':
                        x['status'] = 'stopped'
                a['status'], a['detail'] = 'stopped', 'Stopped by you'
                _save(th)
                return
            done = sum(1 for x in order if x['status'] == 'done')
            a['detail'] = f"Run {done + 1} of {len(order)}: {s['label']}"
            _save(th)
            v = _run_one(th, s)
            if s['key'] == 'co2_a' and v is not None:       # search the opening month: one step earlier if it worked, later if not
                s0 = s['payload']['start']
                nxt = max(FIRST_OPENING, s0 - 4) if _ok(v) else min(LAST_OPENING, s0 + 6)
                if nxt != s0:
                    lab = f"Below {c['co2Max']:g} kg CO2/MWh, opening {nxt}"
                    th['scenarios']['co2_b'] = {'key': 'co2_b', 'label': lab, 'payload': _payload(c, nxt, co2Cap=c['co2Max']), 'jobId': None, 'status': 'queued'}
                    order.append(th['scenarios']['co2_b'])
            i += 1
        failed = sum(1 for x in order if x['status'] == 'failed')
        a['status'] = 'done' if not failed else 'failed'
        a['detail'] = f"{len(order)} optimiser run{'s' if len(order) != 1 else ''} finished" + (f", {failed} failed" if failed else '')
    except Exception as e:
        traceback.print_exc()
        a['status'], a['detail'] = 'failed', f'{type(e).__name__}: {e}'
    _save(th)


# ------------------------------------------------------------------------------------------------ answers (read from the runs on every request)
GAS = ('gas_turbine', 'gas_engine', 'fuel_cell')


def _gas_use(jid):
    """MWh per gas resource in each model year, from the run's dispatch.csv (cached next to it)."""
    d = next(iter(SRV.OUT.glob(f'*_{jid}')), None)
    if d is None or not (d / 'dispatch.csv').is_file():
        return None
    cache = d / 'gas_use.json'
    if cache.is_file():
        return json.loads(cache.read_text())
    import pandas as pd
    cols = [f'{r}_mw' for r in GAS]
    df = pd.read_csv(d / 'dispatch.csv', usecols=lambda c_: c_ in cols)
    yr = df.index // 8760
    out = {r: [float(x) for x in df.groupby(yr)[f'{r}_mw'].sum()] for r in GAS if f'{r}_mw' in df}
    cache.write_text(json.dumps(out))
    return out


def _run_summary(v):
    if not v:
        return None
    r = v.get('result') or {}
    base = {'jobId': v.get('id'), 'tag': v.get('tag'), 'status': v.get('status'), 'resultStatus': r.get('status'), 'phase': v.get('phase'),
            'elapsed': v.get('elapsed'), 'reason': ((r.get('diagnosis') or {}).get('reasons') or [{}])[0].get('title') if r.get('status') == 'INFEASIBLE' else None}
    if r.get('status') != 'OK':
        return base
    k = r.get('kpis') or {}
    req = k.get('requested_mwh') or 1
    caps = {x['id']: x['mw'] for x in r.get('capacities') or [] if x.get('mw')}
    labels = {x['id']: x['label'] for x in r.get('capacities') or []}
    base.update({'opening': r.get('opening'), 'fullDemand': r.get('fullDemand'), 'costM': (k.get('total_decision_cost_usd') or 0) / 1e6,
                 'dollarsPerMwh': k.get('decision_cost_usd_per_requested_mwh'), 'kgPerMwh': k.get('combined_electricity_co2_kg_per_mwh'),
                 'cleanSharePct': (k.get('clean_energy_share') or 0) * 100, 'merchantGwh': (k.get('merchant_mwh') or 0) / 1e3,
                 'merchantSharePct': (k.get('merchant_mwh') or 0) / req * 100, 'interruptedMwh': k.get('interrupted_mwh'), 'relocatedMwh': k.get('relocated_local_equivalent_mwh'),
                 'landAcres': k.get('land_acres'), 'landLimit': (r.get('effective') or {}).get('landAcres'), 'maxImportMw': k.get('max_grid_import_mw'),
                 'caps': caps, 'detail': ' + '.join(f"{labels.get(i, i)} {mw:g} MW" for i, mw in caps.items()) or 'grid only',
                 'capacities': r.get('capacities'), 'capacitiesByYear': r.get('capacitiesByYear'), 'annual': r.get('annual'), 'stress': r.get('stress'),
                 'gridSchedule': (r.get('effective') or {}).get('gridSchedule'), 'checker': r.get('checker'), 'sizingStatus': r.get('sizingStatus'),
                 'co2CapMet': r.get('co2CapMet'), 'staged': r.get('staged')})
    try:
        base['gasMwhByYear'] = _gas_use(v.get('id'))
    except Exception:
        base['gasMwhByYear'] = None
    return base


def _diff(a, b, f='costM'):
    return None if not a or not b or a.get(f) is None or b.get(f) is None else a[f] - b[f]


def _opt_answers(th, runs):
    c, q, A = th['thesis'], th['thesis']['questions'], {}
    g = lambda k: runs.get(k)
    ok = lambda r: bool(r) and r.get('resultStatus') == 'OK'
    pend = lambda *ks: any(th['scenarios'][k]['status'] in ('queued', 'running') for k in ks if k in th['scenarios'])
    if 'speed' in q:
        t, b = g('target'), g('compare')
        if pend('target', 'compare'):
            A['speed'] = {'status': 'pending'}
        elif ok(t) and ok(b):
            d = _diff(t, b)
            A['speed'] = {'status': 'ok', 'target': t, 'compare': b, 'extraM': d, 'extraPct': d / b['costM'] * 100,
                          'text': f"Full demand by month {c['fullTarget']}: {t['detail']}, ${t['costM']:,.1f}M. By month {c['fullCompare']}: {b['detail']}, ${b['costM']:,.1f}M. "
                                  f"Reaching it {c['fullCompare'] - c['fullTarget']} months sooner costs ${d:,.1f}M more ({d / b['costM'] * 100:.1f}%)."}
        else:
            A['speed'] = {'status': 'none', 'target': t, 'compare': b, 'text': 'No feasible portfolio for ' + ' and '.join(x for x, r in ((f"month {c['fullTarget']}", t), (f"month {c['fullCompare']}", b)) if not ok(r)) + '.'}
    if 'cost' in q:
        r = g('latest')
        A['cost'] = {'status': 'pending'} if pend('latest') else {'status': 'ok', 'best': r, 'text': f"Lowest cost with full demand by month {c['fullLatest']}: {r['detail']}, ${r['costM']:,.1f}M; opens month {r['opening']}, full demand month {r['fullDemand']}, {r['kgPerMwh']:.0f} kg CO2/MWh, {r['merchantSharePct']:.0f}% of energy from the merchant grid. Tested opening month {r['opening']} only (later openings are usually cheaper)."} if ok(r) else {'status': 'none', 'best': r, 'text': 'No feasible portfolio for that date.'}
    if 'environment' in q:
        tried = [g(k) for k in ('co2_a', 'co2_b') if g(k)]
        if pend('co2_a', 'co2_b'):
            A['environment'] = {'status': 'pending', 'tried': tried}
        else:
            good = [r for r in tried if ok(r)]
            tested = sorted(th['scenarios'][k]['payload']['start'] for k in ('co2_a', 'co2_b') if k in th['scenarios'])
            if good:
                best = min(good, key=lambda r: (r['fullDemand'], r['costM']))
                A['environment'] = {'status': 'ok', 'best': best, 'tried': tried, 'tested': tested,
                                    'text': f"Earliest full demand found below {c['co2Max']:g} kg CO2/MWh: month {best['fullDemand']} (opening {best['opening']}), {best['detail']}, ${best['costM']:,.1f}M, {best['kgPerMwh']:.0f} kg/MWh. Openings tested: {', '.join(map(str, tested))}."}
            else:
                A['environment'] = {'status': 'none', 'tried': tried, 'tested': tested, 'text': f"No feasible portfolio below {c['co2Max']:g} kg CO2/MWh at openings {', '.join(map(str, tested))}."}
    if 'continuity' in q:
        n, t = g('no_t2'), g('target')
        if pend('no_t2', 'target'):
            A['continuity'] = {'status': 'pending'}
        elif ok(n) and ok(t):
            d = _diff(n, t)
            ch = {}
            for rid in set(n['caps']) | set(t['caps']):
                x = n['caps'].get(rid, 0) - t['caps'].get(rid, 0)
                if abs(x) > 1e-9:
                    ch[rid] = x
            A['continuity'] = {'status': 'ok', 'noTier2': n, 'withTier2': t, 'extraM': d, 'changes': ch,
                               'text': f"Without any Tier 2 interruption: {n['detail']}, ${n['costM']:,.1f}M, against ${t['costM']:,.1f}M when interruption is allowed ({d:+,.1f}M)."}
        else:
            A['continuity'] = {'status': 'none', 'noTier2': n, 'withTier2': t, 'text': 'No feasible portfolio without Tier 2 interruption at that date.' if not ok(n) else 'The comparison run has no feasible portfolio.'}
    if 'slip' in q:
        b0, k1, k2 = g('slip_base'), g('slip_keep'), g('slip_delay')
        if pend('slip_base', 'slip_keep', 'slip_delay'):
            A['slip'] = {'status': 'pending'}
        else:
            o, o2, d = c['slipOpening'], c['slipDelayed'], c['slipMonths']
            parts = []
            if ok(k1):
                parts.append(f"Keeping opening {o} with the grid {d} months late is feasible: {k1['detail']}, ${k1['costM']:,.1f}M" + (f" ({_diff(k1, b0):+,.1f}M against the grid on time)" if ok(b0) else '') + '.')
            else:
                parts.append(f"Keeping opening {o} with the grid {d} months late has no feasible portfolio.")
            if ok(k2):
                parts.append(f"Delaying opening to month {o2}: {k2['detail']}, ${k2['costM']:,.1f}M" + (f" ({_diff(k2, k1):+,.1f}M against adapting)" if ok(k1) else '') + '.')
            A['slip'] = {'status': 'ok' if ok(k1) or ok(k2) else 'none', 'base': b0, 'keep': k1, 'delay': k2, 'text': ' '.join(parts)}
    if 'premium' in q:
        t, b = g('target'), g('compare')
        if pend('target', 'compare'):
            A['premium'] = {'status': 'pending'}
        elif ok(t) and ok(b):
            p = (t['costM'] / b['costM'] - 1) * 100
            fits = p <= c['premiumPct'] + 1e-9
            A['premium'] = {'status': 'ok', 'target': t, 'compare': b, 'premiumPct': p, 'feasible': fits,
                            'text': f"The cheapest month-{c['fullTarget']} portfolio costs {p:.1f}% more than the month-{c['fullCompare']} one, "
                                    + (f"within your {c['premiumPct']:g}%. Emissions: {t['kgPerMwh']:.0f} against {b['kgPerMwh']:.0f} kg CO2/MWh ({t['kgPerMwh'] - b['kgPerMwh']:+.0f})." if fits
                                       else f"above your {c['premiumPct']:g}%: no portfolio reaches month {c['fullTarget']} within that premium (the optimiser already finds the cheapest).")}
        else:
            A['premium'] = {'status': 'none', 'text': 'One of the two dates has no feasible portfolio.'}
    if 'staging' in q:
        u, st = g('target'), g('staged')
        if pend('target', 'staged'):
            A['staging'] = {'status': 'pending'}
        elif ok(u) and ok(st):
            d = _diff(st, u)
            early = lambda r: sum(a.get('total_decision_cost_usd', 0) for a in (r.get('annual') or [])[:3]) / 1e6
            A['staging'] = {'status': 'ok', 'upfront': u, 'staged': st, 'extraM': d, 'earlyUpfrontM': early(u), 'earlyStagedM': early(st),
                            'text': f"Staged: ${st['costM']:,.1f}M against ${u['costM']:,.1f}M upfront ({d:+,.1f}M over ten years). First three model years: ${early(st):,.1f}M staged, ${early(u):,.1f}M upfront. Final capacity staged: {st['detail']}."}
        else:
            A['staging'] = {'status': 'none', 'text': 'One of the two runs has no feasible portfolio.'}
    return A


def _compare(opt, fin):
    """Where both agents answered the same question: their numbers side by side."""
    rows = []
    def pick(a, key):
        return (a or {}).get(key) if isinstance((a or {}).get(key), dict) else None
    spec = {'speed': [('target', 'Full demand by target month'), ('compare', 'Full demand by comparison month')], 'cost': [('best', 'Lowest cost')],
            'environment': [('best', 'Earliest below the emissions limit')], 'continuity': [('noTier2', 'No Tier 2 interruption'), ('withTier2', 'Tier 2 interruption allowed')],
            'premium': [('target', 'Target month'), ('compare', 'Comparison month')]}
    for q, items in spec.items():
        o, f = opt.get(q), (fin or {}).get(q)
        if not o or not f or o.get('status') != 'ok' or f.get('status') != 'ok':
            continue
        for key, lab in items:
            a, b = pick(o, key), pick(f, key)
            if a and b:
                rows.append({'question': q, 'item': lab, 'optimiser': {k: a.get(k) for k in ('opening', 'fullDemand', 'costM', 'kgPerMwh', 'cleanSharePct', 'merchantSharePct', 'detail')},
                             'financial': {k: b.get(k) for k in ('opening', 'fullDemand', 'costM', 'kgPerMwh', 'cleanSharePct', 'merchantSharePct', 'detail')},
                             'costDiffPct': (a['costM'] / b['costM'] - 1) * 100 if b.get('costM') else None})
    return rows


def view(tid):
    th = _load(tid)
    if not th:
        return None
    if (th['agents'].get('moratorium') or {}).get('status') == 'not-connected':     # saved while the site data was a separate server: read it now
        th['agents']['moratorium']['status'] = 'queued'
        with LOCK:
            THESES[tid] = th                               # keep it in memory while the agent works, so polls see its progress
        threading.Thread(target=_run_moratorium, args=(th,), daemon=True).start()
    fa = th['agents'].get('financial') or {}
    if fa.get('status') == 'done' and th.get('financial') and not th['financial'].get('weightless'):    # saved with priority weights: redo cost-only
        fa['status'] = 'queued'
        with LOCK:
            THESES[tid] = th
        threading.Thread(target=_run_financial, args=(th,), daemon=True).start()
    runs = {k: _run_summary(SRV.job_view(s['jobId'])) if s.get('jobId') else None for k, s in th['scenarios'].items()}
    opt = _opt_answers(th, runs)
    fin = (th.get('financial') or {}).get('answers')
    scen = [{k: s.get(k) for k in ('key', 'label', 'jobId', 'status', 'reused', 'error')} | {'run': runs.get(s['key']), 'start': s['payload']['start'],
             'gridDelay': s['payload'].get('gridDelay'), 'noTier2': s['payload'].get('noTier2'), 'staged': s['payload'].get('staged'), 'co2Cap': s['payload'].get('co2Cap')}
            for s in th['scenarios'].values()]
    if th.get('mode') == 'run':
        r0, m = runs.get('run'), (fin or {}).get('match') or {}
        b = m.get('best') if m.get('status') == 'ok' else None
        if r0 and r0.get('resultStatus') == 'OK' and b:
            keys = ('opening', 'fullDemand', 'costM', 'kgPerMwh', 'cleanSharePct', 'merchantSharePct', 'detail')
            compare_rows = [{'question': 'match', 'item': f"Opening month {r0['opening']}", 'optimiser': {k: r0.get(k) for k in keys}, 'financial': {k: b.get(k) for k in keys},
                             'costDiffPct': (r0['costM'] / b['costM'] - 1) * 100 if b.get('costM') else None}]
        else:
            compare_rows = []
    else:
        compare_rows = None
    t = {k: v for k, v in th['thesis'].items() if k != '_stress'}
    th['agents']['moratorium']['name'] = 'Policy agent'          # renamed: moratoriums and local risk
    th['agents'].get('sourcing', {}).setdefault('name', 'Site sourcing agent')
    return {'id': th['id'], 'createdAt': th['createdAt'], 'thesis': t, 'agents': th['agents'], 'scenarios': scen,
            'mode': th.get('mode', 'thesis'), 'runJobId': th.get('runJobId'),
            'answers': {'optimisation': opt, 'financial': fin}, 'compare': compare_rows if compare_rows is not None else _compare(opt, fin),
            'financial': ({k: v for k, v in (th.get('financial') or {}).items() if k != 'answers'} | {'packetNote': PACKET_NOTE}) if th.get('financial') else None, 'moratorium': th.get('moratorium')}


def rerank(tid, weights, full=False):
    th = _load(tid)
    if not th:
        return 404, {'error': 'unknown thesis'}
    if agentbusy(th['agents']['financial']):
        return 409, {'error': 'The financial agent is still working'}
    if full:                                   # answer every question again (e.g. after the pools were rebuilt) and re-screen the sites
        th['agents']['financial']['status'] = 'running'
        th['agents']['moratorium']['status'] = 'running'
        threading.Thread(target=_run_financial, args=(th,), daemon=True).start()
        threading.Thread(target=_run_moratorium, args=(th,), daemon=True).start()
        return 200, {'ok': True}
    try:
        w = {k: max(0.0, float((weights or {}).get(k, 0))) for k in ('speed', 'cost', 'environment')}
    except (TypeError, ValueError):
        return 400, {'error': 'Weights must be numbers'}
    if sum(w.values()) <= 0:
        return 400, {'error': 'At least one weight must be above zero'}
    th['agents']['financial']['status'] = 'running'
    threading.Thread(target=_run_financial, args=(th, w), daemon=True).start()
    return 200, {'ok': True}


def agentbusy(a):
    return a.get('status') in ('running', 'queued')


def stop(tid):
    th = _load(tid)
    if not th:
        return 404, {'error': 'unknown thesis'}
    th['stop'] = True                      # the optimisation agent stops before its next run; the run in progress finishes
    _save(th)
    return 200, {'ok': True}


# ------------------------------------------------------------------------------------------------ agents for one optimiser run
RUNMAP = OUT / 'by_run.json'


def _runmap():
    try:
        return json.loads(RUNMAP.read_text())
    except (OSError, ValueError):
        return {}


def thesis_for_run(jid):
    return _runmap().get(jid)


_TITLES = {}


def thesis_title(tid):
    """The title of a saved thesis (cached), for naming its runs."""
    if tid not in _TITLES:
        try:
            _TITLES[tid] = (json.loads((OUT / f'{tid}.json').read_text()).get('thesis') or {}).get('title')
        except (OSError, ValueError):
            return None
    return _TITLES[tid]


def start_for_run(jid, inputs, agents=None):
    """The four agents around one optimiser run (the Optimizer form): financial, site sourcing and policy agents start now;
    the optimisation agent is the run itself."""
    if thesis_for_run(jid):
        return 200, {'id': thesis_for_run(jid)}
    agents = agents or {}
    w = agents.get('weights') or {'speed': 34, 'cost': 33, 'environment': 33}
    start_m = int(inputs.get('start') or 20)
    stress = inputs.get('stressEvents') if (inputs.get('stressEvents') or {}).get('mode') == 'custom' else None
    c = {'title': agents.get('title') or f"{agents.get('template') or 'Your'} run · {float(inputs.get('peakMw') or 75):g} MW, opening month {start_m}",
         'isos': ['ERCOT'], 'peakMw': float(inputs.get('peakMw') or 75), 'questions': [], 'fullTarget': start_m + 10, 'fullCompare': start_m + 16,
         'fullLatest': None, 'premiumPct': None, 'co2Max': float(inputs['co2Cap']) if inputs.get('co2Cap') not in (None, '') else None,
         'slipMonths': None, 'slipOpening': None, 'slipDelayed': None,
         'wholesaleGrowth': float(inputs.get('wholesaleGrowth') or 0.05) * 100, 'gasGrowth': float(inputs.get('gasGrowth') or 0.03) * 100,
         'reliability': str(inputs.get('reliability') or 'R0'), 'weights': {k: float(w.get(k, 33)) for k in ('speed', 'cost', 'environment')},
         'ownership': None, 'appetite': 'Medium', 'localNotes': '', 'stress': stress, 'excluded': list(inputs.get('excluded') or []),
         'capexUpliftPct': 0, 'gridAdderExtra': float(inputs.get('gridAdderExtra') or 0), 'template': agents.get('template')}
    c['_stress'], c['_stressLabel'] = (stress, stress.get('source', 'your stress events')) if stress else ({'mode': 'packet'}, 'packet events (7 ERCOT, 2023)')
    tid = uuid.uuid4().hex[:10]
    th = {'id': tid, 'mode': 'run', 'runJobId': jid, 'createdAt': time.time(), 'thesis': c,
          'scenarios': {'run': {'key': 'run', 'label': 'Your optimiser run', 'payload': inputs, 'jobId': jid, 'status': 'running'}},
          'agents': {'optimisation': {'status': 'running', 'detail': 'Optimising your run', 'name': 'Optimisation agent'},
                     'financial': {'status': 'queued', 'detail': 'Waiting to start', 'name': 'Financial agent'},
                     'sourcing': {'status': 'queued', 'detail': 'Waiting to start', 'name': 'Site sourcing agent'},
                     'moratorium': {'status': 'queued', 'detail': 'Waiting to start', 'name': 'Policy agent'}},
          'financial': None, 'moratorium': None}
    with LOCK:
        THESES[tid] = th
        m = _runmap(); m[jid] = tid; RUNMAP.write_text(json.dumps(m))
    _save(th)
    threading.Thread(target=_run_financial, args=(th,), daemon=True).start()
    threading.Thread(target=_run_moratorium, args=(th,), daemon=True).start()
    threading.Thread(target=_watch_run, args=(th,), daemon=True).start()
    return 200, {'id': tid}


def _watch_run(th):
    a, s = th['agents']['optimisation'], th['scenarios']['run']
    v = _wait(s['jobId'])
    s['status'] = 'done' if v and v.get('status') == 'done' else 'failed'
    r = (v or {}).get('result') or {}
    a['status'] = 'done' if s['status'] == 'done' else 'failed'
    a['detail'] = ('Found a feasible portfolio' if r.get('status') == 'OK' else 'No feasible portfolio for these inputs' if r.get('status') == 'INFEASIBLE' else (v or {}).get('error') or 'The run did not finish')
    _save(th)


def list_theses():
    out = []
    for f in OUT.glob('*.json'):
        try:
            th = _load(f.stem)
        except Exception:
            continue
        if th:
            out.append({'id': th['id'], 'createdAt': th['createdAt'], 'title': th['thesis']['title'], 'questions': th['thesis']['questions'],
                        'agents': {k: a['status'] for k, a in th['agents'].items()}})
    out.sort(key=lambda x: x['createdAt'], reverse=True)
    return out
