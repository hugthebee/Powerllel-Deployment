"""HTTP server for the Powerllel website, running the GRIDTECH optimiser (team_app/gridtech_model.py).

    python team_app/gridtech_server.py [--port 8770]

The page sends the answers to the optimiser's questions (Q2-Q10 of Faraday.ipynb);
a run sizes the portfolio, dispatches all 87,600 hours, scores it and draws the two dashboards.
A run takes minutes, so it is a background job the page polls. One run at a time.

API  (JSON)
    GET  /api/health
    GET  /api/meta                         resource catalogue, reliability presets, defaults and limits for the form
    POST /api/run                          the answers (see _parse; optional 'alternative': {ofJobId, ofTag, title, changes, original}) -> {jobId} | 400 {error, errors: {field: message}} | 409 {error, jobId}
    GET  /api/runs                         every saved run, newest first (summary rows for the Previous runs page)
    GET  /api/job/<id>                     status, phase, live solver progress, log, the inputs, and the result when done
    GET  /api/job/<id>/<file>              dashboard.png | years.png | portfolio.json | dispatch.csv | summary.json | results.json
    GET  /api/job/<id>/compare             saved runs with the same inputs except the stress events (to show how the portfolio changes)
    GET  /api/stress                       stress events: the packet's, and the bundled 2023 catalog by interconnection and state
    POST /api/stress/parse                 {name, data: base64 of a .csv / .xlsx} -> {rows, problems}: events read from an uploaded file
Also serves the built website from faraday_web/dist so the page and the model share one address.
"""
import argparse
import base64
import dataclasses
import io
import json
import math
import mimetypes
import re
import sys
import threading
import time
import traceback
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_agents as TA  # noqa: E402  (the thesis: four agents in the background)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gridtech_model as G  # noqa: E402
import gridtech_diagnose as D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import stress_data as SD  # noqa: E402

DIST = HERE.parent / 'faraday_web' / 'dist'
OUT = HERE.parent / 'outputs' / 'web'
EDITABLE = ['capex_usd_per_kw', 'fom_usd_per_kw_year', 'payment_usd_per_mwh', 'available_month', 'max_mw']   # Q8 fields
FIELD_LABEL = {'capex_usd_per_kw': 'CAPEX $/kW', 'fom_usd_per_kw_year': 'fixed O&M $/kW-yr', 'payment_usd_per_mwh': 'contract price or variable O&M $/MWh',
               'available_month': 'available from month', 'max_mw': 'maximum MW'}
FILES = {'dashboard.png': 'image/png', 'years.png': 'image/png', 'portfolio.json': 'application/json',
         'dispatch.csv': 'text/csv', 'charts.json': 'application/json', 'summary.json': 'application/json', 'results.json': 'application/json',
         'cost_by_source.csv': 'text/csv', 'co2_by_source.csv': 'text/csv', 'co2_by_source_annual_t.csv': 'text/csv'}
SOURCE_FILES = ('cost_by_source.csv', 'co2_by_source.csv', 'co2_by_source_annual_t.csv')
CHARTS_VERSION = 2            # charts.json written by an older version is rebuilt (v2 adds `sources`)

JOBS = {}
RUN_LOCK = threading.Lock()
CURRENT = {'id': None}


# ------------------------------------------------------------------ live solver progress
# The notebook's _solve_highs creates highspy.Highs() at call time; this subclass adds HiGHS callbacks so the page can show
# the sizing MIP converging (best portfolio cost, lower bound, gap, nodes) and the dispatch advancing year by year.
# The model code itself is not changed.
import highspy  # noqa: E402

TRACK = {'job': None}
PROGRESS_LOCK = threading.Lock()      # the dispatch years now solve in parallel threads: the 'model year k of n' counter must not race


class _TrackedHighs(highspy.Highs):
    def run(self):
        job = TRACK['job']
        pr = job['progress'] if job else None
        if pr is None:
            return super().run()
        if pr['step'] in ('prepare', 'size') and pr['mip'].get('stage') is None:
            return super().run()                                # the capacity pre-check (gridtech_diagnose), not the sizing
        if pr['step'] in ('prepare', 'size'):
            pr['step'] = 'size'
            m = pr['mip']
            m.update(variables=int(self.getNumCol()), constraints=int(self.getNumRow()))
            m['startedAt'] = m['startedAt'] or time.time()
            # sizing is three solves (gridtech_model.size): the relaxation's cost is the lower bound for every portfolio; the
            # restricted search only finds portfolios (its own bound covers the small box, not every portfolio); the polish does both
            stage, offset, nodes0 = m.get('stage'), m.get('offset', 0.0), m.get('nodesBase', 0)
            last = [0.0]

            def cb(e):
                d = e.data_out
                now = time.time()
                improving = e.callback_type == highspy.cb.HighsCallbackType.kCallbackMipImprovingSolution
                # portfolio costs only from improving-solution events (the other events can carry placeholder values early on)
                if improving and math.isfinite(d.objective_function_value) and d.objective_function_value > 0 \
                        and (m['best'] is None or d.objective_function_value < m['best']):
                    m['best'] = float(d.objective_function_value)
                if stage == 'polish' and d.mip_node_count > 0 and math.isfinite(d.mip_dual_bound) and (m['best'] is None or d.mip_dual_bound <= m['best']):
                    m['bound'] = max(float(d.mip_dual_bound), m['bound'] or -math.inf)
                if m['best'] is not None and m['bound'] is not None:
                    m['gap'] = max(0.0, (m['best'] - m['bound']) / abs(m['best']))
                m['nodes'] = nodes0 + int(d.mip_node_count)
                m['seconds'] = round(offset + d.running_time, 1)
                m['updatedAt'] = now
                if improving:
                    m['solutions'] += 1
                if improving or now - last[0] >= 1.0:                   # a point a second for the convergence chart
                    last[0] = now
                    m['history'].append([round(offset + d.running_time, 1), m.get('best'), m.get('bound')])
                    if len(m['history']) > 900:
                        del m['history'][1:len(m['history']) // 2:2]    # thin the older half, keep the first point
            for name in ('cbMipInterrupt', 'cbMipImprovingSolution', 'cbMipLogging'):
                getattr(self, name).subscribe(cb)
            return super().run()
        if pr['step'] == 'dispatch':
            out = super().run()
            with PROGRESS_LOCK:
                pr['dispatch']['done'] = min(pr['dispatch']['done'] + 1, pr['dispatch']['total'])   # carbon-ceiling passes re-solve years
                if pr['dispatch']['done'] >= pr['dispatch']['total']:
                    pr['step'] = 'score'
            return out
        return super().run()


highspy.Highs = _TrackedHighs


# ------------------------------------------------------------------ answers -> model objects
class BadInput(ValueError):
    """Every problem with the answers, keyed by the form field it belongs to."""
    def __init__(self, errors):
        self.errors = errors
        first = next(iter(errors.values()))
        super().__init__(first if len(errors) == 1 else f'{len(errors)} inputs need fixing. First: {first}')


def scaled_max(rid, peak):
    """Resource maximum after scale_catalog, as the cell shows it in Q8."""
    row = G.OPT.loc[rid]
    return float(max(row.block_mw, np.floor(row.max_mw * peak / 75 / row.block_mw + 1e-9) * row.block_mw))


HORIZON_LAST_MONTH = 119            # the model runs development months 0-119


class _Check:
    """Collects one message per field. Rules marked [nb] are the notebook's own question checks."""
    def __init__(self):
        self.errors = {}

    def num(self, key, v, label, lo=None, hi=None, integer=False, lo_open=False, required=True):
        if v is None or (isinstance(v, str) and v.strip() == ''):
            if required:
                self.errors[key] = f'{label}: enter a value'
            return None
        if isinstance(v, bool):
            self.errors[key] = f'{label}: must be a number'
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            self.errors[key] = f'{label}: could not read {v!r}, enter a number'
            return None
        if not math.isfinite(x):
            self.errors[key] = f'{label}: must be a number'
            return None
        if integer and x != int(x):
            self.errors[key] = f'{label}: must be a whole number'
            return None
        rng = (f'{lo:g} - {hi:g}' if hi is not None else (f'> {lo:g}' if lo_open else f'>= {lo:g}')) if lo is not None else f'<= {hi:g}'
        if lo is not None and (x <= lo if lo_open else x < lo) or hi is not None and x > hi:
            self.errors[key] = f'{label}: enter {rng}'
            return None
        return int(x) if integer else x


# ------------------------------------------------------------------ stress events
# The model's hourly data is one reference year (2023) on US Central time: hour 0 = 2023-01-01 00:00 local. The page works in
# that clock (start 'YYYY-MM-DDTHH:MM', duration in hours); the model takes the events in UTC (gridtech_model.Custom.stress_events).
TZ = ZoneInfo('America/Chicago')
TZ_LABEL = 'US Central time (the clock of the model\'s hourly data)'
REF_START = datetime(2023, 1, 1, tzinfo=TZ)
REF_END = datetime(2024, 1, 1, tzinfo=TZ)
STRESS_LIMITS = {'events': 60, 'hours': 168, 'totalHours': 336}      # per event: one week; per year: two weeks of stress hours
LOCAL_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})')


def _local_to_utc(s):
    """'2023-08-24T15:00' on the model's clock -> aware UTC datetime, or None if unreadable."""
    m = LOCAL_RE.fullmatch(str(s or '').strip())
    if not m:
        return None
    try:
        return datetime(*map(int, m.groups()), tzinfo=TZ).astimezone(timezone.utc)
    except ValueError:
        return None


def _utc_to_local(ts):
    return pd.Timestamp(ts).tz_convert(TZ).strftime('%Y-%m-%dT%H:%M')


def _iso_utc(dt):
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _stress_hours_count(events):
    return int(G.stress_hours(events)[0].sum()) if events else 0


def _parse_stress(se, c):
    """The page's stressEvents -> (model events in UTC or None for the packet's, label). Problems go to c.errors."""
    if not se or se.get('mode', 'packet') == 'packet':
        return None, 'packet'
    if se.get('mode') != 'custom':
        c.errors['stressEvents'] = 'Stress events: choose the packet events or your own events'
        return None, 'packet'
    rows = se.get('events')
    if not isinstance(rows, list):
        c.errors['stressEvents'] = 'Stress events: the list of events is missing'
        return None, 'packet'
    if len(rows) > STRESS_LIMITS['events']:
        c.errors['stressEvents'] = f'Stress events: at most {STRESS_LIMITS["events"]} events (you have {len(rows)})'
        return None, 'packet'
    events = []
    before = set(c.errors)
    for i, e in enumerate(rows):
        e = e if isinstance(e, dict) else {}
        n = f'Event {i + 1}' + (f' ({str(e.get("id"))[:40]})' if e.get('id') else '')
        start = _local_to_utc(e.get('start'))
        if start is None:
            c.errors[f'stress.{i}.start'] = f'{n}, start: use a date and time like 2023-08-24T15:00'
        elif not (REF_START <= start < REF_END):
            c.errors[f'stress.{i}.start'] = f'{n}, start: must be in 2023, the model\'s reference year (events repeat every model year)'
            start = None
        hours = c.num(f'stress.{i}.hours', e.get('hours'), f'{n}, duration (hours)', 0, STRESS_LIMITS['hours'], lo_open=True)
        imp = c.num(f'stress.{i}.importPct', e.get('importPct'), f'{n}, import allowed (%)', 0, 100)
        notice = c.num(f'stress.{i}.noticeMin', e.get('noticeMin'), f'{n}, notice (minutes)', 0, 10080)
        if start is None or hours is None:
            continue
        end = start + timedelta(minutes=round(hours * 60))         # whole minutes: a rounded duration (2.57 h) must not spill into the next hour
        if end > REF_END:
            c.errors[f'stress.{i}.hours'] = f'{n}: ends after 31 Dec 2023 24:00; shorten it or split it into two events'
            continue
        if imp is not None and notice is not None:
            events.append({'id': str(e.get('id') or f'event_{i + 1}')[:60], 'start_utc': _iso_utc(start), 'end_utc': _iso_utc(end),
                           'import_fraction': imp / 100, 'notice_minutes': notice})
    # overlapping events are allowed (real ones do, e.g. MISO and SPP in August 2023): in shared hours the stricter
    # import fraction and the shorter notice apply (gridtech_model.stress_hours)
    if set(c.errors) == before and events:
        total = _stress_hours_count(events)
        if total > STRESS_LIMITS['totalHours']:
            c.errors['stressEvents'] = (f'Stress events cover {total} hours a year; the limit is {STRESS_LIMITS["totalHours"]} '
                                        '(sizing keeps every stress period at full detail, so more hours make the solve much slower)')
    label = str(se.get('source') or 'your events')[:120]
    return events, label


MAX_SAMPLED_HOURS = 20000      # packet events 11,400; every bundled catalog region 11,000-14,300


def _check_sample_size(inp, sc):
    """Sizing keeps each month's stress period (first to last stress hour, +-12 h) at full detail. Events spread across a month
    make that most of the month, so a few short events can triple the sizing problem: refuse runs that would be far slower."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)      # a month kept whole has no hours left to sample (harmless)
        n = len(inp.sample(sc.days_per_month)[0])
    if n > MAX_SAMPLED_HOURS:
        raise BadInput({'stressEvents': f'These events are spread across so many days of their months that sizing would have to keep {n:,} hours at full detail '
                                         f'(the limit is {MAX_SAMPLED_HOURS:,}; the packet events need about 11,400). Keep each month\'s events within a few days '
                                         'of each other, or use fewer months.'})


def _catalog_row(r, default_include=True):
    """One catalog event on the page's clock."""
    return {'id': str(r.event_id), 'start': _utc_to_local(r.start_utc), 'hours': round(float(r.duration_hours), 2),
            'importPct': round(float(r.import_fraction) * 100, 2), 'noticeMin': float(r.notice_minutes),
            'operator': str(getattr(r, 'operator', '') or ''), 'type': str(getattr(r, 'event_type', '') or ''),
            'confidence': str(getattr(r, 'timing_confidence', '') or ''), 'notes': str(getattr(r, 'notes', '') or ''),
            'include': default_include}


def _default_include(r):
    """Off by default: low-confidence timings, and CAISO Restricted Maintenance Operations (precautionary, not a shortfall)."""
    return str(getattr(r, 'timing_confidence', '')).lower() != 'low' and 'restricted maintenance' not in str(getattr(r, 'event_type', '')).lower()


_CATALOG = {}


def stress_meta():
    """The packet's events and the bundled 2023 catalog (team_app/stress_data/), on the page's clock."""
    if 'v' not in _CATALOG:
        cat = SD.Catalog()
        ev = pd.read_csv(G.PACKET / 'data' / 'stress_events.csv')
        packet = [{'id': str(r.event_id), 'start': _utc_to_local(r.start_utc), 'hours': float(r.duration_hours),
                   'importPct': float(r.import_fraction) * 100, 'noticeMin': 30.0, 'operator': 'ERCOT', 'type': 'Conservation appeal (voluntary)',
                   'confidence': 'high', 'notes': 'Packet event', 'include': True} for r in ev.itertuples()]
        areas = [{'area': a['area'], 'operators': a['operators'],
                  'events': [_catalog_row(r, _default_include(r)) for r in cat.events_of(a['area']).itertuples()]} for a in cat.areas()]
        states = [{'state': s['state'], 'postal': s['postal'], 'area': s['areas'][0]['area']} for s in cat.state_rows()]
        _CATALOG['v'] = {'timezone': TZ_LABEL, 'year': 2023, 'limits': STRESS_LIMITS, 'relocationNoticeMin': float(G.ck.response['relocation_notice_minutes']),
                         'packet': packet, 'areas': areas, 'states': sorted(states, key=lambda s: s['state']),
                         'source': 'team_app/stress_data/stress_events_all_interconnections_2023.csv'}
    return _CATALOG['v']


def parse_stress_upload(name, data):
    """Events from an uploaded .csv / .xlsx. Two layouts: the page's (start, hours, import_pct, notice_min on the model's clock)
    or the catalog's (start_utc, end_utc, import_fraction, optional notice_utc)."""
    try:
        raw = base64.b64decode(data or '', validate=True)
    except ValueError:
        return {'rows': [], 'problems': ['The file could not be read.']}
    if not raw:
        return {'rows': [], 'problems': ['The file is empty.']}
    try:
        tables = SD._tables(raw)
    except ImportError:
        return {'rows': [], 'problems': [f'{name}: this server cannot read Excel files (its openpyxl package is missing or too old). Save the sheet as CSV and upload that.']}
    except Exception as exc:
        return {'rows': [], 'problems': [f'{name}: not a readable CSV or Excel file ({type(exc).__name__}).']}
    rows, problems = [], []
    for df in tables:
        df = df.rename(columns=lambda x: str(x).strip().lower())
        cols = set(df.columns)
        if {'start_utc', 'end_utc', 'import_fraction'} <= cols:
            ev, bad = SD._events(df)
            problems += bad
            rows += [_catalog_row(r) for r in ev.itertuples()]
        elif {'start', 'hours', 'import_pct'} <= cols:
            for j, r in enumerate(df.to_dict('records')):
                rows.append({'id': str(r.get('id') or f'event_{len(rows) + 1}'), 'start': str(r.get('start', '')).strip().replace(' ', 'T')[:16],
                             'hours': str(r.get('hours', '')).strip(), 'importPct': str(r.get('import_pct', '')).strip(),
                             'noticeMin': str(r.get('notice_min', '') if r.get('notice_min', '') != '' else 30).strip(),
                             'operator': '', 'type': '', 'confidence': '', 'notes': f'from {name}', 'include': True})
    if not rows and not problems:
        problems.append('No events found. Use columns start, hours, import_pct, notice_min (start like 2023-08-24 15:00, US Central), '
                        'or start_utc, end_utc, import_fraction.')
    return {'rows': rows[:STRESS_LIMITS['events'] * 2], 'problems': problems}


def _parse(p):
    """Validate the page's answers (the notebook's rules plus a few for inputs it leaves unchecked); return the Scenario + labels.
    Raises BadInput with every problem found, keyed by field."""
    c = _Check()
    # Q2 [nb] 5-2000 MW
    peak = c.num('peakMw', p.get('peakMw'), 'Peak load (MW)', 5, 2000)
    # Q3 [nb] calendar date YYYY-MM, optional (labels only); opening month whole number 14-109
    cal = p.get('calendarStart')
    cal = str(cal).strip() if cal not in (None, '') else None
    if cal is not None and not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', cal):
        c.errors['calendarStart'] = 'Calendar date of month 0: use the format YYYY-MM with a month 01-12, e.g. 2027-01'
    start = c.num('start', p.get('start'), 'Opening month', 14, 109, integer=True)

    # Q4 [nb] steps between month 18 and 24, MW >= 0; packet schedule only for 75 MW
    ic = p.get('interconnection') or {}
    mode = ic.get('mode')
    grid_schedule, ic_month = None, None
    if mode == 'packet':
        if peak is not None and peak != 75:
            c.errors['interconnection'] = 'The packet schedule is only for a 75 MW campus: set the peak to 75 MW or enter steps'
    elif mode == 'month':
        ic_month = c.num('interconnection', ic.get('month'), 'Interconnection month', 18, 24, integer=True)
    elif mode == 'steps':
        steps = ic.get('steps') or []
        if not steps:
            c.errors['interconnection'] = 'Add at least one interconnection step'
        grid_schedule, seen = [], set()
        for i, st in enumerate(steps):
            m = c.num(f'step.{i}.month', st.get('month'), f'Step {i + 1} month', 18, 24, integer=True)
            mw = c.num(f'step.{i}.mw', st.get('mw'), f'Step {i + 1} import (MW)', 0, 2000)      # [nb] >= 0; added: <= 2000 like the peak
            if m is not None and m in seen:
                c.errors[f'step.{i}.month'] = f'Step {i + 1}: month {m} is already used by an earlier step'   # added: one import level per month
            seen.add(m)
            grid_schedule.append((m, mw))
    else:
        c.errors['interconnection'] = 'Choose steps, one month, or the packet schedule'

    # Q5 [nb] R0-R5
    rel = str(p.get('reliability', '')).upper()
    if rel not in G.PRESETS:
        c.errors['reliability'] = 'Choose a reliability standard: R0, R1, R2, R3, R4 or R5'
    # Q6 [nb] > 0, optional
    co2 = c.num('co2Cap', p.get('co2Cap'), 'Carbon ceiling (kg CO2/MWh)', 0, None, lo_open=True, required=False)
    # Q7 [nb] -0.2 to 0.3 (the page sends fractions)
    wg = c.num('wholesaleGrowth', p.get('wholesaleGrowth'), 'Wholesale price growth', -0.2, 0.3)
    gg = c.num('gasGrowth', p.get('gasGrowth'), 'Henry Hub gas price growth', -0.2, 0.3)
    for key in ('wholesaleGrowth', 'gasGrowth'):
        if key in c.errors and 'enter -0.2 - 0.3' in c.errors[key]:
            c.errors[key] = c.errors[key].replace('enter -0.2 - 0.3', 'enter -20% to 30% a year')

    # Q8 [nb] every value >= 0, available month a whole number; added: month inside the horizon, max >= one block
    resources, eff_max = {}, {}
    for rid, ov_in in (p.get('resources') or {}).items():
        if rid not in G.RES:
            c.errors[f'res.{rid}'] = f'Unknown resource {rid!r}'
            continue
        row = G.OPT.loc[rid]
        ov = {}
        for col in EDITABLE:
            v = (ov_in or {}).get(col)
            if v in (None, '') or (col in ('capex_usd_per_kw', 'fom_usd_per_kw_year') and row.location == 'offsite'):
                continue                                   # blank keeps the packet value; contracts have a price only (as in the notebook)
            key, lab = f'res.{rid}.{col}', f'{row.label}, {FIELD_LABEL[col]}'
            if col == 'available_month':
                x = c.num(key, v, lab, 0, HORIZON_LAST_MONTH, integer=True)
            elif col == 'max_mw':
                x = c.num(key, v, lab, 0)
                if x is not None and x < row.block_mw:
                    c.errors[key] = f'{lab}: must be at least one block ({row.block_mw:g} MW)'
                    x = None
            else:
                x = c.num(key, v, lab, 0)
            if x is None or peak is None:
                continue
            cur = scaled_max(rid, peak) if col == 'max_mw' else (int(row[col]) if col == 'available_month' else float(row[col]))
            if x != cur:
                ov[col] = x
        if 'max_mw' in ov:
            ov['max_mw'] = float(max(row.block_mw, np.floor(ov['max_mw'] / row.block_mw) * row.block_mw))
        if ov:
            resources[rid] = ov
    for rid in G.RES:
        eff_max[rid] = resources.get(rid, {}).get('max_mw', scaled_max(rid, peak) if peak is not None else None)

    # Q9 [nb] known resource, MW >= 0, empty = none; added: not above that resource's maximum (it could never be built)
    mins = {}
    for rid, mw in (p.get('minCaps') or {}).items():
        if rid not in G.RES:
            c.errors[f'min.{rid}'] = f'Unknown resource {rid!r}; use one of {", ".join(G.RES)}'
            continue
        x = c.num(f'min.{rid}', mw, f'{G.OPT.loc[rid, "label"]}, minimum MW', 0, required=False)
        if x is None or x == 0:
            continue
        if eff_max[rid] is not None and x > eff_max[rid] + 1e-9:
            c.errors[f'min.{rid}'] = f'{G.OPT.loc[rid, "label"]}, minimum MW: {x:g} is above its maximum of {eff_max[rid]:g} MW'
            continue
        mins[rid] = x

    # Q10 [nb] yes / no
    staged = p.get('staged')
    if not isinstance(staged, bool):
        c.errors['staged'] = 'Choose upfront or year by year'

    # stress events: the packet's seven ERCOT events, or the customer's own (on the model's clock, converted to UTC)
    stress, stress_label = _parse_stress(p.get('stressEvents'), c)
    # added for the thesis agents: the grid connection slips by N months (example 5), no Tier 2 interruption (example 4)
    delay = c.num('gridDelay', p.get('gridDelay') if p.get('gridDelay') not in (None, '') else 0, 'Grid connection slip (months)', 0, 36, integer=True)
    no_t2 = p.get('noTier2', False)
    grid_extra = c.num('gridAdderExtra', p.get('gridAdderExtra') if p.get('gridAdderExtra') not in (None, '') else 0, 'Extra grid cost ($/MWh)', 0, 500)
    land = c.num('landAcres', p.get('landAcres'), 'Onsite land for equipment (acres)', 0.5, 100000, required=False)   # added: a proposed site's land (None = 8 acres x peak/75)
    excluded = [str(r) for r in (p.get('excluded') or [])]           # added for the thesis: resources switched off (not available)
    for r in excluded:
        if r not in G.RES:
            c.errors['excluded'] = f'Unknown resource {r!r}'
    if not isinstance(no_t2, bool):
        c.errors['noTier2'] = 'No Tier 2 interruption: choose yes or no'

    if c.errors:
        raise BadInput(c.errors)
    custom = G.Custom(peak_mw=peak, grid_schedule=grid_schedule, interconnection_month=ic_month, calendar_start=cal,
                      resources=resources, wholesale_growth=wg, gas_growth=gg, stress_events=stress, tier2_max_fraction=0.0 if no_t2 else None, grid_adder_extra=float(grid_extra) if grid_extra else None, land_acres=land)
    sc = G.Scenario(start=start, rel=G.PRESETS[rel], custom=custom, co2_cap_kg_mwh=co2, min_caps=mins or None, staged=staged, grid_delay=int(delay or 0), forbid=tuple(sorted(r for r in excluded if r in G.RES)))
    return sc, dict(peak=peak, rel=rel, start=start, cal=cal, co2=co2, mins=mins, staged=staged, resources=resources,
                    stress=stress, stressLabel=stress_label, gridDelay=int(delay or 0), noTier2=no_t2, excluded=sorted(set(excluded)), gridAdderExtra=float(grid_extra or 0), landAcres=land)


def _effective(inp, start):
    """What show_inputs() prints: the inputs after the customer's overrides."""
    cal = inp.cu.calendar_start
    cols = ['label', 'location', 'available_month', 'block_mw', 'max_mw', 'capex_usd_per_kw', 'fom_usd_per_kw_year', 'life_years',
            'payment_usd_per_mwh', 'heat_rate_mmbtu_per_mwh_hhv', 'co2_t_per_mwh', 'land_acres_per_mw']
    rows = []
    for r in G.RES:
        d = {c: (inp.opt.loc[r, c].item() if hasattr(inp.opt.loc[r, c], 'item') else inp.opt.loc[r, c]) for c in cols}
        d['id'] = r
        d['fixed_usd_per_mw_year'] = float(inp.fixed_per_mw[r] / max(inp.mask[r].sum(), 1) * 8760)
        rows.append(d)
    return {'peakMw': inp.peak, 'annualEnergyGwh': float(inp.load[-8760:].sum() / 1e3), 'landAcres': float(inp.land),
            'remoteMw': float(inp.remote),
            'gridSchedule': [{'month': m, 'label': G.month_to_date(m, cal), 'mw': mw} for m, mw in inp.grid_schedule],
            'opening': start, 'openingLabel': G.month_to_date(start, cal),
            'fullDemand': start + 10, 'fullDemandLabel': G.month_to_date(start + 10, cal), 'resources': rows}


def _stress_summary(inp, caps, info):
    """How the stress events bite: stress hours, the hour that needs most onsite gas, and the gas built."""
    h0 = inp.h0
    need = inp.load - inp.flex - inp.cap_eff                      # what onsite gas must cover (packet rule R0)
    out = {'label': info['stressLabel'], 'custom': info['stress'] is not None,
           'events': len(info['stress']) if info['stress'] is not None else len(stress_meta()['packet']),
           'hoursPerYear': int(inp.active[:8760].sum()), 'worst': None}
    act = np.nonzero(inp.active[h0:])[0] + h0
    if act.size:
        h = int(act[np.argmax(need[act])])
        ts = pd.Timestamp(G.ck.hourly['timestamp_utc'].iloc[int(inp.c['ref'][h])]).tz_convert(TZ)
        out['worst'] = {'modelHour': h, 'modelMonth': int(inp.month[h]), 'modelYear': int(inp.year[h]) + 1,
                        'when': f'{ts.day} {ts:%b %H:%M}',      # not '%-d': Windows strftime has no '-' flag
                        'loadMw': float(inp.load[h]), 'flexMw': float(inp.flex[h]), 'importMw': float(inp.cap_eff[h]), 'gasNeedMw': float(max(need[h], 0.0))}
    gas = {r: float(caps.get(r, 0)) for r in G.GAS}
    unit = G.Reliability().unit_mw
    out['gasMw'] = sum(gas.values())
    out['largestUnitMw'] = max([unit[r] for r, v in gas.items() if v > 0], default=0.0)
    return out


# ------------------------------------------------------------------ jobs
class _Log(io.TextIOBase):
    """stdout of the model during a run -> the job's log lines."""
    def __init__(self, job):
        self.job, self.buf = job, ''

    def write(self, s):
        self.buf += s
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            if line.strip():
                self.job['log'].append(line.rstrip())
        return len(s)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


SOURCE_IDS = list(G.RES) + ['grid', 'tier1_relocated', 'tier2_curtailed']
_COST_LABEL = {**{r: G.BAR_LABELS[r] for r in G.RES}, 'grid': 'Merchant grid', 'tier1_relocated': 'Tier 1 relocation',
               'tier2_curtailed': 'Tier 2 interruption'}
_CO2_LABEL = {**{r: G.BAR_LABELS[r] for r in G.RES}, 'grid': 'Merchant grid', 'tier1_relocated': 'Tier 1 relocated (remote)'}


def by_source(cdf, edf, eann):
    """Per source: 10-year operational CO2 (t), intensity (kg/MWh), CO2 by model year, upfront capex ($/kW rate and $) and 10-year
    life-cycle cost ($) -- the tables of F1.ipynb (gridtech_model.cost_breakdown / emissions_by_source), unchanged."""
    c = cdf.set_index('option_id'); e = edf.set_index('source')
    ny = len(eann)
    rows = []
    for sid in SOURCE_IDS:
        x = c.loc[sid]
        lab = _CO2_LABEL.get(sid)
        has = lab in e.index
        is_res = sid in G.RES
        rows.append({
            'id': sid, 'label': _COST_LABEL[sid], 'mw': float(x.mw) if is_res else None,
            'mwh': float(x.mwh_delivered),
            'co2_t': float(e.loc[lab, 'co2_t']) if has else 0.0,
            'kg_per_mwh': float(e.loc[lab, 'kg_co2_per_mwh']) if has else 0.0,
            'co2_by_year': [float(v) for v in eann[lab].to_numpy()] if lab in eann.columns else [0.0] * ny,
            'capex_usd_per_kw': float(x.capex_usd_per_kw) if is_res else None,
            'capex_usd': float(x.upfront_capex_usd),
            'lcc_usd': float(x.lifecycle_usd),
            'lcc_usd_per_mwh': float(x.lifecycle_usd_per_mwh) if np.isfinite(x.lifecycle_usd_per_mwh) else None,
            'lcc_parts': {k: float(x[k]) for k in G.COST_PARTS},
        })
    return rows


def _png(fig, path, dpi):
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


SIZING_PHASE = {'relax': 'Sizing, step 1 of 3: the relaxed problem (a lower bound on the cost)',
                'restrict': 'Sizing, step 2 of 3: whole blocks close to the relaxed solution',
                'polish': 'Sizing, step 3 of 3: full search, starting from the best portfolio found'}


def _sizing_step(job, step, phase, info):
    """Progress callback of gridtech_model.size: which of the three sizing solves is running, and its outcome."""
    m = job['progress']['mip']
    now = time.time()
    if phase == 'start':
        m['stage'] = step
        job['phase'] = SIZING_PHASE.get(step, 'Sizing the portfolio')
        m['updatedAt'] = now
        return
    m['steps'].append(info)
    m['offset'] = round(m.get('offset', 0.0) + info['seconds'], 1)
    m['nodesBase'] = m['nodes']
    m['seconds'] = m['offset']
    if info['objective'] is not None:
        if step == 'relax':
            m['bound'] = info['objective']
        elif step == 'polish' and info.get('bound') is not None:
            m['bound'] = max(m['bound'] or -math.inf, info['bound'])
        if step != 'relax' and (m['best'] is None or info['objective'] < m['best']):
            m['best'] = info['objective']
    if m['best'] is not None and m['bound'] is not None:
        m['gap'] = max(0.0, (m['best'] - m['bound']) / abs(m['best']))
    m['history'].append([m['offset'], m['best'], m['bound']])
    m['updatedAt'] = now
    print(f"Sizing {step}: {info['status']} in {info['seconds']:.0f} s" + (f", cost ${info['objective'] / 1e6:,.2f}M" if info['objective'] is not None else ''))


def _work(job, sc, info):
    out = Path(job['dir'])
    t0 = time.time()
    pr = job['progress']
    TRACK['job'] = job
    try:
        with redirect_stdout(_Log(job)):
            job['phase'] = 'Checking the capacity rules'
            if not D.structurally_feasible(sc):
                # no portfolio can meet the capacity rules, so the sizing solve could only end infeasible: explain it now instead
                print('Capacity rules cannot all be met: sizing skipped')
                _infeasible(job, 'not run: the capacity rules cannot all be met', skipped=True)
                return
            job['phase'] = 'Preparing hourly inputs and representative days'
            sized = G.size(sc, on_step=lambda step, phase, info: _sizing_step(job, step, phase, info))
            inp, lp, st = sized
            pr['mip']['status'] = st
            print(f'Sizing: {st} in {lp.seconds:.0f} s')
            if lp.x is None:
                _infeasible(job, st)
                return
            pr['dispatch']['total'] = sum(1 for y in range(10) if max(inp.h0, y * 8760) < (y + 1) * 8760)
            pr['step'] = 'dispatch'
            job['phase'] = 'Dispatching all 87,600 hours, one model year at a time'
            res, d = G.run(sc, out=out, sized=sized)
            if res.get('status') == 'INFEASIBLE':            # the carbon ceiling cannot be met over all 87,600 hours
                _infeasible(job, res.get('sizing_status'))
                return
            if res.get('status') != 'OK':
                job['result'] = {'status': res.get('status'), 'sizingStatus': res.get('sizing_status'), 'message': 'The run did not finish.'}
                job['status'] = 'done'
                return
            pr['step'] = 'plots'
            job['phase'] = 'Drawing the dashboards'
            title = f"{info['rel']} {info['peak']:g}MW"
            _png(G.plot_results(res, d, title=title, save=False, show=False), out / 'dashboard.png', 130)
            _png(G.plot_years(res, d, title=title, save=False, show=False), out / 'years.png', 110)
            for name in ('check_results.json', 'custom_results.json'):
                if (out / name).exists():
                    (out / name).replace(out / 'results.json')
        inp = res['_inp']
        cal = inp.cu.calendar_start
        start = res['initial_service_month']
        caps = res['capacities']
        avail = {r: max(start, int(inp.opt.loc[r, 'available_month'])) for r in G.RES}
        years = [y for y in range(10) if inp.load[y * 8760:(y + 1) * 8760].sum() > 0]
        job['result'] = _clean({
            'status': 'OK', 'checker': res['checker'], 'sizingStatus': res['sizing_status'], 'seconds': round(time.time() - t0),
            'reliability': info['rel'], 'reliabilityText': G.PRESET_TEXT[info['rel']], 'reliabilityLabel': res['reliability'],
            'staged': res['staged'], 'peakMw': inp.peak,
            'opening': start, 'openingLabel': G.month_to_date(start, cal),
            'fullDemand': res['full_demand_month'], 'fullDemandLabel': G.month_to_date(res['full_demand_month'], cal),
            'co2Cap': info['co2'], 'co2CapMet': res.get('co2_cap_met'), 'co2Price': res.get('co2_price_usd_per_t'), 'minCaps': info['mins'],
            'sizingSteps': res.get('sizing_steps'), 'sizingSeconds': res.get('sizing_s'),
            'kpis': {k: res[k] for k in G.KEYS if k in res},
            'capacities': [{'id': r, 'label': G.BAR_LABELS[r], 'mw': caps[r], 'from': avail[r], 'fromLabel': G.month_to_date(avail[r], cal)}
                           for r in G.RES],
            'capacitiesByYear': [{'year': y + 1, 'label': G.month_to_date(12 * y, cal), 'caps': res['capacities_by_year'][y]} for y in years],
            'annual': res.get('annual', []),
            'bySource': by_source(res['_cost_df'], res['_co2_df'], res['_co2_annual']),
            'upfrontCapexUsd': res.get('upfront_capex_usd'),
            'battery': {'rule': res.get('battery_rule'), 'fixedMw': res.get('battery_fixed_mw'), 'note': res.get('battery_note'),
                        'mw': caps.get('battery_4h')},
            'dispatchSeconds': res.get('dispatch_s'),
            'stress': _stress_summary(inp, res['capacities_by_year'][-1], info),
            'effective': _effective(inp, start),
            'plots': [{'key': 'dashboard', 'title': 'Portfolio dashboard: 10-year timeline, capacities, typical week, stress period, annual cost and CO2',
                       'url': f"/api/job/{job['id']}/dashboard.png"},
                      {'key': 'years', 'title': 'Year-by-year operation: daily supply mix against demand and the grid limit',
                       'url': f"/api/job/{job['id']}/years.png"}],
            'files': {k: f"/api/job/{job['id']}/{k}" for k in ('portfolio.json', 'dispatch.csv', 'summary.json', 'results.json') + SOURCE_FILES
                      if (out / k).exists()},
            'outputDir': str(out),
        })
        pr['step'] = 'done'
        job['status'] = 'done'
    except Exception as exc:
        traceback.print_exc()
        job['status'], job['error'] = 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        TRACK['job'] = None
        job['phase'] = None
        job['seconds'] = round(time.time() - t0)
        try:                                                   # keep the run so the page can reopen it after a server restart
            (out / 'job.json').write_text(json.dumps(_clean(job_view(job['id']))))
        except Exception:
            traceback.print_exc()
        CURRENT['id'] = None
        RUN_LOCK.release()


def _infeasible(job, status, skipped=False):
    """No feasible portfolio: find the reasons and nearby inputs that have one (team_app/gridtech_diagnose.py)."""
    pr = job['progress']
    pr['step'] = 'diagnose'
    job['phase'] = 'Finding the reasons and viable alternatives'
    diag = D.diagnose(job['inputs'], _parse, sizing_status=status)
    job['result'] = {'status': 'INFEASIBLE', 'sizingStatus': status, 'skippedSizing': skipped, 'diagnosis': _clean(diag),
                     'message': 'No portfolio can meet these inputs.' if diag['reasons'][0]['key'] != 'solver' else 'The solver found no portfolio in time.'}
    pr['step'] = 'done'
    job['status'] = 'done'


def _alternative_meta(a):
    """A run started from a suggested alternative: which run it is an alternative to and what was changed (shown on the page)."""
    if not isinstance(a, dict):
        return None
    changes = [str(c)[:300] for c in (a.get('changes') or [])][:30]
    if not changes:
        return None                                       # nothing differs from the original inputs: it is the original problem
    return {'ofJobId': str(a.get('ofJobId') or '')[:20], 'ofTag': str(a.get('ofTag') or '')[:80], 'title': str(a.get('title') or '')[:200],
            'changes': changes, 'original': a.get('original') if isinstance(a.get('original'), dict) else None}


def start_job(payload):
    alternative = _alternative_meta(payload.get('alternative'))
    thesis = payload.get('thesis') if isinstance(payload.get('thesis'), dict) else None      # set when a thesis agent started the run
    agents = payload.get('agents') if isinstance(payload.get('agents'), dict) else None      # the Optimizer form: start the four agents around this run
    payload = {k: v for k, v in payload.items() if k not in ('alternative', 'thesis', 'agents')}
    try:
        sc, info = _parse(payload)
        with redirect_stdout(io.StringIO()):
            inp = G.Inputs(sc)                              # catches the model's own input errors (e.g. schedule) before queueing
            if info['stress'] is not None:
                _check_sample_size(inp, sc)
    except BadInput as e:
        return 400, {'error': str(e), 'errors': e.errors}
    except ValueError as e:
        return 400, {'error': str(e), 'errors': {}}
    if not RUN_LOCK.acquire(blocking=False):
        return 409, {'error': 'A run is already in progress. Wait for it to finish.', 'jobId': CURRENT['id']}
    jid = uuid.uuid4().hex[:10]
    tag = ('alt_' if alternative else '') + f"m{info['start']}_{info['rel']}_{info['peak']:g}MW" + ('_staged' if info['staged'] else '') + ('_stress' if info['stress'] is not None else '') + (f"_slip{info['gridDelay']}" if info['gridDelay'] else '') + ('_noT2' if info['noTier2'] else '') + (('_no-' + '-'.join(info['excluded'])) if info['excluded'] else '') + (f"_grid+{info['gridAdderExtra']:g}" if info['gridAdderExtra'] else '') + ('_co2' if info['co2'] is not None else '')
    d = OUT / f'{time.strftime("%Y%m%d-%H%M%S")}_{tag}_{jid}'
    d.mkdir(parents=True, exist_ok=True)
    job = {'id': jid, 'status': 'running', 'phase': 'Starting', 'log': [], 'started': time.time(), 'dir': str(d),
           'tag': tag, 'result': None, 'error': None, 'staged': info['staged'], 'inputs': payload, 'alternative': alternative, 'thesis': thesis,
           'progress': {'step': 'prepare', 'timeLimit': 600,
                        'mip': {'variables': None, 'constraints': None, 'best': None, 'bound': None, 'gap': None, 'nodes': 0,
                                'solutions': 0, 'seconds': 0, 'history': [], 'status': None,
                                'stage': None, 'steps': [], 'offset': 0.0, 'nodesBase': 0,
                                'startedAt': None, 'updatedAt': None},
                        'dispatch': {'done': 0, 'total': 0}}}
    JOBS[jid] = job
    CURRENT['id'] = jid
    threading.Thread(target=_work, args=(job, sc, info), daemon=True).start()
    if agents is not None:
        TA.start_for_run(jid, payload, agents)
    return 200, {'jobId': jid}


def job_view(jid):
    j = JOBS.get(jid)
    if not j:
        if re.fullmatch(r'[0-9a-f]{10}', jid or ''):            # a finished run from before a server restart
            for f in OUT.glob(f'*_{jid}/job.json'):
                v = json.loads(f.read_text())
                v['thesisId'] = TA.thesis_for_run(jid) or (v.get('thesis') or {}).get('id')
                return v
        return None
    return _clean({'id': j['id'], 'status': j['status'], 'phase': j['phase'], 'log': j['log'][-40:], 'tag': j['tag'],
                   'startedAt': j['started'], 'elapsed': round(time.time() - j['started']) if j['status'] == 'running' else j.get('seconds'),
                   'error': j['error'], 'inputs': j['inputs'], 'alternative': j.get('alternative'), 'thesis': j.get('thesis'), 'progress': j['progress'], 'result': j['result'],
                   'thesisId': TA.thesis_for_run(j['id']) or (j.get('thesis') or {}).get('id')})


def _by_source_from_run(r, inp, d):
    """bySource for a run saved before it was stored: recomputed from its dispatch and capacities (same functions as a live run)."""
    cby = {int(x['year']): x['caps'] for x in r.get('capacitiesByYear') or []}
    caps = r['capacities']
    caps = {c_['id']: c_['mw'] for c_ in caps} if isinstance(caps, list) else caps
    ny = inp.n // 8760
    first = min(cby) if cby else None
    caps_y = [cby.get(y + 1, cby.get(first, caps) if first else caps) for y in range(ny)] if r.get('staged') else caps
    with redirect_stdout(io.StringIO()):
        cdf = G.cost_breakdown(inp, caps_y, d)
        edf, eann = G.emissions_by_source(inp, d)
    return _clean(by_source(cdf, edf, eann))


def build_charts(jdir):
    """Chart data (JSON) for a finished run: the same series the static dashboard draws. Cached as charts.json in the run folder."""
    jdir = Path(jdir)
    cache = jdir / 'charts.json'
    if cache.is_file():
        body = cache.read_bytes()
        try:
            if json.loads(body).get('version', 1) >= CHARTS_VERSION:
                return body
        except ValueError:
            pass                                                # unreadable cache: rebuild
    job = json.loads((jdir / 'job.json').read_text())
    r = job['result']
    with redirect_stdout(io.StringIO()):
        sc, _ = _parse({k: v for k, v in job['inputs'].items() if k != 'alternative'})
        inp = G.Inputs(sc)
    d = pd.read_csv(jdir / 'dispatch.csv')
    s = G._supply(d)
    used = [k for k in G.STACK if s[k].sum() > 1e-6]
    load, ordinary, eff, active = inp.load, inp.cap_ord, inp.cap_eff, inp.active
    cal = inp.cu.calendar_start
    md = lambda m_: G.month_to_date(m_, cal)
    rd = lambda a: [round(float(v), 2) for v in a]
    HPM = G.HPM
    month = np.arange(len(load)) // HPM
    mavg = lambda x: np.bincount(month, weights=x, minlength=120)[:120] / HPM
    mmax = lambda x: np.array([x[month == m].max() for m in range(120)])
    start = int(r['opening'])
    caps = {c_['id']: c_['mw'] for c_ in r['capacities']} if isinstance(r['capacities'], list) else r['capacities']
    events = {start: f'Opening {md(start)}', start + 10: f'Full demand {md(start + 10)}'}
    for rid in G.RES:
        if caps.get(rid, 0) > 0:
            m0 = max(start, int(inp.opt.loc[rid, 'available_month']))
            events[m0] = (events[m0] + ' / ' if m0 in events else '') + f'{G.LABELS[rid]} {md(m0)}'
    ys = 7 * 8760
    h0 = ys + 2400
    act = np.nonzero(active[ys:ys + 8760])[0]
    starts = act[np.r_[True, np.diff(act) > 1]] if act.size else np.array([5544])
    first = max(starts, key=lambda s0: (((act >= s0 - 24) & (act < s0 - 24 + 24 * 16)).sum(), -s0))
    a = max(ys, ys + int(first) - 24); b = min(ys + 8760, a + 24 * 16)
    nd = len(load) // 24
    dmean = lambda x: x[:nd * 24].reshape(nd, 24).mean(1)
    ny = len(load) // 8760
    ymean = lambda x: x[:ny * 8760].reshape(ny, 8760).mean(1)
    ann = []
    for x in r['annual']:
        has = (x.get('requested_mwh') or 0) > 0
        co2 = x.get('combined_electricity_co2_kg_per_mwh')
        cs = x.get('clean_energy_share')
        ann.append({'year': x['model_year'], 'requestedGwh': round((x.get('requested_mwh') or 0) / 1000, 2),
                    'costPerMwh': round(x['total_decision_cost_usd'] / x['requested_mwh'], 2) if has else None,
                    'kgPerMwh': round(co2, 2) if has and co2 is not None else None,
                    'cleanShare': round(cs, 2) if has and cs is not None else None})
    ev_d = active[:nd * 24].reshape(nd, 24).any(1).astype(int)
    out = {
        'version': CHARTS_VERSION,
        'sources': r.get('bySource') or _by_source_from_run(r, inp, d),
        'resources': [{'id': k, 'label': G.LABELS[k], 'color': G.COLORS[k]} for k in used],
        'calendarStart': cal, 'opening': start, 'fullDemand': start + 10, 'peakMw': round(float(inp.peak), 2),
        'milestones': [{'month': int(m_), 'label': t} for m_, t in sorted(events.items())],
        'monthly': {'series': {k: rd(mavg(s[k])) for k in used}, 'demand': rd(mavg(load)), 'demandPeak': rd(mmax(load)), 'gridCap': rd(mmax(ordinary))},
        'yearly': {'years': list(range(1, ny + 1)), 'series': {k: rd(ymean(s[k])) for k in used}, 'demand': rd(ymean(load)), 'gridCap': rd(ymean(ordinary))},
        'week': {'startHour': h0, 'modelYear': 8, 'series': {k: rd(s[k][h0:h0 + 168]) for k in used},
                 'demand': rd(load[h0:h0 + 168]), 'gridCap': rd(eff[h0:h0 + 168])},
        'stress': {'startHour': a, 'hours': b - a, 'hasEvents': bool(act.size), 'series': {k: rd(s[k][a:b]) for k in used},
                   'demand': rd(load[a:b]), 'gridCap': rd(eff[a:b]), 'event': [int(v) for v in active[a:b]]},
        'daily': {'series': {k: rd(dmean(s[k])) for k in used}, 'demand': rd(dmean(load)),
                  'gridCap': rd(ordinary[:nd * 24].reshape(nd, 24).max(1)), 'event': [int(v) for v in ev_d]},
        'annual': ann,
        'co2Cap': r.get('co2Cap'),
        'land': {'usedAcres': round(float(r['kpis'].get('land_acres', 0)), 2), 'limitAcres': round(float((r.get('effective') or {}).get('landAcres', inp.land)), 2)},
    }
    body = json.dumps(_clean(out) if '_clean' in globals() else out).encode()
    cache.write_bytes(body)
    return body


def job_dir(jid):
    j = JOBS.get(jid)
    if j:
        return Path(j['dir']), (j.get('result') or {}).get('status')
    if re.fullmatch(r'[0-9a-f]{10}', jid or ''):
        for f in OUT.glob(f'*_{jid}/job.json'):
            return f.parent, (json.loads(f.read_text()).get('result') or {}).get('status')
    return None, None


def _stress_text(inputs):
    se = (inputs or {}).get('stressEvents') or {}
    if se.get('mode') != 'custom':
        return 'packet stress events'
    n = len(se.get('events') or [])
    return f"{se.get('source') or 'custom'} stress events ({n})" if n else 'no stress events'


def _case_keys(inputs):
    """(every input except the stress events and calendar labels, the stress events) as comparable keys; None if they no longer parse."""
    try:
        with redirect_stdout(io.StringIO()):
            sc, _ = _parse({k: v for k, v in inputs.items() if k != 'alternative'})
    except Exception:
        return None
    rest = dataclasses.replace(sc, custom=dataclasses.replace(sc.custom, stress_events=None, calendar_start=None))
    return repr(rest), json.dumps(sc.custom.stress_events, sort_keys=True)


def compare_runs(jid):
    """Finished runs with the same inputs as run `jid` except the stress events, newest first: how the portfolio changes with the events."""
    v = job_view(jid)
    if not v:
        return None
    k = _case_keys(v.get('inputs') or {})
    if not k:
        return []
    out = []
    for w in _all_views():
        r = w.get('result') or {}
        if w.get('id') == jid or w.get('status') != 'done' or r.get('status') != 'OK':
            continue
        k2 = _case_keys(w.get('inputs') or {})
        if k2 and k2[0] == k[0] and k2[1] != k[1]:
            out.append(_clean({'id': w.get('id'), 'tag': w.get('tag'), 'startedAt': w.get('startedAt'), 'stressText': _stress_text(w.get('inputs')),
                               'stressInputs': (w.get('inputs') or {}).get('stressEvents'), 'stress': r.get('stress'), 'kpis': r.get('kpis'),
                               'capacities': r.get('capacities'), 'sizingStatus': r.get('sizingStatus')}))
    out.sort(key=lambda x: x.get('startedAt') or 0, reverse=True)
    return out


def _all_views():
    """Every saved run (outputs/web/*/job.json) plus the one in progress."""
    rows, seen = [], set()
    for jid in [CURRENT['id']] if CURRENT['id'] else []:
        v = job_view(jid)
        if v:
            rows.append(v); seen.add(jid)
    for f in OUT.glob('*/job.json'):
        try:
            v = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if v.get('id') not in seen:
            rows.append(v); seen.add(v.get('id'))
    return rows


def list_runs():
    """Every saved run plus the one in progress, newest first: a summary row each for the Previous runs page."""
    out = []
    for v in _all_views():
        r, i = v.get('result') or {}, v.get('inputs') or {}
        k = r.get('kpis') or {}
        th = v.get('thesis') or {}
        tid = TA.thesis_for_run(v.get('id'))
        rtitle = TA.thesis_title(tid) if tid else None
        name = rtitle or (f"{th['title']} · {th['scenario']}" if th.get('title') and th.get('scenario') else th.get('title'))
        out.append(_clean({
            'name': name, 'id': v.get('id'), 'tag': v.get('tag'), 'status': v.get('status'), 'startedAt': v.get('startedAt'), 'elapsed': v.get('elapsed'),
            'resultStatus': r.get('status'), 'sizingStatus': r.get('sizingStatus'), 'error': v.get('error'),
            'alternative': bool(v.get('alternative')),
            'peakMw': i.get('peakMw'), 'start': i.get('start'), 'reliability': i.get('reliability'), 'staged': i.get('staged'), 'co2Cap': i.get('co2Cap'),
            'stress': _stress_text(i), 'customStress': ((i.get('stressEvents') or {}).get('mode') == 'custom'),
            'cost': k.get('total_decision_cost_usd'), 'costPerMwh': k.get('decision_cost_usd_per_requested_mwh'),
            'co2': k.get('combined_electricity_co2_kg_per_mwh'), 'clean': k.get('clean_energy_share'),
            'capacities': [{'id': c['id'], 'label': c['label'], 'mw': c['mw']} for c in r.get('capacities') or [] if c.get('mw')],
        }))
    out.sort(key=lambda x: x.get('startedAt') or 0, reverse=True)
    return out


def meta():
    res = []
    for r in G.RES:
        row = G.OPT.loc[r]
        res.append({'id': r, 'label': row.label, 'short': G.BAR_LABELS[r], 'location': row.location, 'color': G.COLORS[r],
                    'capex_usd_per_kw': float(row.capex_usd_per_kw), 'fom_usd_per_kw_year': float(row.fom_usd_per_kw_year),
                    'payment_usd_per_mwh': float(row.payment_usd_per_mwh), 'available_month': int(row.available_month),
                    'max_mw': float(row.max_mw), 'block_mw': float(row.block_mw)})
    return {'resources': res,
            'reliability': [{'key': k, 'text': t} for k, t in G.PRESET_TEXT.items()],
            'packetSchedule': [{'month': int(r.from_month), 'mw': float(r.max_import_mw)} for r in G.ck.grid_schedule.itertuples()],
            'limits': {'peak': [5, 2000], 'start': [14, 109], 'interconnection': [18, 24], 'growth': [-0.2, 0.3]},
            'defaults': {'peakMw': 75, 'start': 20, 'reliability': 'R1', 'wholesaleGrowth': 0.05, 'gasGrowth': 0.03, 'staged': True},
            'busy': CURRENT['id']}


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = 'GridtechModel/1.0'

    def log_message(self, fmt, *args):
        if '--quiet' not in sys.argv:
            sys.stderr.write('%s %s\n' % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype='application/json', extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b'')

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == '/api/health':
                return self._send(200, {'ok': True})
            if path == '/api/meta':
                return self._send(200, meta())
            if path == '/api/runs':
                return self._send(200, list_runs())
            if path == '/api/stress':
                return self._send(200, stress_meta())
            if path == '/api/theses':
                return self._send(200, TA.list_theses())
            if path.startswith('/api/thesis/'):
                v = TA.view(path.split('/')[3])
                return self._send(200 if v else 404, v or {'error': 'unknown thesis'})
            if path.startswith('/api/job/'):
                parts = path.split('/')
                if len(parts) == 4:
                    v = job_view(parts[3])
                    return self._send(200 if v else 404, v or {'error': 'unknown run'})
                if len(parts) == 5 and parts[4] == 'compare':
                    c = compare_runs(parts[3])
                    return self._send(200 if c is not None else 404, c if c is not None else {'error': 'unknown run'})
                if len(parts) == 5 and parts[4] == 'charts':
                    jd, st = job_dir(parts[3])
                    if jd is None or st != 'OK':
                        return self._send(404, {'error': 'no charts for this run'})
                    return self._send(200, build_charts(jd), 'application/json')
                j = JOBS.get(parts[3])
                name = parts[4] if len(parts) > 4 else ''
                if not j and re.fullmatch(r'[0-9a-f]{10}', parts[3]):
                    j = next(({'dir': str(f.parent), 'tag': json.loads(f.read_text()).get('tag', 'run')} for f in OUT.glob(f'*_{parts[3]}/job.json')), None)
                f = Path(j['dir']) / name if j and name in FILES else None
                if not f or not f.is_file():
                    return self._send(404, {'error': 'unknown file'})
                extra = {'Cache-Control': 'no-cache'}
                if not name.endswith('.png'):
                    extra['Content-Disposition'] = f'attachment; filename="{j["tag"]}_{name}"'
                return self._send(200, f.read_bytes(), FILES[name], extra)
            return self._static(path)
        except Exception as exc:                                     # never let a model error kill the server
            traceback.print_exc()
            self._send(500, {'error': str(exc)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get('Content-Length', 0) or 0)
            payload = json.loads(self.rfile.read(n) or b'{}')
            if path == '/api/run':
                code, body = start_job(payload)
                return self._send(code, body)
            if path == '/api/thesis':
                code, body = TA.start(payload)
                return self._send(code, body)
            if path.startswith('/api/thesis/') and path.endswith('/rerank'):
                code, body = TA.rerank(path.split('/')[3], payload.get('weights'), bool(payload.get('full')))
                return self._send(code, body)
            if path.startswith('/api/thesis/') and path.endswith('/stop'):
                code, body = TA.stop(path.split('/')[3])
                return self._send(code, body)
            if path.startswith('/api/job/') and path.endswith('/agents'):        # start the agents around an earlier run
                jid = path.split('/')[3]
                v = job_view(jid)
                if not v:
                    return self._send(404, {'error': 'unknown run'})
                code, body = TA.start_for_run(jid, v.get('inputs') or {}, payload)
                return self._send(code, body)
            if path == '/api/stress/parse':
                return self._send(200, parse_stress_upload(str(payload.get('name') or 'file')[:100], payload.get('data')))
            self._send(404, {'error': 'unknown endpoint'})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {'error': str(exc)})

    def _static(self, path):
        if not DIST.exists():
            return self._send(200, '<h3>Model server is running.</h3><p>The website build was not found. Run <code>npm run build</code> in <code>faraday_web</code>, '
                                   'or start the dev server there (it proxies /api to this server).</p>', 'text/html')
        rel = path.lstrip('/') or 'index.html'
        f = (DIST / rel).resolve()
        if DIST.resolve() not in f.parents and f != DIST.resolve():
            return self._send(403, {'error': 'forbidden'})
        if not f.is_file():
            f = DIST / 'index.html'                                  # single-page app fallback
        ctype = mimetypes.guess_type(str(f))[0] or 'application/octet-stream'
        self._send(200, f.read_bytes(), ctype, {'Cache-Control': 'no-cache'})


def main():
    TA.init(sys.modules[__name__])
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f'Powerllel model server on http://{a.host}:{a.port}  (website: {"built" if DIST.exists() else "not built yet"})', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
