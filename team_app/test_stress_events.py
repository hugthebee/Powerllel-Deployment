"""Tests for customer stress events (gridtech_model.Custom.stress_events and the server's stressEvents input).

    python -m pytest team_app/test_stress_events.py              fast checks (a few seconds)
    GRIDTECH_SLOW=1 python -m pytest team_app/test_stress_events.py   + full optimiser runs (several minutes)
"""
import base64
import contextlib
import copy
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gridtech_model as G  # noqa: E402
import gridtech_server as S  # noqa: E402
import gridtech_diagnose as D  # noqa: E402

slow = pytest.mark.skipif(not os.environ.get('GRIDTECH_SLOW'), reason='set GRIDTECH_SLOW=1 to run the optimiser')

BASE = {'peakMw': '75', 'calendarStart': None, 'start': '20', 'interconnection': {'mode': 'packet'}, 'reliability': 'R1',
        'co2Cap': None, 'wholesaleGrowth': 0.05, 'gasGrowth': 0.03, 'resources': {}, 'minCaps': {}, 'staged': False}
PACKET_UTC = [dict(id=r.event_id, start_utc=r.start_utc, end_utc=r.end_utc, import_fraction=r.import_fraction, notice_minutes=30)
              for r in pd.read_csv(G.PACKET / 'data' / 'stress_events.csv').itertuples()]


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def payload(events, **kw):
    p = copy.deepcopy(BASE)
    p.update(kw)
    p['stressEvents'] = {'mode': 'custom', 'source': 'test', 'events': events}
    return p


def ev(start, hours, imp=60, notice=30, id_='e'):
    return {'id': id_, 'start': start, 'hours': str(hours), 'importPct': str(imp), 'noticeMin': str(notice)}


def errors_of(p):
    with pytest.raises(S.BadInput) as e:
        S._parse(p)
    return e.value.errors


PACKET_LOCAL = [ev(r['start'], r['hours'], r['importPct'], r['noticeMin'], r['id']) for r in S.stress_meta()['packet']]


# ---------------------------------------------------------------- model: hourly arrays
def test_packet_events_rebuild_packet_arrays_exactly():
    act, frac, notice = G.stress_hours(PACKET_UTC)
    H = G.ck.hourly
    assert np.array_equal(act, H.stress_active.astype(bool).to_numpy())
    assert np.allclose(frac, H.stress_import_fraction) and np.allclose(notice, H.notice_minutes)
    assert act.sum() == 32


@pytest.mark.parametrize('rel', ['R0', 'R1', 'R3', 'R4', 'R5'])
def test_inputs_identical_with_packet_events_as_custom(rel):
    a = quiet(G.Inputs, G.Scenario(start=20, rel=G.PRESETS[rel]))
    b = quiet(G.Inputs, G.Scenario(start=20, rel=G.PRESETS[rel], custom=G.Custom(stress_events=PACKET_UTC)))
    for k in ('active', 'cap_eff', 'cap_ord', 'reloc_lim', 'curt_lim', 'flex', 'load'):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert a.is_packet and not b.is_packet          # custom events are scored with score_custom, never the packet checker


def test_no_events_means_no_stress():
    i = quiet(G.Inputs, G.Scenario(start=20, custom=G.Custom(stress_events=[])))
    assert not i.active.any() and np.array_equal(i.cap_eff, i.cap_ord) and not i.flex.any()


def test_partial_hours_count_as_stress_hours():
    # 19:26-22:00 UTC touches the hours starting 19:00, 20:00 and 21:00
    act, frac, notice = G.stress_hours([dict(start_utc='2023-07-20T19:26:00Z', end_utc='2023-07-20T22:00:00Z', import_fraction=0.3, notice_minutes=0)])
    h = np.nonzero(act)[0]
    assert [str(G.REF_HOUR_UTC[i]) for i in h] == ['2023-07-20T19:00', '2023-07-20T20:00', '2023-07-20T21:00']
    assert np.allclose(frac[h], 0.3) and np.allclose(notice[h], 0) and np.allclose(frac[~act], 1)


def test_shared_hour_takes_stricter_fraction_and_shorter_notice():
    act, frac, notice = G.stress_hours([dict(start_utc='2023-07-20T19:00:00Z', end_utc='2023-07-20T19:30:00Z', import_fraction=0.8, notice_minutes=60),
                                        dict(start_utc='2023-07-20T19:30:00Z', end_utc='2023-07-20T21:00:00Z', import_fraction=0.4, notice_minutes=10)])
    h = np.nonzero(act)[0]
    assert len(h) == 2 and np.allclose(frac[h], 0.4) and np.allclose(notice[h], 10)


def test_import_fraction_and_notice_drive_limits():
    e = [dict(start_utc='2023-08-01T20:00:00Z', end_utc='2023-08-01T23:00:00Z', import_fraction=0.25, notice_minutes=0)]
    i = quiet(G.Inputs, G.Scenario(start=20, custom=G.Custom(stress_events=e)))
    h = np.nonzero(i.active)[0]
    first = int(np.nonzero(G.REF_HOUR_UTC == np.datetime64('2023-08-01T20:00'))[0][0])
    assert list(h) == [y * 8760 + first + j for y in range(10) for j in range(3)]      # the same 3 hours every model year
    open_ = h[h >= i.h0]
    assert len(open_) == 24 and (i.cap_ord[open_] > 0).all()
    assert np.allclose(i.cap_eff[h], 0.25 * i.cap_ord[h])
    assert not i.reloc_lim.any()                                   # no notice: Tier 1 cannot be relocated
    assert np.allclose(i.curt_lim[open_], i.t2[open_]) and (i.curt_lim[open_] > 0).all()   # Tier 2 can still be interrupted
    e[0]['notice_minutes'] = 30
    i2 = quiet(G.Inputs, G.Scenario(start=20, custom=G.Custom(stress_events=e)))
    assert (i2.reloc_lim[open_] > 0).all()


def test_sb6_cuts_custom_event_hours_to_zero():
    e = [dict(start_utc='2023-03-01T20:00:00Z', end_utc='2023-03-02T02:00:00Z', import_fraction=0.9, notice_minutes=30)]
    i = quiet(G.Inputs, G.Scenario(start=20, rel=G.PRESETS['R4'], custom=G.Custom(stress_events=e)))
    assert i.active.sum() > 0 and not i.cap_eff[i.active].any()


def test_sampling_keeps_custom_stress_hours_exactly():
    e = [dict(start_utc='2023-02-14T18:00:00Z', end_utc='2023-02-15T06:00:00Z', import_fraction=0.5, notice_minutes=30)]
    i = quiet(G.Inputs, G.Scenario(start=20, custom=G.Custom(stress_events=e)))
    H, w, _ = i.sample(4)
    act = np.nonzero(i.active[i.h0:])[0] + i.h0
    assert set(act) <= set(H) and np.allclose(w[np.isin(H, act)], 1.0)
    months = (H // 730)
    for m in range(20, 120):                                       # weights still add up to 730 hours a month
        assert abs(w[months == m].sum() - 730) < 1e-6, m


# ---------------------------------------------------------------- server: parsing and validation
def test_packet_mode_and_missing_field_use_packet_events():
    for p in (BASE, {**BASE, 'stressEvents': {'mode': 'packet'}}, {**BASE, 'stressEvents': None}):
        sc, info = S._parse(copy.deepcopy(p))
        assert sc.custom.stress_events is None and info['stress'] is None and info['stressLabel'] == 'packet'


def test_packet_events_typed_on_the_page_clock_convert_to_the_packet_utc_times():
    sc, info = S._parse(payload(PACKET_LOCAL))
    got = sorted((e['start_utc'], e['end_utc'], e['import_fraction'], e['notice_minutes']) for e in sc.custom.stress_events)
    want = sorted((pd.Timestamp(e['start_utc']).strftime('%Y-%m-%dT%H:%M:%SZ'), pd.Timestamp(e['end_utc']).strftime('%Y-%m-%dT%H:%M:%SZ'),
                   e['import_fraction'], 30.0) for e in PACKET_UTC)
    assert got == want
    a = quiet(G.Inputs, G.Scenario(start=20, rel=G.PRESETS['R1']))
    b = quiet(G.Inputs, sc)
    assert np.array_equal(a.active, b.active) and np.array_equal(a.cap_eff, b.cap_eff) and np.array_equal(a.flex, b.flex)


def test_page_clock_is_us_central_with_daylight_saving():
    assert S._local_to_utc('2023-08-24T15:00').strftime('%Y-%m-%dT%H:%M') == '2023-08-24T20:00'   # CDT, UTC-5
    assert S._local_to_utc('2023-01-15T15:00').strftime('%Y-%m-%dT%H:%M') == '2023-01-15T21:00'   # CST, UTC-6
    assert S._local_to_utc('2023-01-01T00:00').strftime('%Y-%m-%dT%H:%M') == '2023-01-01T06:00'   # reference hour 0
    assert S._local_to_utc('2023-08-24 15:00') is not None
    for bad in ('', None, '2023-13-01T00:00', '2023-02-30T10:00', '24/08/2023 15:00', '2023-08-24'):
        assert S._local_to_utc(bad) is None, bad


@pytest.mark.parametrize('event,key,text', [
    (ev('not a date', 4), 'stress.0.start', 'date and time'),
    (ev('2024-08-24T15:00', 4), 'stress.0.start', 'must be in 2023'),
    (ev('2022-12-31T23:00', 4), 'stress.0.start', 'must be in 2023'),
    (ev('2023-08-24T15:00', 0), 'stress.0.hours', 'enter'),
    (ev('2023-08-24T15:00', 169), 'stress.0.hours', 'enter'),
    (ev('2023-08-24T15:00', 'abc'), 'stress.0.hours', 'could not read'),
    (ev('2023-08-24T15:00', 4, imp=101), 'stress.0.importPct', 'enter'),
    (ev('2023-08-24T15:00', 4, imp=-1), 'stress.0.importPct', 'enter'),
    (ev('2023-08-24T15:00', 4, notice=''), 'stress.0.noticeMin', 'enter a value'),
    (ev('2023-12-31T20:00', 5), 'stress.0.hours', 'ends after 31 Dec'),
])
def test_bad_event_rows_are_rejected_on_their_field(event, key, text):
    e = errors_of(payload([event]))
    assert key in e and text in e[key], e


def test_overlapping_events_are_allowed_and_merge_per_hour():
    sc, _ = S._parse(payload([ev('2023-08-24T15:00', 4, imp=60, notice=60, id_='a'), ev('2023-08-24T17:00', 4, imp=20, notice=10, id_='b')]))
    act, frac, notice = G.stress_hours(sc.custom.stress_events)
    h = np.nonzero(act)[0]
    assert len(h) == 6                                             # 15:00-21:00 local, counted once
    assert list(frac[h]) == [0.6, 0.6, 0.2, 0.2, 0.2, 0.2] and list(notice[h]) == [60, 60, 10, 10, 10, 10]


def test_limits_on_number_of_events_and_total_hours():
    many = [ev(f'2023-{1 + i // 28:02d}-{1 + i % 28:02d}T10:00', 1, id_=str(i)) for i in range(61)]
    assert 'at most 60' in errors_of(payload(many))['stressEvents']
    weeks = [ev('2023-07-01T00:00', 168, id_='a'), ev('2023-08-01T00:00', 168, id_='b'), ev('2023-09-01T00:00', 1, id_='c')]
    assert '337 hours' in errors_of(payload(weeks))['stressEvents']
    S._parse(payload(weeks[:2]))                                  # exactly 336 hours is allowed


def test_stress_errors_do_not_hide_other_errors_and_vice_versa():
    p = payload([ev('2023-07-01T00:00', 168, id_='a'), ev('2023-08-01T00:00', 168, id_='b'), ev('2023-09-01T00:00', 1, id_='c')])
    p['peakMw'] = '1'
    e = errors_of(p)
    assert 'peakMw' in e and 'stressEvents' in e


def test_empty_custom_list_means_no_stress_events():
    sc, info = S._parse(payload([]))
    assert sc.custom.stress_events == [] and not quiet(G.Inputs, sc).active.any()


def test_unknown_mode_and_bad_shapes():
    assert 'stressEvents' in errors_of({**BASE, 'stressEvents': {'mode': 'weird'}})
    assert 'stressEvents' in errors_of({**BASE, 'stressEvents': {'mode': 'custom', 'events': 'x'}})
    assert 'stress.0.start' in errors_of(payload(['not an object']))


def test_events_spread_across_months_are_refused_before_queueing():
    spread = [ev(f'2023-{mo:02d}-{d:02d}T15:00', 1, id_=f'{mo}_{d}') for mo in range(1, 13) for d in (1, 28)]   # only 24 stress hours
    code, body = S.start_job(payload(spread))
    assert code == 400 and 'full detail' in body['errors']['stressEvents'] and S.CURRENT['id'] is None


@pytest.mark.parametrize('area_index', [0, 1, 2])
def test_every_catalog_area_fits_the_sizing_limit(area_index):
    a = S.stress_meta()['areas'][area_index]
    sc, _ = S._parse(payload([ev(e['start'], e['hours'], e['importPct'], e['noticeMin'], e['id']) for e in a['events']], start='14'))
    S._check_sample_size(quiet(G.Inputs, sc), sc)                  # every event ticked, earliest opening: still accepted


# ---------------------------------------------------------------- catalog and uploads
def test_catalog_meta():
    m = S.stress_meta()
    assert len(m['packet']) == 7 and m['packet'][0]['start'] == '2023-08-24T15:00' and m['packet'][0]['hours'] == 7
    areas = {a['area']: a for a in m['areas']}
    assert set(areas) == {'ERCOT (Texas Interconnection)', 'Eastern Interconnection', 'Western Interconnection'}
    assert [len(areas[a]['events']) for a in sorted(areas)] == [7, 5, 9]
    ercot = areas['ERCOT (Texas Interconnection)']['events']
    assert [(e['start'], e['hours'], e['importPct']) for e in ercot] == [(e['start'], e['hours'], e['importPct']) for e in m['packet']]
    off = {e['id'] for a in m['areas'] for e in a['events'] if not e['include']}
    types = {e['id']: (e['type'], e['confidence']) for a in m['areas'] for e in a['events']}
    assert all(types[i][1] == 'low' or 'Restricted Maintenance' in types[i][0] for i in off) and len(off) == 7
    st = {s['state']: s['area'] for s in m['states']}
    assert st['Texas'] == 'ERCOT (Texas Interconnection)' and st['California'] == 'Western Interconnection' and st['Virginia'] == 'Eastern Interconnection'


@pytest.mark.parametrize('area_index', [0, 1, 2])
@pytest.mark.parametrize('only_default', [True, False])
def test_every_catalog_area_is_a_valid_input(area_index, only_default):
    a = S.stress_meta()['areas'][area_index]
    rows = [ev(e['start'], e['hours'], e['importPct'], e['noticeMin'], e['id']) for e in a['events'] if e['include'] or not only_default]
    sc, _ = S._parse(payload(rows))
    i = quiet(G.Inputs, sc)
    hours = set()                                                  # independent count: every clock hour an event touches, once
    for e in sc.custom.stress_events:
        s, t = pd.Timestamp(e['start_utc']), pd.Timestamp(e['end_utc'])
        hours |= set(pd.date_range(s.floor('h'), t - pd.Timedelta(seconds=1), freq='h'))
    assert i.active[:8760].sum() == len(hours) == i.active[9 * 8760:].sum()      # same hours in every model year


def test_upload_page_layout_csv():
    csv = 'id,start,hours,import_pct,notice_min\nheat,2023-08-10 16:00,5,40,60\ncold,2023-02-02T07:00,3,0,\n'
    r = S.parse_stress_upload('mine.csv', base64.b64encode(csv.encode()).decode())
    assert r['problems'] == [] and [(x['id'], x['start'], x['hours'], x['importPct'], x['noticeMin']) for x in r['rows']] == \
        [('heat', '2023-08-10T16:00', '5', '40', '60'), ('cold', '2023-02-02T07:00', '3', '0', '30')]
    sc, _ = S._parse(payload([ev(x['start'], x['hours'], x['importPct'], x['noticeMin'], x['id']) for x in r['rows']]))
    assert len(sc.custom.stress_events) == 2


def test_upload_catalog_layout_and_xlsx():
    raw = (HERE / 'stress_data' / 'stress_events_all_interconnections_2023.csv').read_bytes()
    r = S.parse_stress_upload('cat.csv', base64.b64encode(raw).decode())
    assert len(r['rows']) == 21 and r['problems'] == []
    buf = io.BytesIO()
    try:
        pd.read_csv(io.BytesIO(raw)).to_excel(buf, index=False)
        pd.read_excel(io.BytesIO(buf.getvalue()))
        excel = True
    except ImportError:
        excel = False
    r2 = S.parse_stress_upload('cat.xlsx', base64.b64encode(buf.getvalue() or b'PK\x03\x04').decode())
    if excel:
        assert [x['start'] for x in r2['rows']] == [x['start'] for x in r['rows']]
    else:                                                          # this Python cannot read Excel: the page is told to use CSV
        assert r2['rows'] == [] and 'Save the sheet as CSV' in r2['problems'][0]


def test_upload_garbage():
    assert S.parse_stress_upload('x.csv', 'not base64!!')['problems']
    assert S.parse_stress_upload('x.csv', '')['problems']
    r = S.parse_stress_upload('x.csv', base64.b64encode(b'a,b\n1,2\n').decode())
    assert r['rows'] == [] and 'No events found' in r['problems'][0]


# ---------------------------------------------------------------- comparison keys
def test_case_keys_separate_stress_from_everything_else():
    k_packet = S._case_keys(BASE)
    k_custom = S._case_keys(payload(PACKET_LOCAL[:3]))
    k_cal = S._case_keys({**payload(PACKET_LOCAL[:3]), 'calendarStart': '2027-01'})
    k_other = S._case_keys({**payload(PACKET_LOCAL[:3]), 'reliability': 'R2'})
    assert k_packet[0] == k_custom[0] == k_cal[0] != k_other[0]
    assert k_packet[1] != k_custom[1] == k_cal[1]
    assert S._case_keys({'peakMw': 'x'}) is None


def test_stress_text():
    assert S._stress_text(BASE) == 'packet stress events'
    assert S._stress_text(payload([])) == 'no stress events'
    assert S._stress_text(payload(PACKET_LOCAL[:2])) == 'test stress events (2)'


# ---------------------------------------------------------------- capacity rules respond to the events
def test_harsher_events_raise_the_gas_requirement():
    def need(events):                                              # gas needed once the grid connection is complete (month 30+)
        sc, _ = S._parse(payload(events, reliability='R0'))
        i = quiet(G.Inputs, sc)
        return float(np.max((i.load - i.flex - i.cap_eff)[30 * 730:]))
    base = need(PACKET_LOCAL)
    harsh = need([{**e, 'importPct': '0'} for e in PACKET_LOCAL])
    none = need([])
    assert none <= 0 < base < harsh
    assert abs(base - 9.4878) < 1e-3                               # the packet's worst hour (model.md walkthrough)


def test_capacity_precheck_follows_the_events():
    # gas limited to one 25 MW turbine + 5 MW engine + 5 MW fuel cell (35 MW; 10 MW left under N-1)
    # full import from month 18, so only the stress events need gas
    small = {'gas_turbine': {'max_mw': '25'}, 'gas_engine': {'max_mw': '5'}, 'fuel_cell': {'max_mw': '5'}}
    kw = dict(reliability='R1', resources=small, interconnection={'mode': 'month', 'month': 18})
    sc_ok, _ = S._parse(payload(PACKET_LOCAL, **kw))
    assert quiet(D.structurally_feasible, sc_ok)                   # packet events need 9.5 MW: fits
    harsh = [{**e, 'importPct': '0', 'noticeMin': '0'} for e in PACKET_LOCAL]
    sc_bad, _ = S._parse(payload(harsh, **kw))
    assert not quiet(D.structurally_feasible, sc_bad)              # grid cut to 0 without notice: does not
    diag = quiet(D.diagnose, payload(harsh, **kw), S._parse)
    assert diag['reasons'] and 'stress event' in ' '.join(r['detail'] for r in diag['reasons'])
    for a in diag['alternatives']:                                 # every suggested alternative keeps the customer's events
        assert a['inputs']['stressEvents']['mode'] == 'custom'


def test_stress_summary_of_a_run():
    sc, info = S._parse(BASE)
    s = S._stress_summary(quiet(G.Inputs, sc), {'gas_turbine': 25, 'gas_engine': 10}, info)
    assert (s['label'], s['custom'], s['events'], s['hoursPerYear']) == ('packet', False, 7, 32)
    assert s['worst']['when'] == '26 Aug 15:00' and abs(s['worst']['gasNeedMw'] - 9.4878) < 1e-3 and abs(s['worst']['importMw'] - 45) < 1e-9
    assert (s['gasMw'], s['largestUnitMw']) == (35, 25.0)
    sc, info = S._parse(payload([]))
    s = S._stress_summary(quiet(G.Inputs, sc), {}, info)
    assert s['hoursPerYear'] == 0 and s['worst'] is None and s['events'] == 0 and s['largestUnitMw'] == 0


# ---------------------------------------------------------------- full optimiser runs
@slow
def test_full_run_packet_events_as_custom_give_the_packet_result():
    # A: the pure packet case, scored by the official checker; B: the same with the packet's events typed in as custom events
    sc_a = G.Scenario(start=20, rel=G.PRESETS['R1'])
    sc_b = G.Scenario(start=20, rel=G.PRESETS['R1'], custom=G.Custom(stress_events=S._parse(payload(PACKET_LOCAL))[0].custom.stress_events))
    ra, _ = quiet(G.run, sc_a)
    rb, _ = quiet(G.run, sc_b)
    assert ra['checker'] == 'PASS' and rb['checker'] == 'CUSTOM-OK'
    assert ra['capacities'] == rb['capacities']
    for k in ('total_decision_cost_usd', 'combined_electricity_co2_kg_per_mwh', 'clean_energy_share', 'interrupted_mwh', 'relocated_local_equivalent_mwh'):
        assert abs(ra[k] - rb[k]) <= 1e-6 * max(1, abs(ra[k])), k


@slow
def test_full_run_harsher_events_change_the_portfolio():
    sc_a, _ = S._parse(BASE)
    harsh = [{**e, 'importPct': '0', 'noticeMin': '0'} for e in PACKET_LOCAL]
    sc_b, _ = S._parse(payload(harsh))
    ra, _ = quiet(G.run, sc_a)
    rb, _ = quiet(G.run, sc_b)
    gas = lambda r: sum(r['capacities'][g] for g in G.GAS)
    assert rb['checker'] == 'CUSTOM-OK'
    assert gas(rb) > gas(ra) and rb['total_decision_cost_usd'] > ra['total_decision_cost_usd']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q'] + sys.argv[1:]))
