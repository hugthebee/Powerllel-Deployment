"""Build every site: ingest rows (cached Atria + public layers, Example public sources, LLM extraction over example documents),
model outputs (A, C, D, F at default inputs), gap rows for unanswered data-map rows, gate flags. Writes
data/indicators.sqlite (the Indicator table) and data/normalized/sites/<id>.json, plus data/sources.json."""
from __future__ import annotations

import json
from datetime import date

from . import config as C
from . import indicators as I
from .ingest import atria_cached, local, policy_docs, public_reads, public_sources
from .ingest.base import MOCK, MOCK_DATE, Source, row
from .models import a_time_to_power as A, c_air_permit as CP, d_bridge_mc as D, f_underwriting as F
from .models._faraday import cached_analysis

OUT = C.ROOT / 'data' / 'normalized'
MODELS = Source('models', 'Atria models A-F (Faraday dispatch, air-permit rules, underwriting)', '', [
    'technical.grid_arrival', 'technical.bridge_power_need', 'technical.power_prices_by', 'environmental.air_permit_path',
    'environmental.carbon_intensity_of', 'environmental.cooling_water_demand', 'economics.bridge_contract_cash'], 'Derived (model)', kind='model', status='Answered')
CUSTOMER = Source('customer', 'Your thesis (customer provided)', '', ['security.foreignownership_limits'], 'Customer provided', kind='customer',
                  note='Answered on the home page; overrides defaults.')
DEFAULT_COOLING = 'Air-cooled'


def cooling_mgd(mw: float, cooling: str) -> float:
    return round(mw * 24 * C.scoring()['water_intensity_gal_per_mwh'].get(cooling, 25) / 1e6, 3)


def model_rows(site_id: str, an: dict | None, arrival: dict, air: dict | None, mw: float) -> list:
    M = lambda dm, name, v, unit='', source='', note='': row(site_id, dm, name, v, unit=unit, source=source, how='Derived (model)', date=MOCK_DATE,   # noqa: E731
                                                             status='Answered' if v is not None else 'Not pulled', note=note)
    out = [M('technical.grid_arrival', 'grid_p50_months', arrival['p50'], 'months', 'Atria grid arrival model',
             f'P10 {arrival["p10"]}, P90 {arrival["p90"]} months. {A.METHOD}.')]
    s = D.summary(an)
    out += [M('technical.bridge_power_need', 'bridge_mw_years', round(s['bridgeMwYears']) if s else None, 'MW-years', 'Atria power optimization model',
              s['detail'] if s else 'Faraday has not been run for this site.'),
            M('technical.bridge_power_need', 'onsite_mw', s['onsiteMw'] if s else None, 'MW', 'Atria power optimization model')]
    out.append(row(site_id, 'technical.power_prices_by', 'power_price_basis', 'ERCOT 2023 hourly prices' if s else None, source='Hackathon packet (ERCOT 2023)',
                   how='Download', date='2026-10-27' if s else None, status='Answered' if s else 'Not pulled', note='2023 only; one zone price stands in for the site.'))
    if air:
        out.append(M('environmental.air_attainment_status', 'air_major_tpy', air.get('majorThresholdTpy'), 'tpy', 'Atria airQuality -> Clean Air Act thresholds',
                     'Major-source NOx line for the county attainment class.'))
        out.append(M('environmental.air_permit_path', 'air_path', {'psd': 'Major: PSD', 'nnsr': 'Major: nonattainment NSR', 'minor': 'Minor', 'synthetic_minor': 'Synthetic minor'}
                     .get(air['path'], air['path']), '', 'Atria air permit model', f'NOx {air["emissionsTpy"]["nox"]:.0f} tpy actual vs {air.get("majorThresholdTpy")} tpy line.'))
    k = ((an or {}).get('model') or {}).get('recommendation', {}).get('kpis') or {}
    out.append(M('environmental.carbon_intensity_of', 'co2_kg_mwh', round(k['kgPerMwh']) if k.get('kgPerMwh') else None, 'kg/MWh', 'Atria power optimization model (EPA eGRID factors)',
                 'Campus supply, on-site plus grid, average (not marginal).'))
    out.append(M('environmental.cooling_water_demand', 'cooling_water_mgd', cooling_mgd(mw, DEFAULT_COOLING), 'Mgal/day', 'Derived: target MW x water intensity',
                 f'{DEFAULT_COOLING} default until your thesis sets the cooling type.'))
    uw = F.underwrite(an) if an else None
    out.append(M('economics.bridge_contract_cash', 'breakeven_usd_mwh', uw['breakeven'] if uw else None, '$/MWh', 'Atria bridge economics model',
                 'Contract price where the probability-weighted NPV is zero.'))
    return out


def gap_rows(site_id: str, have: set[str]) -> list:
    out = []
    for r in C.data_map():
        if r['id'] in have:
            continue
        cust = 'Customer provided' in r['how_tags'] and r['status_class'] == 'Need to add' and r['category'] == 'Security'
        status = 'Not pulled' if cust or r['status_class'] != 'Need to add' else 'No source'
        note = 'Answered once you state buyer ownership in the thesis.' if cust else (
            'Nothing answers this yet (data map: Need to add).' if status == 'No source' else 'A source exists but was not called for this site.')
        out.append(row(site_id, r['id'], r['id'].split('.', 1)[1], None, source=r['data_api'], how=r['how_obtained'], date=None, status=status, note=note))
    return out


def build_cached(site_id: str) -> tuple[dict, list]:
    rec = atria_cached.record(site_id)
    rows, flags = atria_cached.rows(site_id)
    r2, f2 = public_sources.rows(site_id)
    ex = policy_docs.extract_site(site_id)
    r3, f3 = policy_docs.rows(site_id, ex)
    rows += [r for r in public_reads.rows(site_id, (rec['lng'], rec['lat'])) + [x for x in r3 if x.status == 'Answered'] if r.name not in {x.name for x in rows}]
    answered = {r.name for r in rows}
    rows += [r for r in r2 + r3 + local.rows(site_id) if r.name not in answered]   # a value read from a source wins over a placeholder
    for k, f in {**f2, **f3}.items():
        flags.setdefault(k, f)
    rows, flags = _no_placeholders(rows, flags)
    an = cached_analysis(site_id)
    case = an['case'] if an else _case(site_id)   # no on-site power plan (e.g. no gas within reach): the grid plan still stands
    arrival = A.from_case(case)
    air = CP.from_analysis(an)
    mw = an['targetMW'] if an else (case['targetMw'] if case else 100.0)
    rows += model_rows(site_id, an, arrival, air, mw)
    d = D.summary(an)
    mc = D.monte_carlo(an, arrival) if an else None
    read = max((r.read_date for r in rows if r.read_date and r.status == 'Answered'), default='2026-10-28')
    doc = dict(id=site_id, kind='cached', label=rec['label'], county=rec['county'], state=rec['state'], lat=rec['lat'], lng=rec['lng'],
               iso=case['grid']['iso'] if case else None, targetMW=mw, source=f'Atria screen + public layers (latest read {read})',
               readDate=read, flags=flags, extraction=ex, stateLayer=ex.get('state'),
               models=dict(A=dict(arrival, bridgeFullMonth=(d or {}).get('fullDemand'), opening=(d or {}).get('opening')), C=air, D=dict(d or {}, monteCarlo=mc) if d else None,
                           F=dict(default=_uw_summary(F.underwrite(an))) if an else None,
                           grid=dict(schedule=case['grid']['schedule'], steps=case['grid'].get('steps'), substation=case['grid'].get('substation'),
                                     kind=case['grid'].get('kind'), queueMonths=case['grid'].get('queueMonths')) if case else None))
    return doc, rows


def _case(site_id: str) -> dict | None:
    from .models._faraday import mod
    SA = mod('site_adapter')
    site = SA.load_site(site_id)
    return SA.case_for(site, float(SA.default_target(site)[0])) if site else None


def _no_placeholders(rows: list, flags: dict) -> tuple[list, dict]:
    """No made-up values: a value whose source has not been read yet is shown as not available (and skipped in the score), never filled in."""
    out = []
    for r in rows:
        if r.status == 'Example':
            what = r.source.split('would read ', 1)[-1].rstrip(')') if 'would read' in r.source else r.source
            r = I.Indicator(site_id=r.site_id, category=r.category, name=r.name, value=None, unit=r.unit, source=what, how=r.how, read_date=None, status='Not pulled',
                            data_map_id=r.data_map_id, note='Not available yet: this source has not been read for this site.', url=r.url)
        out.append(r)
    flags = {k: (dict(value=None, status='Not pulled', source=f.get('source', '').replace('Example (would read ', '').rstrip(')'), note='Not checked yet.') if f.get('status') == 'Example' else f)
             for k, f in flags.items()}
    return out, flags


def _uw_summary(uw: dict) -> dict:
    return dict(weightedNpvM=uw['weightedNpvM'], breakeven=uw['breakeven'], inputs=uw['inputs'],
                cases={k: {kk: v[kk] for kk in ('npvM', 'irr', 'probability', 'tenorYears', 'capexM', 'yearlyM')} for k, v in uw['cases'].items()})


def build_example(site_id: str, m: dict) -> tuple[dict, list]:
    src = 'Example (made-up site)'
    rows = []
    for name, v in m['values'].items():
        vf = C.scoring()['value_functions'][name]
        rows.append(row(site_id, vf['data_map'], name, v, unit=vf['unit'], source=src, how=C.data_map_row(vf['data_map'])['how_obtained'], date=MOCK_DATE, status='Example',
                        note='Made-up site for the demo.'))
    for name, dm in (('owner', 'technical.parcel'), ('zoning', 'policy.zoning_status_and'), ('flood_zone', 'environmental.floodplain')):
        rows.append(row(site_id, dm, name, m[name], source=src, how='Example', date=MOCK_DATE, status='Example', note='Made-up site for the demo.'))
    arrival = A.arrival(m['grid_p50_months'])
    arrival.update(curve=A.curve(m['grid_p50_months']))
    rows.append(row(site_id, 'technical.grid_arrival', 'grid_p50_months', arrival['p50'], unit='months', source='Example (model A on example inputs)', how='Derived (model)',
                    date=MOCK_DATE, status='Example'))
    flags = {k: dict(value=v, status='Example', source=src) for k, v in m['flags'].items()}
    doc = dict(id=site_id, kind='example', label=m['label'], county=m['county'], state=m['state'], lat=m['lat'], lng=m['lng'], iso=m['iso'], targetMW=100.0,
               source='Example site (made up)', readDate=MOCK_DATE, flags=flags,
               extraction=dict(events=[], statements=[], dropped=[], documents=[], features=dict(share_oppose=None, net_stance=None, volume=0, volume_trend=None,
                                                                                                top_concern=None, concerns={}), moratorium_stage='none', months=[], posts_per_week=[]),
               models=dict(A=dict(arrival, bridgeFullMonth=None, opening=None), C=dict(path=m['air_path']), D=None, F=None, grid=None))
    return doc, rows


def credits_total() -> dict:
    import csv
    p = C.ROOT / 'data' / 'credits_log.csv'
    rows = list(csv.DictReader(p.open())) if p.exists() else []
    this = sum(int(r['credits']) for r in rows if r['build'] == 'atria-site-decisions')
    return dict(rows=rows, thisBuild=this, inherited=sum(int(r['credits']) for r in rows if r['build'] != 'atria-site-decisions'), budget=20000)


def run(db_path=I.DB, verbose=True) -> dict:
    con = I.connect(db_path)
    con.execute('DELETE FROM indicator')
    (OUT / 'sites').mkdir(parents=True, exist_ok=True)
    sites, all_rows = [], []
    for sid in atria_cached.site_ids():
        doc, rows = build_cached(sid)
        sites.append((doc, rows))
    for sid, m in MOCK.get('example_sites', {}).items():   # none in this build: every site is a real parcel
        if not sid.startswith('_'):
            sites.append(build_example(sid, m))
    sources = [atria_cached.ATRIA, atria_cached.PERMIT_PATH, atria_cached.PUBLIC, public_reads.READS, *public_sources.ALL, policy_docs.LLM, local.GAPS, MODELS, CUSTOMER]
    for doc, rows in sites:
        rows += gap_rows(doc['id'], {r.data_map_id for r in rows})
        I.upsert(con, rows)
        doc['counts'] = {s: sum(r.status == s for r in rows) for s in I.STATUSES}
        doc['indicators'] = [I.as_dict(r) for r in rows]
        (OUT / 'sites' / f'{doc["id"]}.json').write_text(json.dumps(doc, indent=1, default=float))
        all_rows += rows
        for s in sources:
            mine = [r for r in rows if r.data_map_id in s.data_map]
            s.sites[doc['id']] = 'Answered' if any(r.status == 'Answered' for r in mine) else 'Not pulled'
    for s in public_sources.ALL:
        s.status, s.fetch_date = 'Not pulled', None
    for s in (atria_cached.ATRIA, atria_cached.PUBLIC):
        s.fetch_date = '2026-10-28'
    MODELS.fetch_date = MOCK_DATE
    covered = {i for s in sources for i in s.data_map}
    reg = dict(mode='mock', built=date.today().isoformat(), sources=[s.as_dict() for s in sources],
               uncovered=[r['id'] for r in C.data_map() if r['id'] not in covered])
    (C.ROOT / 'data' / 'sources.json').write_text(json.dumps(reg, indent=1))
    (OUT / 'index.json').write_text(json.dumps([{k: d[k] for k in ('id', 'kind', 'label', 'county', 'state', 'iso', 'counts')} for d, _ in sites], indent=1))
    if verbose:
        tot = {s: sum(r.status == s for r in all_rows) for s in I.STATUSES}
        print(f'{len(sites)} sites, {len(all_rows)} indicator rows {tot}; {len(sources)} sources; credits this build: {credits_total()["thisBuild"]}')
    return dict(sites=len(sites), rows=len(all_rows))
