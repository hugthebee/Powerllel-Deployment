"""One site checked against one thesis: gates, category scores, stall risk, time to power, the score and its breakdown,
headline answers per category, top risks and what would change the call. The server calls this for every page; the
site documents come from pipeline.run() (data/normalized/sites/<id>.json)."""
from __future__ import annotations

import json
from functools import lru_cache

from . import config as C, features as FT, score as SC
from .indicators import SCORED
from .labels import STAGE_WORDS, label
from .models import a_time_to_power as A, b_stall as B
from .models import f_underwriting as F
from .models._faraday import ANALYSIS, cached_analysis
from .models import g_reliability as G

SITES = C.ROOT / 'data' / 'normalized' / 'sites'
CATS = ['Technical', 'Environmental', 'Cultural', 'Policy', 'Security', 'Economics']
BRIDGE = {'gas_turbine', 'gas_engine', 'fuel_cell', 'battery_4h'}
ALL_RESOURCES = ['gas_turbine', 'gas_engine', 'fuel_cell', 'battery_4h', 'wind', 'utility_solar', 'nuclear']


def default_thesis() -> dict:
    return dict(org='investor', isos=['ERCOT'], targetMW=None, weights=dict(C.scoring()['default_weights']), r=C.default_r(), costPerMonthM=None,
                appetite='Medium', resources=list(ALL_RESOURCES), carbonMax=None, cooling='Air-cooled', redundancy='N+1', backup='R1', majorPermit='if_needed',
                maxPermitMonths=None, price=120.0, priceBasis='mwh', postGrid=5.0, minIrr=None, ownership='domestic', commercial='Unknown', answered=[])


def thesis_of(t: dict | None) -> dict:
    out = default_thesis()
    for k, v in (t or {}).items():
        if k in out and v is not None or k in ('ownership', 'targetMW', 'carbonMax', 'maxPermitMonths', 'minIrr', 'costPerMonthM'):
            out[k] = v
    w = {k: max(0.0, float(out['weights'].get(k, 0) or 0)) for k in C.scoring()['categories']}
    tot = sum(w.values()) or 1.0
    out['weights'] = {k: v / tot for k, v in w.items()}
    return out


@lru_cache(maxsize=None)
def _doc(site_id: str, mtime: float) -> dict:
    return json.loads((SITES / f'{site_id}.json').read_text())


def site_doc(site_id: str) -> dict | None:
    p = SITES / f'{site_id}.json'
    if not p.exists():
        return None
    return _doc(site_id, p.stat().st_mtime)


def site_ids() -> list[str]:
    idx = json.loads((C.ROOT / 'data' / 'normalized' / 'index.json').read_text())
    return [s['id'] for s in idx]


def tag_of(row: dict) -> str:
    """Customer-facing source tag: example | atria | public | model | customer."""
    if row['status'] == 'Example':
        return 'example'
    if row['status'] == 'No source':
        return 'none'
    h = row['how'].lower()
    if h.startswith('atria'):
        return 'atria'
    if h.startswith('derived'):
        return 'model'
    if h.startswith('customer'):
        return 'customer'
    return 'public'


# ------------------------------------------------------------------------------------------------ gates
def gate_flags(doc: dict, t: dict, values: dict) -> dict[str, dict]:
    g = {k: dict(v) for k, v in doc['flags'].items()}
    own = t.get('ownership')
    designated = C.scoring()['sb17_designated']['countries']
    mil = values.get('military_dist_mi')
    if own is None:
        g['ownership_blocked_sb17_cfius'] = dict(value=None, status='Not pulled', source='Your thesis', note='State buyer ownership to check SB 17 and CFIUS.')
    elif own == 'designated':
        g['ownership_blocked_sb17_cfius'] = dict(value=True, status='Customer provided', source='Your thesis + SB 17',
                                                 note=f'Buyer tied to a designated country ({", ".join(designated)}): SB 17 bars the purchase.')
    elif own == 'foreign':
        near = isinstance(mil, (int, float)) and mil <= C.scoring()['cfius_military_mi']
        g['ownership_blocked_sb17_cfius'] = dict(value=bool(near), status='Customer provided', source='Your thesis + CFIUS site list',
                                                 note=(f'Foreign buyer within {mil} mi of a military site: CFIUS review blocks a quick close.' if near else
                                                       'Foreign buyer, not near a listed military site: CFIUS filing likely voluntary. Legal review decides.'))
    else:
        stated = 'security.buyer_ownership_and' in (t.get('answered') or [])
        g['ownership_blocked_sb17_cfius'] = dict(value=False, status='Customer provided', source='Your thesis' if stated else 'Atria default thesis (U.S. buyer)',
                                                 note='U.S. buyer: SB 17 and CFIUS do not apply.' + ('' if stated else ' Assumed by the default thesis; change the buyer in your thesis.'))
    meta = C.scoring()['gates']
    return {k: dict(g.get(k, dict(value=None, status='No source', source='', note='Not checked.')), id=k, label=meta[k]['label'], category=meta[k]['category'],
                    state='fail' if g.get(k, {}).get('value') is True else 'pass' if g.get(k, {}).get('value') is False else 'unknown') for k in meta}


# ------------------------------------------------------------------------------------------------ evaluation
def _rows(doc):
    return doc['indicators']


def _values(doc):
    return {r['name']: r['value'] for r in _rows(doc) if r['status'] in SCORED and r['value'] is not None}


def _score_with(values, flags, t, stall, months):
    s = FT.category_scores(values)
    return SC.atria_score(flags, s, t['weights'], stall, months, t['r'])


def evaluate(site_id: str, thesis: dict | None = None, detail: bool = False) -> dict | None:
    doc = site_doc(site_id)
    if doc is None:
        return None
    t = thesis_of(thesis)
    rows = _rows(doc)
    values = _values(doc)
    by_name = {r['name']: r for r in rows}
    gates = gate_flags(doc, t, values)
    flags = {k: g['value'] for k, g in gates.items()}
    cd = FT.category_detail(values)
    feats = dict(doc['extraction']['features'])
    feats['moratorium_stage'] = FT.numeric('moratorium_stage', values.get('moratorium_stage'))
    stall = B.stall(feats, t.get('commercial'), p_state=(doc.get('stateLayer') or {}).get('p_state'))
    stall['stateNote'] = (doc.get('stateLayer') or {}).get('note')
    mA = doc['models']['A']
    bridge_ok = bool(BRIDGE & set(t['resources']))
    tp = A.expected_months(mA, mA.get('bridgeFullMonth'), bridge_ok)
    res = SC.atria_score(flags, {k: v['score'] for k, v in cd.items() if v['score'] is not None}, t['weights'], stall['p_stall'], tp['months'], t['r'])
    maxp = C.scoring()['appetite_max_stall'].get(t['appetite'], 0.2)

    # economics at the thesis price (cached Faraday inputs; no model run)
    an = cached_analysis(site_id) if doc['kind'] == 'cached' else None
    uw = F.underwrite(an, price=t['price'], price_basis=t['priceBasis'], p_stall=stall['p_stall'], post_grid=t['postGrid']) if an else None
    econ = dict(modeled=uw is not None, breakeven=uw['breakeven'] if uw else None, weightedNpvM=uw['weightedNpvM'] if uw else None, basis=t['priceBasis'], price=t['price'])

    cats = {k: _category(k, cd.get(k), rows, gates, values, by_name, doc, t, stall, maxp, econ) for k in CATS}
    thesis_fails = [c['reason'] for c in cats.values() if c['outcome'] == 'fail' and c.get('thesis')]
    examples = sum(1 for r in rows if r['status'] == 'Example')
    out = dict(id=doc['id'], kind=doc['kind'], label=doc['label'], county=doc['county'], state=doc['state'], iso=doc['iso'], lat=doc['lat'], lng=doc['lng'],
               source=doc['source'], readDate=doc['readDate'], counts=doc['counts'], examples=examples,
               hasRecon=(C.ROOT / 'data' / 'raw' / 'sites' / 'atria' / doc['id'] / 'recon_card.json').exists(),
               score=res['score'], G=res['G'], verdict=res['verdict'] if not thesis_fails or res['G'] == 0 else 'Fails your thesis',
               thesisFails=thesis_fails, result=res, gates=list(gates.values()), categories=cats,
               stall=dict(stall, max=maxp, appetite=t['appetite']), time=dict(tp, grid={k: mA.get(k) for k in ('p10', 'p50', 'p90', 'mean')}, r=t['r'], bridgeAllowed=bridge_ok),
               economics=econ, breakdown=_breakdown(res, cd, gates, stall, tp, t, rows))
    out['backup'] = _backup(site_id, an, t)
    out['risks'] = _risks(out, cd, values, by_name, t)
    out['changes'] = _changes(out, values, flags, t, stall, tp, by_name)
    if detail:
        out['indicators'] = [dict(r, label=label(r['name']), tag=tag_of(r), u=FT.score_indicator(r['name'], r['value']) if r['status'] in SCORED else None,
                                  weight=(FT.value_functions().get(r['name']) or {}).get('weight'), factor=C.data_map_row(r['data_map_id'])['factor'],
                                  frequency=C.data_map_row(r['data_map_id'])['frequency'], dataStatus=C.data_map_row(r['data_map_id'])['status_class'])
                             for r in rows]
        out['models'] = _models(doc, an, t, stall, uw)
        if out['backup']:
            out['models']['reliability'] = _reliability(site_id, an)
        out['extraction'] = doc['extraction']
        out['dataMap'] = [dict(id=r['id'], category=r['category'], factor=r['factor'], frequency=r['frequency']) for r in C.data_map()]
    return out


def _lc(s: str) -> str:
    """Lower-case the first letter only, so 'kV' and 'MW' survive inside a sentence."""
    return s[:1].lower() + s[1:]


def _fmt(v, d=0):
    return f'{v:,.{d}f}' if isinstance(v, (int, float)) else str(v)


def show(name: str, v, unit: str = '') -> str:
    """A value as a customer reads it: 45 of 100, 3% , adopted, 2.6 mi."""
    if v is None:
        return 'not answered'
    if name == 'moratorium_stage':
        if isinstance(v, (int, float)):
            v = {0: 'none', 1: 'public_request', 2: 'staff_study', 3: 'first_reading', 4: 'adopted'}.get(int(v), v)
        return STAGE_WORDS.get(v, str(v)).lower()
    if not isinstance(v, (int, float)):
        return str(v)
    if unit == 'share':
        return f'{_fmt(v * 100, 1 if v < 0.1 else 0)}%'
    if unit in ('0-100', '0-5'):
        return f'{_fmt(v, 1 if unit == "0-5" else 0)} of {unit[2:]}'
    d = 1 if abs(v) < 10 and v != int(v) else 0
    return f'{_fmt(v, d)} {unit}'.strip()


def _category(k, d, rows, gates, values, by, doc, t, stall, maxp, econ) -> dict:
    """Headline answer, status against the thesis, and the source tag for one tile."""
    mine = [r for r in rows if r['data_map_id'] and C.data_map_row(r['data_map_id'])['category'] == k]
    tags = {tag_of(r) for r in mine if r['status'] in SCORED}
    tag = 'example' if 'example' in tags else 'atria' if 'atria' in tags else 'model' if 'model' in tags else 'public' if tags else 'none'
    g = [x for x in gates.values() if x['category'] == k]
    s = (d or {}).get('score')
    v = lambda n: values.get(n)   # noqa: E731
    out = dict(category=k, s=s, coverage=(d or {}).get('coverage'), tag=tag, used=(d or {}).get('used', []), missing=(d or {}).get('missing', []),
               gates=[x['id'] for x in g], anchor='')
    fail_g = [x for x in g if x['state'] == 'fail']
    unk_g = [x for x in g if x['state'] == 'unknown']

    def res(outcome, reason, anchor='', thesis=False):
        out.update(outcome=outcome, reason=reason, anchor=anchor, thesis=thesis)
        return out

    if k == 'Technical':
        sub = by.get('nearest_substation', {}).get('value')
        out['headline'] = ' · '.join(x for x in [
            f'345 kV substation {_fmt(v("dist_345kv_mi"), 1)} mi' if v('dist_345kv_mi') is not None else 'No 345 kV substation found',
            ('gas line on the parcel' if v('gas_dist_mi') == 0 else f'gas {_fmt(v("gas_dist_mi"), 1)} mi') if v('gas_dist_mi') is not None else None,
            f'grid median month {_fmt(doc["models"]["A"]["p50"])}'] if x)
        out['sub'] = sub
    elif k == 'Environmental':
        path = by.get('air_path', {}).get('value') or {'psd': 'Major: PSD', 'nnsr': 'Major: nonattainment NSR', 'minor': 'Minor'}.get((doc['models'].get('C') or {}).get('path'), None)
        words = {'Major: PSD': 'major-source air permit (PSD)', 'Major: nonattainment NSR': 'major-source air permit (nonattainment NSR)', 'Minor': 'minor-source air permit'}
        out['headline'] = ' · '.join(x for x in [f'Water stress {_fmt(v("water_stress_0_5"), 1)} of 5' if v('water_stress_0_5') is not None else None,
                                                 f'{_fmt((v("wetland_share") or 0) * 100, 1)}% wetlands' if v('wetland_share') is not None else None,
                                                 words.get(path, path) if path else None] if x)
        if path and path.startswith('Major') and t['majorPermit'] == 'no':
            return res('fail', f'Needs a {words.get(path, path)}; your thesis does not accept one', 'air', True)
    elif k == 'Cultural':
        so = v('share_oppose')
        out['headline'] = ' · '.join(x for x in [f'{_fmt(so * 100)}% of statements opposed' if so is not None else 'No local statements found',
                                                 f'{_fmt(v("noise_dba_home"))} dBA at the nearest home' if v('noise_dba_home') is not None else None] if x)
    elif k == 'Policy':
        st = values.get('moratorium_stage')
        out['headline'] = ' · '.join(x for x in [by.get('zoning', {}).get('value'), f'moratorium: {STAGE_WORDS.get(st, st).lower()}' if st else None,
                                                 f'~{_fmt(v("permit_months"))} months to approve' if v('permit_months') is not None else None] if x)
        if t.get('maxPermitMonths') and v('permit_months') and v('permit_months') > t['maxPermitMonths']:
            return res('fail', f'About {_fmt(v("permit_months"))} months to local approval, above your {t["maxPermitMonths"]}', 'approval', True)
    elif k == 'Security':
        own = gates['ownership_blocked_sb17_cfius']
        out['headline'] = ' · '.join(x for x in ['Buyer ownership not stated' if own['state'] == 'unknown' else 'Ownership check: ' + ('blocked' if own['state'] == 'fail' else 'clear'),
                                                 f'hazard index {_fmt(v("hazard_index"))} of 100' if v('hazard_index') is not None else None,
                                                 f'military site {show("military_dist_mi", v("military_dist_mi"), "mi")}' if v('military_dist_mi') is not None else None] if x)
        if own['state'] == 'unknown':
            return res('unknown', 'State buyer ownership in your thesis to check SB 17 and CFIUS', 'gate-ownership_blocked_sb17_cfius')
    elif k == 'Economics':
        be = econ['breakeven']
        unit = '/MWh' if econ['basis'] == 'mwh' else '/kW-month'
        if not econ['modeled']:
            out['headline'] = 'Grid only: no on-site bridge power plan'
            out['tag'] = 'model'
            return res('caution', 'No gas line within reach, so there is no on-site bridge plan to underwrite: power waits for the grid', 'breakeven')
        if be is None:
            out['headline'] = f'No contract price up to $400{unit} breaks even'
            out['tag'] = 'model'
            return res('fail', f'The bridge contract loses money at any price up to $400{unit} at this stall risk', 'breakeven', 'economics.bridge_contract_price' in t.get('answered', []))
        out['headline'] = f'Breakeven ${_fmt(be)}{unit} vs your ${_fmt(econ["price"])}'
        out['tag'] = 'model'
        if be > econ['price']:
            return res('fail' if 'economics.bridge_contract_price' in t.get('answered', []) else 'caution', f'Breakeven ${_fmt(be)}{unit} is above the ${_fmt(econ["price"])} contract price', 'breakeven',
                       'economics.bridge_contract_price' in t.get('answered', []))
        if be > econ['price'] * 0.9:
            return res('caution', f'Breakeven ${_fmt(be)}{unit}, close to your ${_fmt(econ["price"])}', 'breakeven')
        return res('pass', f'Breakeven ${_fmt(be)}{unit}, under your ${_fmt(econ["price"])}', 'breakeven')
    if fail_g:
        return res('fail', f'{fail_g[0]["label"]}: {fail_g[0].get("note") or "fails"}', f'gate-{fail_g[0]["id"]}')
    if k == 'Cultural' and stall['p_stall'] > maxp:
        return res('fail', f'{_fmt(stall["p_stall"] * 100)}% chance the project stalls, above your {_fmt(maxp * 100)}%', 'stall', True)
    if s is None:
        return res('unknown', 'Nothing answered in this category yet', 'indicators')
    if unk_g:
        return res('caution', f'{unk_g[0]["label"]}: not checked. {unk_g[0].get("note") or ""}'.strip(), f'gate-{unk_g[0]["id"]}')
    if s >= 0.65:
        return res('pass', f'Category score {s:.2f} of 1', 'indicators')
    if s >= 0.4:
        worst = min(out['used'], key=lambda i: i['u'])
        return res('caution', f'Category score {s:.2f}; weakest: {_lc(label(worst["name"]))}', f'ind-{worst["name"]}')
    worst = min(out['used'], key=lambda i: i['u'])
    return res('fail', f'Category score {s:.2f}; weakest: {_lc(label(worst["name"]))}', f'ind-{worst["name"]}')


def _breakdown(res, cd, gates, stall, tp, t, rows) -> dict:
    """The formula left to right, with the value and sources of each step."""
    src = lambda cat: sorted({r['source'] for r in rows if r['status'] in SCORED and r['category'] == cat and r['name'] in FT.value_functions()})   # noqa: E731
    suit = res['suitability']
    after_stall = 100 * res['G'] * suit * (1 - stall['p_stall'])
    return dict(
        gates=dict(G=res['G'], items=[dict(id=g['id'], label=g['label'], state=g['state'], source=g['source']) for g in gates.values()]),
        suitability=dict(value=round(100 * suit, 1), parts=[dict(category=k, weight=round(t['weights'][k], 3), s=round(cd[k]['score'], 3), points=round(100 * v, 1), sources=src(k))
                                                            for k, v in res['contributions'].items()],
                         missing=res['categories_missing']),
        stall=dict(factor=res['risk_factor'], p=round(stall['p_stall'], 3), layers=dict(local=round(stall['p_local'], 3), state=stall['p_state'], commercial=stall['p_commercial']),
                   value=round(after_stall, 1)),
        time=dict(factor=res['time_factor'], months=tp['months'], basis=tp['basis'], note=tp['note'], r=t['r'], value=res['score']),
        score=res['score'])


def _risks(out, cd, values, by, t) -> list[dict]:
    items = []
    sid = out['id']
    for g in out['gates']:
        if g['state'] == 'fail':
            items.append(dict(prio=0, text=f'{g["label"]}: {g.get("note") or "fails the gate"}', category=g['category'], href=f'/site/{sid}/{g["category"].lower()}#gate-{g["id"]}'))
    for c in out['categories'].values():
        if c['outcome'] == 'fail' and c.get('thesis'):
            items.append(dict(prio=1, text=c['reason'], category=c['category'], href=f'/site/{sid}/{c["category"].lower()}#{c["anchor"]}'))
    for g in out['gates']:
        if g['state'] == 'unknown':
            items.append(dict(prio=2, text=f'Not checked: {g["label"]}. {g.get("note") or ""}'.strip(), category=g['category'],
                              href=f'/site/{sid}/{g["category"].lower()}#gate-{g["id"]}'))
    b = out.get('backup')
    if b and not b['ok']:
        items.append(dict(prio=1.5, text=f'Backup power: on-site gas falls {_fmt(b["shortMw"])} MW short of the {b["short"]} rule ({b["level"]}) in month {b["month"]}' + (', before the grid arrives' if b['preGrid'] else ''),
                          category='Technical', href=f'/site/{sid}/technical#backup'))
    fw = values.get('floodway_pct')
    if fw:
        items.append(dict(prio=3, text=f'{_fmt(fw, 1)}% of the parcel is floodway; build around it', category='Environmental', href=f'/site/{sid}/environmental#ind-floodway_pct'))
    if out['stall']['max'] * 0.8 < out['stall']['p_stall'] <= out['stall']['max']:
        items.append(dict(prio=3, text=f'{_fmt(out["stall"]["p_stall"] * 100)}% chance the project stalls (your limit {_fmt(out["stall"]["max"] * 100)}%)', category='Cultural',
                          href=f'/site/{sid}/cultural#stall'))
    # indicators losing the most points
    w = t['weights']
    wsum = sum(w[k] for k in cd if cd[k]['score'] is not None) or 1
    loss = []
    for k, d in cd.items():
        tw = sum(i['weight'] for i in d['used']) or 1
        for i in d['used']:
            pts = 100 * (w[k] / wsum) * (i['weight'] / tw) * (1 - i['u']) * out['result']['risk_factor'] * out['result']['time_factor']
            if pts >= 1:
                r = by.get(i['name'], {})
                loss.append(dict(prio=4 + 1 / (1 + pts), text=f'{label(i["name"])}: {show(i["name"], i["value"], r.get("unit", ""))} costs {_fmt(pts, 1)} points' +
                                 (' (example value)' if r.get('status') == 'Example' else ''),
                                 category=k, href=f'/site/{sid}/{k.lower()}#ind-{i["name"]}'))
    items += sorted(loss, key=lambda x: x['prio'])
    return [{k: v for k, v in x.items() if k != 'prio'} for x in sorted(items, key=lambda x: x['prio'])][:3]


def _verdict_of(score, G, unknown=()):
    return SC.verdict(score, G, list(unknown))


def _changes(out, values, flags, t, stall, tp, by) -> list[dict]:
    sid, base = out['id'], out['score']
    unknown = out['result']['unknown_gates']
    items = []
    for g in out['gates']:
        if g['state'] == 'fail':
            s2 = _score_with(values, dict(flags, **{g['id']: False}), t, stall['p_stall'], tp['months'])['score']
            items.append(dict(delta=100, text=f'Clear the {g["label"].lower()} gate (survey, legal review or another parcel): score {_fmt(s2)} ({_verdict_of(s2, 1).lower()})',
                              category=g['category'], href=f'/site/{sid}/{g["category"].lower()}#gate-{g["id"]}'))
    for g in out['gates']:
        if out['G'] == 0:
            break
        if g['state'] == 'unknown':
            if g['id'] == 'ownership_blocked_sb17_cfius':
                items.append(dict(delta=100, text='State buyer ownership: SB 17 or CFIUS could block this buyer (score → 0)', href='/#thesis', category='Security'))
            else:
                items.append(dict(delta=base, text=f'If the {g["label"].lower()} check fails, the score drops to 0', category=g['category'],
                                  href=f'/site/{sid}/{g["category"].lower()}#gate-{g["id"]}'))
    for name, vf in FT.value_functions().items():
        if out['G'] == 0:
            break
        r = by.get(name)
        cur = values.get(name)
        for anchor, word in ((vf['worst'], 'worst'), (vf['best'], 'best')):
            v2 = dict(values, **{name: anchor})
            s2 = _score_with(v2, flags, t, stall['p_stall'], tp['months'])['score']
            d = s2 - base
            if abs(d) < 1 or (cur is not None and FT.numeric(name, cur) == anchor):
                continue
            if cur is None or word != 'best' or d < 1:   # only what-ifs on values actually measured, toward the best mark
                continue
            items.append(dict(delta=d, text=f'{label(name)} at {show(name, anchor, vf["unit"])} instead of {show(name, cur, vf["unit"])}: score {_fmt(s2)}, {_verdict_of(s2, out["G"]).lower()}',
                              category=vf['category'], href=f'/site/{sid}/{vf["category"].lower()}#ind-{name}'))
    g = out['time']['grid']
    if tp['basis'] == 'grid' and g.get('p10') and out['G']:
        s2 = _score_with(values, flags, t, stall['p_stall'], g['p10'])['score']
        items.append(dict(delta=s2 - base, text=f'Grid arrives at the early estimate (month {_fmt(g["p10"])}): score {_fmt(s2)}', category='Technical', href=f'/site/{sid}/technical#arrival'))
    flips = lambda x: _verdict_of(base + x['delta'], out['G']) != _verdict_of(base, out['G']) if x['delta'] != 100 else True   # noqa: E731
    items.sort(key=lambda x: (not flips(x), -abs(x['delta'])))
    seen, res = set(), []
    for x in items:
        key = x['text'].split(' at ')[0]
        if key in seen:
            continue
        seen.add(key)
        res.append(dict(text=x['text'], href=x['href'], category=x['category'], delta=round(x['delta'], 1) if x['delta'] != 100 else None))
    res = res[:3]
    b = out.get('backup')
    if b and not b['ok'] and b['fix']['mw']:
        res.append(dict(text=f'Add {_fmt(b["fix"]["mw"])} MW of gas engines (about ${_fmt(b["fix"]["costPerYearM"], 1)}M a year) to meet the {b["short"]} backup rule; recheck the air permit',
                        href=f'/site/{sid}/technical#backup', category='Technical', delta=None))
    return res


_REL: dict = {}


def _reliability(site_id, an) -> dict | None:
    """GRIDTECH R0-R5 check of the cached portfolio (about 0.3 s; memoised per site and run)."""
    if not an:
        return None
    f = ANALYSIS / f'{site_id}.json'
    key = (site_id, f.stat().st_mtime if f.exists() else 0)
    if key not in _REL:
        _REL[key] = G.from_analysis(an)
    return _REL[key]


def _backup(site_id, an, t) -> dict | None:
    rel = _reliability(site_id, an)
    if not rel:
        return None
    lv = t.get('backup') if t.get('backup') in G.LEVELS else 'R1'
    bad = [rel['levels'][x] for x in G.LEVELS[:G.LEVELS.index(lv) + 1] if not rel['levels'][x]['ok']]
    worst = min(bad, key=lambda x: x['margin']) if bad else rel['levels'][lv]
    short = max(0.0, -worst['margin'])
    return dict(level=lv, label=G.LABELS[lv], short=G.SHORT[lv], preGrid=worst['preGrid'], ok=not bad, month=worst['month'], shortMw=round(short, 1), highestMet=rel['highestMet'],
                gasMw=rel['gasMw'], fix=G.extra_gas(short))


def _models(doc, an, t, stall, uw) -> dict:
    m = dict(doc['models'])
    if an:
        A_ = m['A']
        arrival = {k: A_[k] for k in ('p10', 'p50', 'p90', 'mean', 'sigma')}
        m['F'] = dict(underwriting=uw, distribution=F.distribution(an, arrival, stall['p_stall'], price=t['price'], price_basis=t['priceBasis'], n=500),
                      costOfDelay=F.cost_of_delay(an, price=t['price'], p_stall=stall['p_stall']),
                      developerCostPerMonth=t.get('costPerMonthM'))
        m['monthly'] = an.get('monthly')
        m['airFull'] = an.get('airPermit')
    return m


def all_sites(thesis: dict | None = None) -> list[dict]:
    out = [evaluate(i, thesis) for i in site_ids()]
    group = lambda r: (r['kind'] != 'cached') * 2 + (r['G'] == 0 or bool(r['thesisFails']))   # noqa: E731
    return sorted(out, key=lambda r: (group(r), -r['score'], r['label']))
