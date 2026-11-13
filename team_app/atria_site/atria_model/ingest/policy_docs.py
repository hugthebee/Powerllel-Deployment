"""Policy and sentiment: LLM extraction over local documents (tab "3 Policy LLM search"). In this build the documents and the
LLM answers are Examples (data/examples/policy_documents.json); they still go through extract.validate, so malformed output
and quotes not found in the text are dropped exactly as they would be live."""
from __future__ import annotations

import json

from .. import config as C
from .. import extract as X
from .base import MOCK_DATE, Source, row

DOCS = json.loads((C.ROOT / 'data' / 'examples' / 'policy_documents.json').read_text())
LLM = Source('policy_llm', 'Local agendas, minutes, news and public posts (LLM extraction)', '', ['cultural.opposition_history_and', 'policy.local_moratoriums_and'],
             'Crawled', kind='llm', note='Example documents and answers in this build.')


def real_docs(site_id: str) -> dict:
    p = C.ROOT / 'data' / 'raw' / 'sites' / 'raw' / site_id / 'policy_documents.json'
    return json.loads(p.read_text()) if p.exists() else {}


def extract_site(site_id: str) -> dict:
    events, statements, dropped, docs = [], [], [], []
    real = real_docs(site_id)
    pool = [dict(d, real=True) for d in real.get('documents', [])]   # real documents only; a site with none reads as not checked
    for d in pool:
        res = X.validate(d['llm_answer'], d['text'])     # the canned answer stands in for call_llm(EXTRACT_PROMPT, text)
        meta = dict(doc=d['id'], title=d['title'], kind=d['kind'], date=d['date'], jurisdiction=d['jurisdiction'], url=d.get('url', ''), publisher=d.get('publisher', ''), real=bool(d.get('real')))
        events += [dict(meta, **{k: v for k, v in e.items() if v is not None or k not in meta}) for e in res['events']]   # the event's own date wins over the document's
        statements += [dict(s, **meta) for s in res['statements']]
        dropped += [dict(reason=r, **meta) for r in res['dropped']]
        docs.append(dict(meta, kept=len(res['events']) + len(res['statements']), dropped=len(res['dropped'])))
    feats = X.sentiment_features(statements, real.get('posts_per_week') if real else DOCS['posts_per_week'].get(site_id))
    stage_rank = {s: i for i, s in enumerate(X.STAGES)}
    mor = sorted((e for e in events if e['type'] == 'moratorium'), key=lambda e: (e.get('date') or '', stage_rank[e['stage']]))
    latest = mor[-1]['stage'] if mor else 'none'
    stage = ('none' if latest in ('repealed', 'rejected') else latest) if docs else None   # the latest event decides; nothing read: unknown, not "none"
    return dict(events=events, statements=statements, dropped=dropped, documents=docs, features=feats, moratorium_stage=stage, real=bool(real), state=real.get('state'),
                months=[dict(jurisdiction=j, month=m, **f) for (j, m), f in X.jurisdiction_month_features(statements).items()],
                posts_per_week=(real.get('posts_per_week') if real else DOCS['posts_per_week'].get(site_id)) or [])


INCENTIVE = ('State sales tax exemption for certified qualifying data centers (Tax Code §151.359): 10 or 15 years, state tax only',
             'Single-occupant data centers that commit to the capital investment and jobs the Comptroller requires; the investment must be met within five years. Local sales tax still applies.',
             'https://comptroller.texas.gov/taxes/data-centers/')


def _statewide(site_id: str, ex: dict, rd: str) -> list:
    """Rules that apply to every Texas site: grid and permit pauses (from the site's saved state events) and the state data-center exemption."""
    out = []
    st = ex.get('state') or {}
    if st.get('events'):
        out.append(row(site_id, 'policy.state_and_grid', 'state_rules', ' · '.join(f'{e["date"]}: {e["title"]}' for e in st['events']), source='Governor, PUCT and ERCOT notices (news and law-firm alerts)',
                       how='Crawled', date=rd, url=st['events'][-1]['url'], note=f'Affects: {", ".join(sorted({e["applies"] for e in st["events"]}))}. {st.get("note", "")}'))
    if ex.get('real'):
        out.append(row(site_id, 'policy.incentives', 'dc_incentive', INCENTIVE[0], source='Texas Comptroller, data center program', how='Desktop research', date=rd, url=INCENTIVE[2], note=INCENTIVE[1]))
    return out


def rows(site_id: str, ex: dict) -> tuple[list, dict]:
    f = ex['features']
    real = ex.get('real')
    rd = (real_docs(site_id).get('read_date') or '2026-10-29') if real else MOCK_DATE
    src = ('Local news and council coverage, read by AI: ' + '; '.join(sorted({d['publisher'] for d in ex['documents'] if d.get('publisher')}))) if real else 'Example (LLM extraction over example documents)'
    R = lambda dm, name, v, unit='', note='': row(site_id, dm, name, v, unit=unit, source=src, how='Crawled + LLM extraction', date=rd,   # noqa: E731
                                                  status=('Answered' if real else 'Example') if v is not None else 'Not pulled', note=note)
    out = [R('cultural.opposition_history_and', 'share_oppose', None if f['share_oppose'] is None else round(f['share_oppose'], 3), 'share',
             f'{f["volume"]} statements kept from {len(ex["documents"])} documents.'),
           R('cultural.opposition_history_and', 'net_stance', None if f['net_stance'] is None else round(f['net_stance'], 3), '-1..+1'),
           R('cultural.opposition_history_and', 'volume_trend', None if f['volume_trend'] is None else round(f['volume_trend'], 2), 'x',
             'Change in weekly public posts, first to last week.'),
           R('policy.local_moratoriums_and', 'moratorium_stage', ex['moratorium_stage'], 'stage', 'Highest stage of any data-center moratorium event found.' if ex['documents'] else
             'No local agendas, minutes or news read for this site yet.')]
    out += _statewide(site_id, ex, rd)
    if not ex['documents']:
        return out, {'active_moratorium': dict(value=None, status='Not pulled', source='Local agendas and minutes (not read yet)', note='Data-center moratoriums not checked; wind and solar ordinances show none.')}
    active = ex['moratorium_stage'] in ('adopted', 'extended')
    if real:
        mor = [e for e in ex['events'] if e['type'] == 'moratorium']
        hist = '; '.join(f'{e["stage"].replace("_", " ")} {e["date"]}' for e in sorted(mor, key=lambda e: e['date']))
        juris = sorted({d['jurisdiction'] for d in ex['documents']})
        out.append(row(site_id, 'policy.local_moratoriums_and', 'moratorium_history', hist or 'None proposed', source=src, how='Crawled + LLM extraction', date=rd,
                       note=f'Data-center moratorium history for {", ".join(juris)}.'))
        for e in ex['events']:
            if e['type'] == 'ordinance':
                out.append(row(site_id, 'policy.local_approval_path', 'data_center_approval', f'Specific Use Authorization by City Council for every new data center (adopted {e["date"]}, {e["vote_for"]}-{e["vote_against"]})',
                               source=src, how='Crawled + LLM extraction', date=rd, url=e.get('url', ''),
                               note="New data centers limited to heavy commercial and industrial districts; whether this parcel's district qualifies needs the ordinance text. No published timeline."))
        ap = real_docs(site_id).get('approval')
        if ap:
            out.append(row(site_id, 'policy.local_approval_path', 'data_center_approval', ap['value'], source=ap['source'], how='Desktop research', date=rd, url=ap.get('url', ''), note=ap['note']))
    return out, {'active_moratorium': dict(value=active, status='Answered' if real else 'Example', source=src, note=f'Stage: {ex["moratorium_stage"].replace("_", " ")}.' +
                                           (' A city-wide moratorium was proposed and set aside in favour of zoning rules.' if any(e['stage'] == 'rejected' for e in ex['events']) else ''))}
