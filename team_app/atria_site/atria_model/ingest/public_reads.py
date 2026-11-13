"""Free public sources read for a site and saved in data/raw/sites/raw/<id>/ (each file is the service's own response):
WRI Aqueduct 4.0 baseline water stress, CDC/ATSDR Social Vulnerability Index 2022 (tract), FEMA USA Structures (homes for
the noise screen). The noise value is derived: a screening estimate, with its assumptions in the note."""
from __future__ import annotations

import json
import math

from .. import config as C
from .base import Source, row

RAW = C.ROOT / 'data' / 'raw' / 'sites' / 'raw'
AQUEDUCT_URL = 'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/aqueduct_water_risk/FeatureServer/1'
SVI_URL = 'https://services3.arcgis.com/ZvidGQkLaDJxRSJ2/arcgis/rest/services/CDC_ATSDR_Social_Vulnerability_Index_2022_USA/FeatureServer/2'
STRUCT_URL = 'https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0'
NOISE_LW_DBA = 110.0   # assumption: total A-weighted sound power of a ~55 MW enclosed turbine + engine plant
READS = Source('public_reads', 'WRI Aqueduct 4.0, CDC SVI 2022, FEMA USA Structures (public ArcGIS services)', AQUEDUCT_URL,
               ['environmental.water_stress', 'cultural.community_burden', 'cultural.noise_exposure'], 'Download', status='Answered', fetch_date='2026-10-29')


def _m(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
    return 2 * 6371000 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))


def _date(p) -> str:
    """The date the file was read (saved in it by scripts/fetch_public_reads.py; older files were read 2026-10-29)."""
    return (json.loads(p.read_text()).get('_fetched_at') or '2026-10-29')[:10]


def rows(site_id: str, centre: tuple[float, float]) -> list:
    d, out = RAW / site_id, []
    if (d / 'aqueduct.json').exists():
        f = json.loads((d / 'aqueduct.json').read_text())['features']
        if f:
            a = f[0]['attributes']
            out.append(row(site_id, 'environmental.water_stress', 'water_stress_0_5', round(a['bws_score'], 2), unit='0-5', source='WRI Aqueduct 4.0 baseline annual', how='Download',
                           date=_date(d / 'aqueduct.json'), note=f'{a["bws_label"]} (basin {a["pfaf_id"]}). Drought risk: {a["drr_label"]}; interannual variability: {a["iav_label"]}.', url=AQUEDUCT_URL))
    if (d / 'cdc_svi.json').exists():
        f = json.loads((d / 'cdc_svi.json').read_text())['features']
        if f:
            a = f[0]['attributes']
            out.append(row(site_id, 'cultural.community_burden', 'burden_pct', round(a['RPL_THEMES'] * 100), unit='pctile', source='CDC/ATSDR Social Vulnerability Index 2022',
                           how='Download', date=_date(d / 'cdc_svi.json'), url=SVI_URL,
                           note=f'{a["LOCATION"]}: national percentile. Stands in for EJScreen (taken down in 2025). Housing and transport theme {round(a["RPL_THEME4"] * 100)}th.'))
    if (d / 'usa_structures.json').exists():
        feats = json.loads((d / 'usa_structures.json').read_text())['features']
        homes = sorted((_m(centre, (f['centroid']['x'], f['centroid']['y'])), f['attributes'].get('PRIM_OCC')) for f in feats
                       if (f['attributes'].get('OCC_CLS') or '').startswith('Resid') and f.get('centroid'))
        if homes:
            r, occ = homes[0]
            lp = NOISE_LW_DBA - 20 * math.log10(r) - 8
            out.append(row(site_id, 'cultural.noise_exposure', 'noise_dba_home', round(lp), unit='dBA', source='Derived: FEMA USA Structures + hemispherical spreading', how='Derived (model)',
                           date=_date(d / 'usa_structures.json'), url=STRUCT_URL,
                           note=f'Nearest home ({occ}) {r:.0f} m from the parcel centre; {len(homes)} homes within 2 km. Assumes a {NOISE_LW_DBA:.0f} dBA plant at the centre, '
                                'no barriers or air absorption (ISO 9613-2 simplified). Screening only.'))
    return out
