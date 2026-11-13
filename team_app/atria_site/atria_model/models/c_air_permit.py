"""Model C: air-permit path for on-site gas.

`path()` is the skeleton's rules engine (actual vs potential emissions against the major-source line for the county's
attainment class). `assess()` is atria-power-desk's full screen (AP-42 factors on the Faraday hourly dispatch, PSD and
nonattainment thresholds, Class I distance, run-hours trade-off), used unchanged."""
from __future__ import annotations

from ._faraday import mod

METHOD = 'Emissions of the on-site plant (EPA AP-42 factors) against Clean Air Act major-source thresholds'


def path(nox_tpy_actual: float, nox_tpy_potential: float, major_threshold_tpy: float) -> str:
    if nox_tpy_potential < major_threshold_tpy:
        return 'minor'
    if nox_tpy_actual < major_threshold_tpy:
        return 'synthetic minor (cap run hours)'
    return 'major (PSD / nonattainment NSR)'


def assess(caps: dict, hourly: dict, air: dict | None, monthly_grid_cap=None) -> dict:
    return mod('air_permit').assess(caps, hourly, air, monthly_grid_cap)


def from_analysis(analysis: dict | None) -> dict | None:
    """The screen stored with a cached Faraday analysis, plus the skeleton's path label on its NOx numbers."""
    ap = (analysis or {}).get('airPermit')
    if not ap:
        return None
    nox_thr = (ap.get('thresholds') or {}).get('nox', {}).get('tpy')
    out = dict(ap)
    if nox_thr:
        out['skeletonPath'] = path(ap['emissionsTpy']['nox'], ap['pteTpy']['nox'], nox_thr)
        out['majorThresholdTpy'] = nox_thr
    return out
