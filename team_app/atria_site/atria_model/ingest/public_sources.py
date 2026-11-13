"""The free public sources named for the Texas pilot. One Source each; in mock mode each returns Example rows and reports
'Not pulled'. `fetch()` is where the live reader goes; it is not called in this build."""
from __future__ import annotations

from .base import MODE, Source, example, mock_for

TCEQ_TURBINES = Source('tceq_turbines', 'TCEQ issued gas-turbine permits (xlsx)', 'https://www.tceq.texas.gov/permitting/air/guidance/newsourcereview/combustion/nsr_fac_turbines.html',
                       ['environmental.air_permit_precedent'], 'Download')
TCEQ_NSR = Source('tceq_nsr', 'TCEQ pending NSR permits (HTML)', 'https://www2.tceq.texas.gov/airperm/index.cfm', ['economics.projects_stuck_waiting'], 'Crawled')
COMPTROLLER = Source('comptroller', 'Texas Comptroller qualifying data center list', 'https://comptroller.texas.gov/taxes/data-centers/',
                     ['policy.incentives', 'economics.projects_stuck_waiting'], 'Download')
ERCOT_LLI = Source('ercot_lli', 'ERCOT large-load interconnection status (PDF)', 'https://www.ercot.com/services/rq/large-load-integration',
                   ['technical.ercot_largeload_queue'], 'Crawled')
GREENBOOK = Source('greenbook', 'EPA Green Book nonattainment areas', 'https://www.epa.gov/green-book', ['environmental.air_attainment_status'], 'Download',
                   note='Cross-check for the Atria airQuality answer, which already reads the Green Book layer.')
FEMA_NRI = Source('fema_nri', 'FEMA National Risk Index', 'https://hazards.fema.gov/nri/data-resources',
                  ['security.natural_hazards', 'environmental.heat_stress_cooling'], 'Download')
AQUEDUCT = Source('aqueduct', 'WRI Aqueduct 4.0', 'https://www.wri.org/data/aqueduct-global-maps-40-data', ['environmental.water_stress'], 'Download')
IPAC = Source('ipac', 'USFWS IPaC location API', 'https://ipac.ecosphere.fws.gov/location/api', ['environmental.endangered_species_critical'], 'API')
PEERINGDB = Source('peeringdb', 'PeeringDB API', 'https://www.peeringdb.com/apidocs/', ['technical.fiber_internet_exchange'], 'API')

ALL = [TCEQ_TURBINES, TCEQ_NSR, COMPTROLLER, ERCOT_LLI, GREENBOOK, FEMA_NRI, AQUEDUCT, IPAC, PEERINGDB]


def fetch(src: Source, **kw):
    if MODE == 'mock':
        raise RuntimeError(f'{src.id}: mock mode, no network calls in this build')
    raise NotImplementedError(src.id)


def rows(site_id: str) -> tuple[list, dict]:
    """(indicator rows, gate flags with their status) for one cached site."""
    out, flags = [], {}
    m = lambda k: mock_for(site_id, k)   # noqa: E731
    v = m('aqueduct')
    out.append(example(site_id, AQUEDUCT, 'environmental.water_stress', 'water_stress_0_5', v.get('water_stress_0_5'), '0-5'))
    v = m('fema_nri')
    out.append(example(site_id, FEMA_NRI, 'environmental.heat_stress_cooling', 'heat_days', v.get('heat_days'), 'days/yr'))
    out.append(example(site_id, FEMA_NRI, 'security.natural_hazards', 'hazard_index', v.get('hazard_index'), '0-100'))
    out.append(example(site_id, FEMA_NRI, 'security.natural_hazards', 'hazard_top', v.get('hazard_top')))
    v = m('peeringdb')
    out.append(example(site_id, PEERINGDB, 'technical.fiber_internet_exchange', 'fiber_ix_dist_mi', v.get('fiber_ix_dist_mi'), 'mi'))
    out.append(example(site_id, PEERINGDB, 'technical.fiber_internet_exchange', 'nearest_ix', v.get('nearest_ix')))
    v = m('ipac')
    out.append(example(site_id, IPAC, 'environmental.endangered_species_critical', 'critical_habitat_on_site',
                       None if v.get('critical_habitat_on_site') is None else ('Yes' if v['critical_habitat_on_site'] else 'No')))
    out.append(example(site_id, IPAC, 'environmental.endangered_species_critical', 'listed_species_count', v.get('listed_species_count'), 'species'))
    if 'critical_habitat_on_site' in v:
        flags['critical_habitat_on_site'] = dict(value=v['critical_habitat_on_site'], status='Example', source=f'Example (would read {IPAC.name})')
    v = m('ercot_lli')
    out.append(example(site_id, ERCOT_LLI, 'technical.ercot_largeload_queue', 'zone_queue_gw', v.get('zone_queue_gw'), 'GW',
                       note=f'{v.get("zone", "")}. Made-up value for the demo.'.strip('. ') if v else ''))
    v = m('tceq_turbines')
    out.append(example(site_id, TCEQ_TURBINES, 'environmental.air_permit_precedent', 'similar_permits_count', v.get('similar_permits_count'), 'permits'))
    out.append(example(site_id, TCEQ_TURBINES, 'environmental.air_permit_precedent', 'median_nox_ppm', v.get('median_nox_ppm'), 'ppm'))
    v = m('tceq_nsr')
    out.append(example(site_id, TCEQ_NSR, 'economics.projects_stuck_waiting', 'pending_nsr_in_county', v.get('pending_nsr_in_county'), 'applications'))
    v = m('comptroller')
    out.append(example(site_id, COMPTROLLER, 'policy.incentives', 'dc_incentive', v.get('dc_incentive')))
    out.append(example(site_id, COMPTROLLER, 'economics.projects_stuck_waiting', 'qualifying_dcs_in_county', v.get('qualifying_dcs_in_county'), 'data centers'))
    return out, flags
