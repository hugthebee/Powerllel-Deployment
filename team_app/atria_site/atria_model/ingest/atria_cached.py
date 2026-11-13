"""Atria modules and public layers already read for the three cached Texas sites (atria-power-desk, 2026-10-28: atria_lookup +
a 22-module atria_screen per site, 1,620 credits, logged in data/credits_log.csv). Nothing is called here; the saved responses
in data/raw/sites/ are parsed into Answered rows."""
from __future__ import annotations

import json

from .. import config as C
from .base import Source, row

RAW = C.ROOT / 'data' / 'raw' / 'sites'
ATRIA = Source('atria_screen', 'Atria MCP screen (saved responses)', 'https://askatria.ai', [
    'technical.parcel', 'technical.adjoining_parcels_assemblage', 'technical.easements_and_lines', 'technical.substations', 'technical.gas_pipelines',
    'technical.soils', 'technical.grid_headroom_load', 'environmental.water_supply', 'environmental.wetlands', 'environmental.floodplain',
    'environmental.air_attainment_status', 'environmental.contamination', 'policy.zoning_status_and', 'policy.local_moratoriums_and',
    'security.critical_infrastructure_exposure', 'cultural.demographics'], 'Atria module', kind='atria', cost='Paid (credits)', status='Answered',
    note='Read 2026-10-28 by atria-power-desk; no new calls in this build. demographics was not in that screen.')
PERMIT_PATH = Source('atria_permit_path', 'Atria atria_permit_path', 'https://askatria.ai', ['policy.local_approval_path'], 'Atria module', kind='atria',
                     cost='100 credits per site', note='Not pulled: shortlisted sites only (credit funnel).')
PUBLIC = Source('public_layers', 'Public GIS layers (TxGIO parcels, HIFLD, EIA, EPA, FEMA)', 'https://geographic.texas.gov', [
    'technical.parcel', 'economics.land_value_vs'], 'Download', status='Answered', note='Read 2026-10-28 by atria-power-desk scripts/fetch_site.py.')


def site_ids() -> list[str]:
    return sorted(p.stem for p in RAW.glob('*.json'))


def record(site_id: str) -> dict:
    return json.loads((RAW / f'{site_id}.json').read_text())


def screen(site_id: str) -> dict:
    return json.loads((RAW / 'atria' / site_id / 'screen.json').read_text())


MODULE_NAMES = {'assessorRecord': 'county appraisal record', 'adjoiningParcels': 'adjoining parcels', 'easementIndicators': 'easements and lines', 'substations': 'substations (HIFLD)',
                'transmissionLines': 'transmission lines (HIFLD)', 'gridCapacity': 'grid capacity', 'gasInfrastructure': 'gas pipelines (EIA)', 'elevation': 'terrain (USGS 3DEP)',
                'waterSewer': 'water systems (EPA)', 'wetlands': 'wetlands (USFWS NWI)', 'floodZone': 'flood zones (FEMA)', 'airQuality': 'air quality (EPA Green Book)',
                'contamination': 'contamination (EPA)', 'zoningJurisdiction': 'zoning jurisdiction', 'energyOrdinance': 'energy ordinances (NREL)',
                'regulatedFacilities': 'regulated facilities (EPA)', 'naturalHazardRisk': 'natural hazard risk (FEMA NRI)', 'coolingClimate': 'climate for cooling',
                'telecomInfrastructure': 'fiber and exchanges (PeeringDB)', 'criticalHabitat': 'critical habitat (USFWS)', 'listedSpecies': 'listed species (USFWS)',
                'geologyHazards': 'geology and karst', 'waterSupply': 'water supply (USGS)'}


def lookup(site_id: str) -> dict:
    return json.loads((RAW / 'atria' / site_id / 'lookup.json').read_text())


def rows(site_id: str) -> tuple[list, dict]:
    rec, sc = record(site_id), screen(site_id)['screens']
    out, flags = [], {}
    val = lambda k: (sc.get(k) or {}).get('value') or {}   # noqa: E731
    date = lambda k: ((sc.get(k) or {}).get('fetched_at') or '')[:10] or None   # noqa: E731

    def A(dm, name, value, module, unit='', note='', status='Answered'):
        out.append(row(site_id, dm, name, value, unit=unit, source=f'Atria: {MODULE_NAMES.get(module, module)}', how='Atria module', date=date(module) or '2026-10-28',
                       status=status, note=note, url=(sc.get(module) or {}).get('source_url') or ''))

    # Technical
    w, p = val('wetlands'), rec['parcel']['data']
    A('technical.parcel', 'owner', lookup(site_id)['parcel']['value'].get('owner_name') or p.get('ownerName'), 'assessorRecord')
    A('technical.parcel', 'parcel_acres', round(w.get('parcelAcres') or p.get('gisAcres'), 1), 'assessorRecord', 'acres')
    if w.get('uplandAcres') is not None:
        A('technical.parcel', 'usable_acres', round(w['uplandAcres'], 1), 'wetlands', 'acres', 'Parcel acres less mapped wetlands (NWI).')
    av = val('adjoiningParcels')
    adj = av.get('parcels')
    if adj is not None:
        A('technical.adjoining_parcels_assemblage', 'adjoining_parcels', len(adj), 'adjoiningParcels', 'parcels')
        A('technical.adjoining_parcels_assemblage', 'same_owner_acres', round(sum(a.get('acres') or 0 for a in adj if a.get('sameOwner')), 1), 'adjoiningParcels', 'acres')
    elif av.get('sameOwnerAdjoiningAcres') is not None:   # newer screens ask for the counts only (the neighbours' outlines are not needed)
        A('technical.adjoining_parcels_assemblage', 'same_owner_acres', round(float(av['sameOwnerAdjoiningAcres']), 1), 'adjoiningParcels', 'acres',
          f'{av.get("sameOwnerAdjoiningCount")} adjoining parcels held by the same owner.' + (f' Institutional neighbours: {", ".join(av["institutionalNeighbours"])}.' if av.get('institutionalNeighbours') else ''))
    ease = val('easementIndicators').get('indicators') or []
    A('technical.easements_and_lines', 'easements_crossing', len(ease), 'easementIndicators', 'lines',
      '; '.join(f'{e["kind"].replace("_", " ")}{" " + e["detail"] if e.get("detail") else ""}{" (" + e["operator"] + ")" if e.get("operator") and e["operator"] != "NOT AVAILABLE" else ""}' for e in ease))
    subs = val('substations').get('nearest') or []
    hv = [s for s in subs if (s.get('maxVoltageKv') or 0) >= 345]
    d345 = min((s['distanceMi'] for s in hv), default=None)
    prox = RAW / 'atria' / site_id / 'proximity_345kv.json'
    if d345 is None and prox.exists():   # the screen lists the nearest substations of any voltage; atria_proximity finds the nearest 345 kV
        pj = json.loads(prox.read_text())
        c = pj['candidates'][0]
        out.append(row(site_id, 'technical.substations', 'dist_345kv_mi', c['miles'], unit='mi', source='Atria: nearest 345 kV substation (HIFLD)', how='Atria module',
                       date=pj['fetched_at'][:10], note=f'{c["name"].title()} ({c["maxVoltageKv"]} kV), straight line from the parcel centre.'))
        d345 = c['miles']
    else:
        A('technical.substations', 'dist_345kv_mi', d345, 'substations', 'mi', 'Measured from the parcel outline.')
    lines345 = sorted((l for l in val('transmissionLines').get('lines') or [] if (l.get('voltageKv') or 0) >= 345), key=lambda l: l['distanceMi'])
    if lines345:
        l0 = lines345[0]
        A('technical.substations', 'line_345kv_mi', l0['distanceMi'], 'transmissionLines', 'mi', f'{(l0.get("from") or "?").title()} to {(l0.get("to") or "?").title()} ({l0["voltageKv"]} kV). A tap needs a new substation.')
    if subs:
        s0 = subs[0]
        A('technical.substations', 'nearest_substation', f'{s0.get("name") or "Unnamed"} ({s0.get("maxVoltageKv")} kV, {s0["distanceMi"]} mi)', 'substations')
    lim = C.scoring()['gate_thresholds']['max_345kv_mi']
    flags['no_feasible_voltage'] = dict(value=d345 is None or d345 > lim, status='Answered', source='Atria substations',
                                        note=f'Fails when no 345 kV substation is within {lim} mi.')
    A('technical.grid_headroom_load', 'grid_headroom_mw', None, 'gridCapacity', 'MW', 'Atria gridCapacity has no data for ERCOT or WECC. Unknown is not zero.', status='No source')
    g = val('gasInfrastructure').get('gasPipelines', {}).get('nearest') or {}
    A('technical.gas_pipelines', 'gas_dist_mi', g.get('distanceMi'), 'gasInfrastructure', 'mi', f'{g.get("operator", "")} ({g.get("type", "")}), from the parcel outline.')
    sl = (val('elevation').get('slope') or {})
    A('technical.soils', 'slope_mean_pct', sl.get('meanPercent'), 'elevation', '%', f'Grading difficulty: {sl.get("gradingDifficulty")}. Soil survey (USDA) to follow.')
    # Environmental
    ws = val('waterSewer')
    prov = ws.get('territoryProvider') or ((ws.get('adjacentSystems') or [None])[0])
    A('environmental.water_supply', 'water_system', (prov or {}).get('name'), 'waterSewer', '',
      f'{"Inside its territory" if ws.get("territoryProvider") else str((prov or {}).get("distanceMi")) + " mi away"}. No water rights data.' if prov else 'No public system nearby.')
    if w.get('wetlandPctOfParcel') is not None:
        A('environmental.wetlands', 'wetland_share', round(w['wetlandPctOfParcel'] / 100, 4), 'wetlands', 'share', f'{w["wetlandAcresOnParcel"]} acres of NWI wetland on the parcel.')
    f = val('floodZone')
    thr = C.scoring()['gate_thresholds']['floodway_pct_of_parcel']
    if f.get('mapped'):
        fw = f.get('floodwayPercent') or 0
        fw_ac = sum(z['acres'] for z in f.get('zones') or [] if z.get('subtype') == 'FLOODWAY')
        A('environmental.floodplain', 'flood_zone', f'{f.get("sfhaPercent") or 0}% in the 100-year floodplain; {fw}% floodway ({fw_ac:.1f} acres)', 'floodZone')
        A('environmental.floodplain', 'floodway_pct', fw, 'floodZone', '%')
        flags['in_floodway'] = dict(value=fw > thr, status='Answered', source='Atria floodZone (FEMA NFHL)',
                                    note=f'Fails above {thr}% of the parcel in the floodway; {fw}% here.')
    else:
        rr = val('naturalHazardRisk').get('inlandFloodingRating') or ((val('naturalHazardRisk').get('hazards') or {}).get('inlandFlooding') or {}).get('riskRating')
        A('environmental.floodplain', 'flood_zone', 'No FEMA digital flood map' + (f'; river flood risk {rr.lower()} for the area (FEMA NRI)' if rr else ''), 'floodZone', '',
          'FEMA has no digital flood map for this parcel. Confirm with the paper FIRM at msc.fema.gov before design.')
        flags['in_floodway'] = dict(value=False, status='Answered', source='Atria: flood zones (FEMA)',
                                    note='No regulatory floodway: FEMA designates floodways only where it has mapped the area in detail, and it has not here.' + (f' River flood risk for the area: {rr} (FEMA National Risk Index).' if rr else ''))
    aq = val('airQuality')
    des = aq.get('designations') or []
    A('environmental.air_attainment_status', 'attainment', 'Attainment' if aq.get('attainment') else '; '.join(d['areaName'] for d in des), 'airQuality')
    con = val('contamination')
    n = sum(v for k, v in con.items() if isinstance(v, int))
    A('environmental.contamination', 'contamination_sites_nearby', n, 'contamination', 'sites',
      ', '.join(f'{k} {v}' for k, v in con.items() if isinstance(v, int) and v) or 'None in the screening radius.')
    # Policy
    z = val('zoningJurisdiction')
    dist = z.get('district') or {}
    zone = ('Unzoned county land' if not z.get('insideIncorporatedPlace') and not z.get('countyZoningExists') else
            f'{dist["code"]} ({dist.get("description")}), {z.get("placeBaseName") or z.get("placeName")}' if dist.get('code') else (z.get('placeName') or 'Zoned'))
    A('policy.zoning_status_and', 'zoning', zone, 'zoningJurisdiction', '', z.get('evidence') or '')
    eo = val('energyOrdinance')
    mor = any(((eo.get(k) or {}).get('moratorium') or {}).get('status') not in (None, 'none_found') for k in ('wind', 'solar'))
    A('policy.local_moratoriums_and', 'wind_solar_moratorium', 'Yes' if mor else 'None found', 'energyOrdinance', '',
      'Wind and solar only (NREL 2025); data-center rules are not covered.')
    # Security
    rf = val('regulatedFacilities')
    A('security.critical_infrastructure_exposure', 'regulated_facilities_nearby', sum((v or {}).get('countWithinScreeningRadius') or 0 for v in rf.values() if isinstance(v, dict)),
      'regulatedFacilities', 'facilities', 'RMP, TRI and other EPA-regulated facilities within the screening radius.')
    # Modules only newer screens carry (each replaces an Example value for the same indicator)
    nri = val('naturalHazardRisk')
    if nri.get('riskScore') is not None:
        A('security.natural_hazards', 'hazard_index', nri['riskScore'], 'naturalHazardRisk', '0-100',
          f'FEMA National Risk Index {nri.get("nriVersion")}, census tract {nri.get("tractFips")}: {nri.get("riskRating")}. Top hazards: {", ".join(nri.get("topHazards") or [])}.')
        A('security.natural_hazards', 'hazard_top', ', '.join(nri.get('topHazards') or []), 'naturalHazardRisk')
    cc = val('coolingClimate')
    if cc.get('daysAtOrAbove32C') is not None:
        A('environmental.heat_stress_cooling', 'heat_days', cc['daysAtOrAbove32C'], 'coolingClimate', 'days/yr',
          f'Days at or above 32 °C (90 °F), {cc.get("dailyPeriod")} average. Design wet bulb {cc.get("designWetBulb04PctC")} °C; {cc.get("freeCoolingHoursAtOrBelow15C")} free-cooling hours at or below 15 °C.')
    tel = val('telecomInfrastructure')
    ix = (tel.get('internetExchanges') or [None])[0]
    if ix:
        A('technical.fiber_internet_exchange', 'fiber_ix_dist_mi', ix['distanceMi'], 'telecomInfrastructure', 'mi', f'{ix["name"]} ({ix["participants"]} networks), straight line. PeeringDB.')
        A('technical.fiber_internet_exchange', 'nearest_ix', ix['name'], 'telecomInfrastructure')
        fac = ((tel.get('facilities') or {}).get('nearest') or [None])[0]
        if fac:
            A('technical.fiber_internet_exchange', 'nearest_carrier_facility_mi', fac['distanceMi'], 'telecomInfrastructure', 'mi', f'{fac["name"]} ({fac["operator"]}). PeeringDB.')
    ch = sc.get('criticalHabitat')
    if ch and ch.get('status') == 'ok':
        on = bool(ch['value'].get('intersectsCriticalHabitat'))
        A('environmental.endangered_species_critical', 'critical_habitat_on_site', 'Yes' if on else 'No', 'criticalHabitat', '', f'USFWS designated critical habitat, {ch["value"].get("searchRadiusMi")} mi search.')
        flags['critical_habitat_on_site'] = dict(value=on, status='Answered', source='Atria criticalHabitat (USFWS)', note='Desktop screen; not a survey.')
    ls = val('listedSpecies')
    if ls.get('species') is not None:
        sp = ls['species']
        A('environmental.endangered_species_critical', 'listed_species_count', len(sp), 'listedSpecies', 'species',
          '; '.join(f'{x["commonName"]} ({x["listingStatus"]})' for x in sp))
    geo = val('geologyHazards')
    if geo.get('inKarstArea') is not None:
        A('environmental.endangered_species_critical', 'karst', f'{"Yes" if geo["inKarstArea"] else "No"}{": " + geo.get("karstExposureClass", "") + " " + geo.get("karstType", "") if geo["inKarstArea"] else ""}',
          'geologyHazards', '', 'Karst can host listed cave invertebrates; a karst survey may be required.')
    ws = val('waterSupply')
    if ws.get('surfaceWaterSupplyUseIndex'):
        u = ws['surfaceWaterSupplyUseIndex']
        A('environmental.water_supply', 'supply_use_index', u['meanMonthly'], 'waterSupply', '',
          f'USGS supply/use index, HUC-12 {ws.get("huc12", {}).get("name")}: mean {u["meanMonthly"]}, {u["monthsAbove05"]} of {u["months"]} months above 0.5. Not the WRI baseline water stress score.')
    mb = RAW / 'raw' / site_id / 'military_bases.json'
    if mb.exists():
        import math
        d = json.loads(mb.read_text())
        ring = (lookup(site_id)['parcel']['value'].get('geometry_rings') or [[[rec['lng'], rec['lat']]]])[0]
        def hav(a, b):
            la1, lo1, la2, lo2 = map(math.radians, (a[1], a[0], b[1], b[0]))
            return 2 * 3958.8 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
        best = min(((min(hav(a, b) for a in ring for b in r), f['attributes']['siteName']) for f in d['features'] for r in f['geometry']['rings']), default=None)
        if best:
            out.append(row(site_id, 'security.cfius_real_estate', 'military_dist_mi', round(best[0], 1), unit='mi', source='NTAD Military Bases (USDOT, public)', how='Download',
                           date=d.get('_fetched_at', '2026-10-29')[:10], note=f'Nearest: {best[1]}. Parcel vertices to installation boundary.',
                           url='https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_Military_Bases/FeatureServer/0'))
    zs = RAW / 'atria' / site_id / 'zoning_standards.json'
    if zs.exists():
        z2 = json.loads(zs.read_text())
        out.append(row(site_id, 'policy.zoning_status_and', 'zoning_uses', ', '.join(z2['permittedUses']), source='Atria: zoning standards', how='Atria module',
                       date=z2['fetched_at'][:10], note=f'Data centers are not named; confirm with the city. Extraction confidence: {z2["confidence"]}. {z2["purpose"]}', url=z2['source']))
        if z2.get('overlays'):
            out.append(row(site_id, 'policy.zoning_status_and', 'zoning_overlays', '; '.join(z2['overlays']), source='Atria zoningJurisdiction (COSA layers)', how='Atria module',
                           date=z2['fetched_at'][:10], note='Height limits near the airport approach can apply to stacks and cooling towers.'))
    pp = RAW / 'atria' / site_id / 'permit_path.json'
    if pp.exists():
        r = json.loads(pp.read_text())['permit_path']
        if r.get('status') != 'ok':
            out.append(row(site_id, 'policy.local_approval_path', 'permit_months', None, unit='months', source='Atria: permit path', how='Atria module', date=None,
                           status='No source', note=r.get('reason', '')))
    # Economics (public parcel record)
    mv, ac = p.get('marketValue'), p.get('gisAcres')
    ag = str(p.get('stateLandUse') or '').startswith('D1')
    out.append(row(site_id, 'economics.land_value_vs', 'land_value_per_acre', round(mv / ac) if mv and ac else None, unit='$/acre',
                   source=rec['parcel']['source'], how='Download', date=rec['parcel']['fetched_at'][:10],
                   note='Agricultural (open-space) valuation, far below market.' if ag else 'Appraisal district market value; lags the market.',
                   url=rec['parcel'].get('source_url', '')))
    return out, flags
