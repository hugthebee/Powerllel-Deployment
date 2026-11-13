"""Customer-facing names for indicators. Module names and internal terms never appear here; they live in hover text and Methods."""
LABELS = {
    'owner': 'Owner of record', 'parcel_acres': 'Parcel size', 'usable_acres': 'Usable acres (excluding wetlands)',
    'adjoining_parcels': 'Adjoining parcels', 'same_owner_acres': 'Same owner next door', 'easements_crossing': 'Lines crossing the parcel',
    'dist_345kv_mi': 'Distance to a 345 kV substation', 'nearest_substation': 'Nearest substation', 'grid_headroom_mw': 'Spare grid capacity',
    'zone_queue_gw': 'Large loads queued in the zone', 'grid_p50_months': 'Grid arrival, median', 'bridge_mw_years': 'On-site power needed',
    'onsite_mw': 'On-site capacity', 'power_price_basis': 'Power prices used', 'gas_dist_mi': 'Distance to a gas pipeline',
    'fiber_ix_dist_mi': 'Distance to an internet exchange', 'nearest_ix': 'Nearest internet exchange', 'slope_mean_pct': 'Average slope',
    'water_stress_0_5': 'Baseline water stress', 'water_system': 'Nearest public water system', 'cooling_water_mgd': 'Cooling water demand',
    'heat_days': 'Hot days a year', 'wetland_share': 'Share of parcel in wetlands', 'critical_habitat_on_site': 'Critical habitat on site',
    'listed_species_count': 'Listed species that may occur', 'flood_zone': 'Flood zone', 'floodway_pct': 'Floodway share of parcel',
    'attainment': 'Air quality status', 'air_major_tpy': 'Major-source NOx line', 'similar_permits_count': 'Similar turbine permits issued',
    'median_nox_ppm': 'Typical NOx limit granted', 'air_path': 'Air permit path', 'contamination_sites_nearby': 'Contamination records nearby',
    'co2_kg_mwh': 'Carbon intensity of campus power', 'share_oppose': 'Share of statements opposed', 'net_stance': 'Net stance',
    'volume_trend': 'Change in public discussion', 'burden_pct': 'Community burden percentile', 'noise_dba_home': 'Noise at the nearest home',
    'zoning': 'Zoning', 'wind_solar_moratorium': 'Wind and solar moratorium', 'moratorium_stage': 'Data center moratorium stage',
    'permit_months': 'Months to local approval', 'dc_incentive': 'State data center incentive', 'military_dist_mi': 'Distance to a military site',
    'hazard_index': 'Natural hazard index', 'hazard_top': 'Main hazards', 'regulated_facilities_nearby': 'Regulated facilities nearby',
    'pending_nsr_in_county': 'Pending air permits in the county', 'qualifying_dcs_in_county': 'Qualifying data centers in the county',
    'projects_stuck_in_county': 'Projects waiting for power nearby', 'land_value_per_acre': 'Appraised land value',
    'nearest_carrier_facility_mi': 'Nearest carrier facility', 'karst': 'Karst terrain', 'moratorium_history': 'Data center moratorium history', 'data_center_approval': 'Data center approval', 'state_rules': 'State and grid rule changes', 'zoning_uses': 'Uses the district permits', 'zoning_overlays': 'Overlay districts', 'line_345kv_mi': 'Distance to a 345 kV line', 'supply_use_index': 'Surface water supply/use index',
    'breakeven_usd_mwh': 'Breakeven bridge price', 'ownership': 'Buyer ownership (SB 17 / CFIUS)',
}
STAGE_WORDS = {'none': 'None found', 'public_request': 'Public request', 'staff_study': 'Staff study', 'first_reading': 'First reading',
               'adopted': 'Adopted', 'extended': 'Extended', 'repealed': 'Repealed', 'rejected': 'Rejected'}


def label(name: str) -> str:
    return LABELS.get(name) or name.replace('_', ' ').capitalize()
