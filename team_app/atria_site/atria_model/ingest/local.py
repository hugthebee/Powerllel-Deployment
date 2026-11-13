"""Rows with no free source wired yet whose value the score needs: Example values from mock_values.json ('local'), each naming
what would answer it."""
from __future__ import annotations

from .base import MOCK_DATE, Source, mock_for, row

SPECS = [  # name, data-map row, unit, what would answer it, how
    ('noise_dba_home', 'cultural.noise_exposure', 'dBA', 'building footprints + ISO 9613-2 propagation', 'Derived (model)'),
    ('burden_pct', 'cultural.community_burden', 'pctile', 'EJScreen mirror / Census ACS', 'Download'),
    ('permit_months', 'policy.local_approval_path', 'months', 'Atria: permit path (not researched for this jurisdiction yet)', 'Atria module'),
    ('military_dist_mi', 'security.cfius_real_estate', 'mi', '31 CFR Part 802 site list + DoD installation boundaries (MIRTA)', 'Download'),
    ('projects_stuck_in_county', 'economics.projects_stuck_waiting', 'projects', 'Comptroller + TCEQ + PUCT + agendas, entity matching', 'Crawled'),
]

GAPS = Source('local_examples', 'Not built yet: noise model, EJ burden, military sites, project matching', '', [s[1] for s in SPECS if s[1] != 'policy.local_approval_path'],
              'Derived (model)', note='Example values stand in; see each row for what would answer it.')


def rows(site_id: str) -> list:
    v = mock_for(site_id, 'local')
    return [row(site_id, dm, name, v.get(name), unit=unit, source=f'Example (would read {what})', how=how, date=MOCK_DATE if v.get(name) is not None else None,
                status='Example' if v.get(name) is not None else 'Not pulled', note='Made-up value for the demo; replaced when the source is pulled.')
            for name, dm, unit, what, how in SPECS]
