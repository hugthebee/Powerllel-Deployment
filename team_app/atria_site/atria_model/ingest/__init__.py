"""Ingest: one module per source. Each declares a Source (name, url, data-map rows it answers, how) and returns Indicator rows
for a site. This build runs in MOCK mode: no network calls. Sources already read (the cached Atria screens and public layers
from atria-power-desk) give Answered rows; the free public sources give Example rows from data/examples/mock_values.json and
report status 'Not pulled'; data-map rows nobody answers get 'No source' rows so the gap stays visible."""
