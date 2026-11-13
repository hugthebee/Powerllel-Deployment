# Validation

- npm ci and npm run build: passed (Vite production build).
- Existing server started with --host 0.0.0.0 --port 8771.
- HTTP GET /, /api/health, /api/meta, /api/stress, /api/runs: all 200; API responses parsed as JSON.
- Existing stress-event suite: 49 passed, 2 skipped (full solver runs are opt-in).
- Full optimization was not run. Docker container build was not run because Docker was unavailable.
- Local checks used the available Node/Python runtime, not the Docker image's exact versions.
- Missing model_alt/gridmodel.py prevents complete financial-agent validation.
