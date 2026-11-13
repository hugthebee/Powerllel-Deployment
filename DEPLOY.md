# Deploy Faraday / Powerllel

The uploaded frontend, server and packet data are preserved. No optimizer logic has been changed. Added: Dockerfile, requirements.txt, render.yaml, root ignore files and deployment/validation notes.

## GitHub

Extract this ZIP. Create a GitHub repository. From the extracted folder containing Dockerfile, run:

```bash
git init
git add .
git commit -m "Prepare Faraday for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Substitute your repository URL. A private repository works. Push extracted files, not the ZIP. node_modules, Mac Node binaries, virtual environments and outputs are excluded.

## Render

At https://dashboard.render.com choose New > Web Service. Connect your GitHub repository. Select Docker runtime, keep Root Directory empty (repository root), Dockerfile Path ./Dockerfile, and Health Check Path /api/health. Leave Docker Command empty. Deploy. Alternatively create a Blueprint using render.yaml.

Docker builds React with Node 22 and runs the Python server on 0.0.0.0 using Render's PORT. Both the website and /api use the same URL, so no CORS or frontend API URL changes are needed. This server uses ThreadingHTTPServer, not Flask/FastAPI: do not use a Gunicorn start command.

To preserve previous runs, attach a persistent disk at /app/outputs; Render requires a paid service for disks. Otherwise run history is temporary. Use one instance: jobs and the single-run lock are in process memory. In-progress jobs cannot survive restarts.

## Verify after deployment

Open the site and /api/health. Load the form, submit a default scenario, watch it complete, and check charts and downloads. Measure a complete solve before choosing CPU/memory resources. Startup alone does not prove the hosting tier can complete optimization.

## Missing financial-agent dependency

team_app/alt_adapter.py imports ../model_alt/gridmodel.py. The uploaded files do not include model_alt. The core optimizer starts without it, but financial-agent features import it later through faraday_service.py. Add your original model_alt folder at repository root before expecting those features to work. Some legacy modules also expect root starter/ and data/ folders; review those paths when enabling this feature. The core optimizer uses the included packet_v6.6.1.

## Local Docker check

```bash
docker build -t faraday .
docker run --rm -p 8770:10000 faraday
```

Open http://localhost:8770. Mount a directory at /app/outputs if you want persistent local results.

## Scope

This retains your existing threaded HTTP server for an initial controlled demo. Authentication, user-isolated run history and multi-instance job management have not been added. Existing saved runs are shared through the API; consider that before entering private customer data.

Docker was unavailable during preparation, so the complete container build was not tested. See VALIDATION.md for completed checks. Python dependency ranges are not a fully locked environment; freeze versions from a successful deployment for reproducibility.
