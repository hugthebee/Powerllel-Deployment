FROM node:22-bookworm-slim AS frontend
WORKDIR /build/faraday_web
COPY faraday_web/package.json faraday_web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY faraday_web/ ./
RUN npm run build
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg PORT=10000
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend /build/faraday_web/dist ./faraday_web/dist
RUN mkdir -p /app/outputs
EXPOSE 10000
CMD ["sh", "-c", "exec python team_app/gridtech_server.py --host 0.0.0.0 --port ${PORT:-10000}"]
