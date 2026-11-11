# FlareNet API. It serves the forecast and the alert history; it does not run the model, so there is no PyTorch here.
#   docker build -t flarenet-api backend
#   docker run -p 8000:8000 -e FRONTEND_ORIGIN=http://localhost:5173 flarenet-api
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data/forecast.json ./data/forecast.json

# the alert store (SQLite) lives in /srv/db: mount a volume there to keep it across container restarts
RUN useradd --system --uid 10001 --home-dir /srv app && mkdir -p /srv/db && chown -R app /srv/db /srv/data
USER app
ENV DATABASE_URL=sqlite:////srv/db/flarenet.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=4)"

# $PORT is set by hosts such as Render; 8000 otherwise
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
