# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------
# Credit Scoring API — Prêt à dépenser (P08)
# Image exposant l'API REST FastAPI (/predict, /health)
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1 : librairie OpenMP REQUISE par LightGBM au runtime.
# Sans elle, l'import du modèle échoue avec une erreur "libgomp.so.1 not found"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances d'abord : tant que requirements.txt ne change pas,
# Docker réutilise le cache et ne réinstalle pas tout à chaque build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif + modèle exporté du P06
COPY app/ ./app/
COPY model/ ./model/
# features.py est requis pour désérialiser le pipeline (FunctionTransformer)
COPY features.py .

# Exécution en utilisateur non-root (bonne pratique sécurité)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8800

# Healthcheck aligné sur l'endpoint /health (sans installer curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8800/health').status==200 else 1)"

# L'API FastAPI sert les endpoints REST (/predict, /health).
# L'interface Streamlit tourne séparément (hors de ce conteneur).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8800"]
