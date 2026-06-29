# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------
# Space Hugging Face — API de scoring crédit (P08)
# Variante du Dockerfile racine, adaptée aux contraintes HF Spaces :
#   • le service DOIT écouter sur le port 7860 (port unique exposé par HF) ;
#   • exécution en utilisateur non-root d'UID 1000 (convention HF), AVEC
#     propriété du répertoire de travail pour que l'API puisse y écrire le
#     journal logs/predictions.jsonl (sinon échec d'écriture au démarrage).
#
# Ce fichier attend, à la racine du contexte de build, le contenu assemblé
# par deploy/assemble.py : requirements.txt, app/, model/, features.py.
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp1 : OpenMP requis par LightGBM au runtime (sinon "libgomp.so.1 not found").
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances d'abord (cache Docker tant que requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif + modèle + features.py (requis pour désérialiser le pipeline)
COPY app/ ./app/
COPY model/ ./model/
COPY features.py .

# Utilisateur non-root UID 1000 + propriété de /app : indispensable pour que
# l'API crée et écrive logs/predictions.jsonl (journalisation des prédictions).
RUN useradd --create-home --uid 1000 user && chown -R user /app
USER user

EXPOSE 7860

# Healthcheck aligné sur /health (sans installer curl)
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
