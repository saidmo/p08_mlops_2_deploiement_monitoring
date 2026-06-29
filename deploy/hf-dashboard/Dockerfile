# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------
# Space Hugging Face — Dashboard de monitoring (P08)
# Service Streamlit autonome qui interroge l'API distante via GET /logs.
# Aucune dépendance au modèle (ni LightGBM, ni le .pkl) : ce conteneur ne fait
# qu'afficher des données reçues par HTTP.
#
# Attend, à la racine du contexte de build (assemblé par deploy/assemble.py) :
# requirements.txt et dashboard_streamlit.py.
# ----------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard_streamlit.py .

RUN useradd --create-home --uid 1000 user && chown -R user /app
USER user

# URL de l'API distante (Space FastAPI). Ajustable ici ou via les "Variables"
# du Space sur Hugging Face. Le dashboard lit cette variable pour appeler /logs.
ENV API_URL=https://saidmo2-p08-mlops-deploiement-monitoring-fastapi.hf.space

EXPOSE 7860

# Options Streamlit nécessaires sur HF : port 7860, écoute sur 0.0.0.0, mode
# headless, et désactivation CORS/XSRF pour l'affichage dans l'iframe HF.
CMD ["streamlit", "run", "dashboard_streamlit.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
