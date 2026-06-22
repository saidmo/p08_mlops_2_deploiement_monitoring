"""
app/main.py — API FastAPI de scoring crédit (Prêt à dépenser).

Connecte le modèle (app.model) au contrat d'entrée/sortie (app.schemas) :
  • GET  /health   — vérifie que l'API et le modèle sont opérationnels
  • POST /predict   — score une demande de crédit et journalise l'appel
  • middleware       — mesure la latence end-to-end de chaque requête
  • logging JSON      — une ligne par prédiction dans logs/predictions.jsonl
                        (inputs, output, latence, temps d'inférence) ; c'est
                        la source de données du monitoring (Étape 3).
"""
import warnings

# Filtre CIBLÉ : warning interne de scikit-learn signalant la perte des noms
# de colonnes entre le ColumnTransformer (sortie NumPy) et LightGBM. Sans
# impact sur la prédiction ; filtré pour garder des logs propres. Doit
# précéder l'import du modèle (qui déclenche le chargement du pipeline).
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request

# Import robuste de python-json-logger (l'emplacement de JsonFormatter
# a changé entre les versions < 3.1 et >= 3.1).
try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:  # python-json-logger < 3.1
    from pythonjsonlogger.jsonlogger import JsonFormatter

from app import model
from app.schemas import ClientFeatures, HealthResponse, PredictionResponse

# ---------------------------------------------------------------------------
# Logging structuré JSON (JSON Lines) — une ligne = une prédiction
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.jsonl"

pred_logger = logging.getLogger("predictions")
pred_logger.setLevel(logging.INFO)
pred_logger.propagate = False
if not pred_logger.handlers:  # évite la duplication de handlers au rechargement
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _handler.setFormatter(JsonFormatter("%(message)s"))
    pred_logger.addHandler(_handler)


def _log_prediction(payload: dict) -> None:
    """
    Écrit une ligne JSON par prédiction.

    Best-effort : si l'écriture du log échoue, la prédiction NE doit PAS
    échouer pour autant (une panne de journalisation ne casse pas le service).
    """
    try:
        pred_logger.info("prediction", extra=payload)
    except Exception:  # noqa: BLE001
        logging.getLogger("uvicorn.error").warning(
            "Échec d'écriture du log de prédiction (sans impact sur la réponse)"
        )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de scoring crédit — Prêt à dépenser",
    description=(
        "Expose le modèle de scoring LightGBM développé au P06. Reçoit les "
        "données d'une demande de crédit et retourne une probabilité de "
        "défaut ainsi qu'une décision (seuil métier 0.49)."
    ),
    version=model.MODEL_VERSION,
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Mesure la latence end-to-end (toutes routes) et l'expose en header."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    """Sonde de disponibilité : confirme que le modèle est chargé en mémoire."""
    return HealthResponse(
        status="ok",
        model_version=model.MODEL_VERSION,
        n_features_attendues=len(model.INPUT_COLS),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    """
    Score une demande de crédit.

    La validation Pydantic (champs requis, plages, types) s'exécute AVANT
    cette fonction : une entrée invalide renvoie automatiquement un 422.
    """
    t0 = time.perf_counter()

    # Features réellement soumises (noyau + extras éventuels) ; les None sont
    # exclus pour être complétés à NaN par le réalignement côté model.py.
    features = client.model_dump(exclude_none=True)

    t_inf = time.perf_counter()
    result = model.predict_one(features)
    inference_ms = (time.perf_counter() - t_inf) * 1000

    latency_ms = (time.perf_counter() - t0) * 1000

    _log_prediction(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": uuid.uuid4().hex[:12],
            "model_version": result["model_version"],
            "inputs": features,
            "proba_defaut": result["proba_defaut"],
            "decision": result["decision"],
            "seuil": result["seuil"],
            "inference_ms": round(inference_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "http_status": 200,
        }
    )

    return PredictionResponse(**result)
