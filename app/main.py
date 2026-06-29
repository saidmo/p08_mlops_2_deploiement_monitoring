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


class _CleanJsonFormatter(JsonFormatter):
    """JsonFormatter qui n'émet pas le champ `message` (inutile ici : toutes
    les données utiles passent par `extra`)."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record.pop("message", None)


if not pred_logger.handlers:  # évite la duplication de handlers au rechargement
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _handler.setFormatter(_CleanJsonFormatter("%(message)s"))
    pred_logger.addHandler(_handler)


def _log_prediction(payload: dict) -> None:
    """
    Écrit une ligne JSON par prédiction.

    Best-effort : si l'écriture du log échoue, la prédiction NE doit PAS
    échouer pour autant (une panne de journalisation ne casse pas le service).
    """
    try:
        # Chaîne vide : on ne veut PAS de champ "message" parasite dans la
        # ligne JSON (les données utiles sont passées via `extra`).
        pred_logger.info("", extra=payload)
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
async def journaliser_et_chronometrer(request: Request, call_next):
    """
    Mesure la latence END-TO-END de la requête (réseau d'entrée, validation,
    scoring, sérialisation) et journalise les prédictions.

    L'endpoint /predict dépose les éléments à logger dans `request.state` ;
    ce middleware y ajoute la latence réelle — qu'il est le seul à connaître,
    puisqu'il enveloppe tout le traitement — puis écrit la ligne JSON.
    """
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"

    # Journalise uniquement si l'endpoint a déposé une prédiction à logger
    record = getattr(request.state, "prediction_log", None)
    if record is not None:
        record["latency_ms"] = round(latency_ms, 3)
        record["http_status"] = response.status_code
        _log_prediction(record)

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
def predict(client: ClientFeatures, request: Request):
    """
    Score une demande de crédit.

    La validation Pydantic (champs requis, plages, types) s'exécute AVANT
    cette fonction : une entrée invalide renvoie automatiquement un 422.

    L'inférence pure est chronométrée ici ; la latence end-to-end et
    l'écriture du log sont gérées par le middleware (qui enveloppe tout le
    traitement de la requête).
    """
    # Features réellement soumises (noyau + extras éventuels) ; les None sont
    # exclus pour être complétés à NaN par le réalignement côté model.py.
    features = client.model_dump(exclude_none=True)

    t_inf = time.perf_counter()
    result = model.predict_one(features)
    inference_ms = (time.perf_counter() - t_inf) * 1000

    # Dépose le contenu à journaliser ; le middleware ajoutera latency_ms
    # et http_status, puis écrira la ligne JSON.
    request.state.prediction_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": uuid.uuid4().hex[:12],
        "model_version": result["model_version"],
        "inputs": features,
        "proba_defaut": result["proba_defaut"],
        "decision": result["decision"],
        "seuil": result["seuil"],
        "inference_ms": round(inference_ms, 3),
    }

    return PredictionResponse(**result)


@app.get("/logs")
def logs(limit: int = 1000):
    """
    Renvoie les `limit` dernières prédictions journalisées (JSON brut, une
    liste d'objets), pour alimenter le dashboard de monitoring à distance.

    - `limit` est borné à [1, 5000] pour éviter de renvoyer un volume excessif.
    - Si le journal n'existe pas encore (API fraîchement démarrée, aucun
      trafic), renvoie une liste vide plutôt qu'une erreur.
    - Lecture tolérante : une ligne illisible est ignorée, pas fatale.
    """
    import json

    limit = max(1, min(limit, 5000))
    if not LOG_FILE.exists():
        return {"count": 0, "predictions": []}

    with LOG_FILE.open(encoding="utf-8") as f:
        lignes = f.readlines()

    recentes = lignes[-limit:]
    predictions = []
    for ligne in recentes:
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            predictions.append(json.loads(ligne))
        except json.JSONDecodeError:
            continue  # ligne corrompue : on l'ignore

    return {"count": len(predictions), "predictions": predictions}
