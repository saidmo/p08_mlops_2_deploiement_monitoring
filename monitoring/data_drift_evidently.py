"""
monitoring/data_drift_evidently.py

Analyse de data drift avec Evidently (API 0.7.x). Compare la distribution
des données de PRODUCTION (extraites du journal logs/predictions.jsonl) à la
RÉFÉRENCE d'entraînement (data/reference_sample.parquet).

Deux rapports HTML sont produits dans monitoring/reports/ :
  1. data_drift_all.html            — référence vs TOUTE la production (1000)
  2. data_drift_vague_derivee.html  — référence vs vague DÉRIVÉE (500 dern.)

Le second montre un drift fort et net (scénario « récession » simulé), le
premier un drift moyenné — illustrant qu'Evidently détecte bien la dérive
quand elle est présente.

Usage (venv activé, à la racine du projet) :
    python monitoring/data_drift_evidently.py
"""
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parent.parent
REF_FILE = ROOT / "data" / "reference_sample.parquet"
LOG_FILE = ROOT / "logs" / "predictions.jsonl"
OUT_DIR = ROOT / "monitoring" / "reports"


def load_reference() -> pd.DataFrame:
    ref = pd.read_parquet(REF_FILE)
    # Le drift de DONNÉES porte sur les features, pas sur le label
    return ref.drop(columns=["TARGET"], errors="ignore")


def load_production() -> tuple[pd.DataFrame, pd.Series]:
    log = pd.read_json(LOG_FILE, lines=True)
    log["timestamp"] = pd.to_datetime(log["timestamp"])
    log = log.sort_values("timestamp").reset_index(drop=True)
    # Le champ `inputs` (dict par ligne) -> DataFrame de features
    features = pd.json_normalize(log["inputs"].tolist())
    return features, log["proba_defaut"]


def run_drift(reference: pd.DataFrame, current: pd.DataFrame,
              titre: str, fichier: Path) -> None:
    # Colonnes communes, en retirant celles entièrement vides d'un côté
    common = [c for c in reference.columns if c in current.columns]
    valides = [c for c in common
               if reference[c].notna().any() and current[c].notna().any()]
    ref, cur = reference[valides], current[valides]

    report = Report([DataDriftPreset()])
    # API 0.7.x : ordre (current, reference) — production d'abord !
    result = report.run(cur, ref)
    result.save_html(str(fichier))
    print(f"  {titre:22s} : {len(valides)} features comparées → {fichier.name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reference = load_reference()
    prod_all, proba = load_production()
    prod_drift = prod_all.tail(500)

    # Vérification du découpage des vagues (doit montrer un net écart)
    print("Contrôle du découpage des vagues :")
    print(f"  proba moyenne 500 premières (normale) : {proba.iloc[:500].mean():.3f}")
    print(f"  proba moyenne 500 dernières (dérivée) : {proba.iloc[500:].mean():.3f}")
    print("Génération des rapports de drift :")

    run_drift(reference, prod_all, "Toute la production", OUT_DIR / "data_drift_all.html")
    run_drift(reference, prod_drift, "Vague dérivée", OUT_DIR / "data_drift_vague_derivee.html")

    print(f"✅ Rapports écrits dans {OUT_DIR}")


if __name__ == "__main__":
    main()
