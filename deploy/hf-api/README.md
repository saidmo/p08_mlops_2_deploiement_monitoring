---
title: P08 Scoring Credit API
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# API de scoring crédit — P08 (Prêt à Dépenser)

API REST FastAPI exposant un modèle de scoring crédit (LightGBM). Déployée sur
Hugging Face Spaces (SDK Docker, port 7860).

## Endpoints

- `GET /health` — sonde de disponibilité (confirme que le modèle est chargé).
- `POST /predict` — score une demande de crédit ; renvoie la probabilité de
  défaut et la décision (seuil métier 0,49).
- `GET /logs?limit=N` — renvoie les N dernières prédictions journalisées (JSON),
  consommées par le dashboard de monitoring (Space séparé).
- `GET /docs` — documentation interactive Swagger.

## Architecture

Ce Space héberge **uniquement l'API**. Le dashboard de monitoring est déployé
dans un Space distinct qui interroge cette API via `GET /logs` — deux services
indépendants communiquant par HTTP.

Le code source complet (avec monitoring, optimisation et CI/CD) est sur le dépôt
GitHub du projet ; ce Space est généré automatiquement par le pipeline CI/CD.

> Données de démonstration (Home Credit, publiques). Le système de fichiers d'un
> Space est éphémère : les journaux sont réinitialisés à chaque redémarrage.
