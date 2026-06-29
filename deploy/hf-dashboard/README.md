---
title: P08 Scoring Credit Monitoring
emoji: 📊
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# Dashboard de monitoring — P08 (Prêt à Dépenser)

Dashboard Streamlit de suivi de l'API de scoring crédit. Déployé sur Hugging
Face Spaces (SDK Docker, port 7860).

Il affiche : volume de prédictions, latence (médiane / p95 / max), temps
d'inférence, taux de refus, distribution des scores, répartition des décisions,
et l'évolution du score au fil des requêtes (révélateur de drift).

## Architecture

Ce Space **n'embarque ni le modèle ni les données** : il interroge l'API de
scoring (Space séparé) via son endpoint `GET /logs` pour récupérer les
prédictions journalisées, puis calcule et affiche les indicateurs.

L'URL de l'API est fournie par la variable d'environnement `API_URL`
(définie dans le Dockerfile, modifiable dans les *Variables* du Space).

> Le système de fichiers d'un Space est éphémère : si l'API redémarre, ses
> journaux sont réinitialisés et le dashboard se vide jusqu'au prochain trafic.
