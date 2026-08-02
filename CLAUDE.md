# TerraOps — instructions projet

Projet d'apprentissage MLOps autour d'un classifieur EuroSAT (ResNet-18, 97.8 % test).
Le modèle n'est PAS le sujet : le sujet, c'est l'outillage autour (repro, traçabilité,
gouvernance). Le propriétaire du repo est en préparation d'entretiens MLOps.

## Mode tuteur (obligatoire)

- Expliquer les concepts AVANT de coder ; avancer par petits incréments.
- Poser des questions de compréhension et attendre les réponses avant d'implémenter.
- Justifier chaque choix technique ; terminer chaque sprint par un récap + questions
  type recruteur.
- Recadrer si la discussion dérive vers le tuning du modèle : c'est un projet MLOps.
- Documenter les leçons dans `learning.md` (journal d'apprentissage, en français).

## Architecture

- `dvc.yaml` : DAG `prepare → train → evaluate`. `params.yaml` = source unique de
  vérité des hyperparamètres (aucun nombre magique dans le code).
- `docker-compose.yml` : MinIO (buckets `terraops-dvc` = données DVC,
  `terraops-mlflow` = artefacts MLflow — séparés volontairement), PostgreSQL
  (backend store MLflow), serveur MLflow 3.4.0 sur http://localhost:5000
  (mode proxied artifacts : les clients n'ont besoin que de `MLFLOW_TRACKING_URI`).
- `src/train.py` : 1 exécution = 1 run MLflow (expérience `terraops-eurosat`) avec
  tags de lineage `git_commit` + `dvc_data_hash` posés AVANT l'entraînement.
- `src/promote.py` : gate champion/challenger sur le jeu figé
  `gate/frozen_val.json` (commité ; ne JAMAIS le régénérer sans re-baseliner le
  champion). Seuils dans `params.yaml` section `promote`. Refus = exit 1.
- Registry : `terraops-eurosat`, alias `@champion`. Versions refusées conservées
  avec tag `gate_result: refused`.
- `src/preprocessing.py` : contrat UNIQUE train/serving (Sprint 3). Possède toute la
  chaîne bytes → tensor (décodage RGB/EXIF inclus, resize bilinear épinglé). Importé
  par `dataset.py` (branche éval), le gate et l'API → skew impossible par construction.
  NE JAMAIS redéfinir la transform d'inférence ailleurs.
- `src/api.py` : FastAPI. Charge le modèle par ALIAS (`models:/terraops-eurosat@champion`),
  jamais un `.pth`. Endpoints `/health` (liveness, toujours 200), `/model-info`,
  `/predict`, `/predict/batch`, `/reload` (hot-swap si la version d'alias a changé).
  Démarrage dégradé : boote sans modèle, 503 sur predict/model-info jusqu'à chargement.
- `ui/streamlit_app.py` : client LÉGER de l'API (pas de modèle, image sans torch).
  Upload → `/predict` ; grille de tuiles → `/predict/batch` + carte folium.
- `tests/` : unitaires preprocessing + contrat API (toujours) ; non-régression du
  champion servi sur le jeu figé (auto-skip si MLflow/données absents). Seuils dans
  `params.yaml` section `nonreg`.

## Workflow d'expérience (à respecter strictement)

1. Modifier UN facteur dans `params.yaml` (budget commun actuel : `epochs: 6`,
   `patience: 3` — machine CPU-only, ~15 min/epoch pour ResNet-18 éveillée).
2. `git commit` AVANT `dvc repro` (sinon lineage invalide ; `git_dirty` ignore les
   sorties du pipeline dvc.lock/metrics mais pas le code/params).
3. `dvc repro`, puis commit de résultats : `git add dvc.lock metrics && git commit`.
4. `dvc push` après chaque expérience.
5. Promotion : `python src/promote.py --run-id <RUN_ID>` (jamais d'alias posé à la main).

## Commandes

- Stack : `docker compose up -d` / `docker compose stop` (jamais `down -v` : détruit
  l'historique). Santé : GET http://localhost:5000/health.
- Pipeline : `dvc repro` ; données : `dvc push` / `dvc pull`.
- Ne pas couper Docker pendant un `dvc repro` (le logging MLflow ferait planter le run).
- Serving (Sprint 3) : API sur http://localhost:8000 (`/docs` = Swagger), UI sur
  http://localhost:8501. Images : `Dockerfile.api` (`requirements-api.txt`, torch CPU),
  `Dockerfile.ui` (`ui/requirements.txt`, sans torch). Rebuild : `docker compose build
  api ui`.
- Après promotion d'un nouveau champion : `curl -X POST http://localhost:8000/reload`
  (l'API sert la nouvelle version sans redémarrage). Ne PAS déplacer `@champion` à la
  main pour tester — utiliser un alias jetable.
- Tests : `python -m pytest` (non-régression skip si stack down ; `docker compose up -d`
  d'abord pour les exécuter — ~5 min, scoring des 4050 du jeu figé).

## État (fin Sprint 3, 2026-08-01)

- Sprint 1 : pipeline DVC reproductible (repro bit-à-bit du modèle de référence).
- Sprint 2 : clos. Champion = v1 (backfill Sprint 1, 98.10 % sur jeu figé).
  Campagne de 5 expériences (lr ×3, augmentation, gel, MobileNet) : aucune n'a battu
  le champion ; 3 refus du gate documentés (v2, v3, v4). Détails dans `learning.md`.
- Sprint 3 : clos (2026-08-01). Serving en place : preprocessing partagé (anti-skew),
  API FastAPI chargeant `@champion` par alias avec hot-swap `/reload`, UI Streamlit
  cliente + carte folium, tests unitaires + non-régression, images Docker slim (API
  2.53 Go torch-CPU, UI 784 Mo sans torch). Test d'acceptation validé en conteneurs.
  Détails + questions recruteur dans `learning.md`.
- Dette connue : images 64×64 natives upscalées à 224 (~12× de calcul) — ne pas
  changer en cours de campagne comparative. `/reload` manuel (pas de TTL/webhook).
  API 2.53 Go (piste : `mlflow-skinny`).
- Sprint 4 : à cadrer — CI qui rejoue gate/non-régression sur PR (tests lents ~5 min :
  nightly vs bloquant), monitoring de drift en prod.
