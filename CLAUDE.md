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

## État (fin Sprint 2, 2026-07-30)

- Sprint 1 : pipeline DVC reproductible (repro bit-à-bit du modèle de référence).
- Sprint 2 : clos. Champion = v1 (backfill Sprint 1, 98.10 % sur jeu figé).
  Campagne de 5 expériences (lr ×3, augmentation, gel, MobileNet) : aucune n'a battu
  le champion ; 3 refus du gate documentés (v2, v3, v4). Détails dans `learning.md`.
- Dette connue : images 64×64 natives upscalées à 224 (~12× de calcul) — ne pas
  changer en cours de campagne comparative.
- Sprint 3 : à cadrer (serving / CI qui rejoue le gate / monitoring de drift).
