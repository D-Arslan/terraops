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
- `src/image_features.py` : contrat UNIQUE du monitoring (Sprint 4) — même rôle que
  `preprocessing.py` mais pour la dérive. `FEATURE_NAMES` est la source de vérité :
  les colonnes Postgres et le mapping Evidently en sont dérivés. Importé par l'API
  (par requête), le builder de référence et le rapport. NE JAMAIS redéfinir une
  feature ailleurs : on mesurerait l'écart entre deux extracteurs, pas entre deux
  distributions.
- `src/prediction_log.py` : écriture non bloquante dans `monitoring.predictions`
  (schéma dédié DANS la base MLflow — une autre base imposerait `down -v`). File
  bornée + thread ; saturation = lignes DROPPÉES ET COMPTÉES, jamais de blocage de
  `/predict`. Compteurs exposés sur `/monitoring/status`.
- `monitoring/reference.json` : ancre d'ENTRÉE, commitée, construite sur le split
  TRAIN **sans augmentation**. Jumelle de `gate/frozen_val.json` (ancre de
  performance). Ne JAMAIS la remplacer par une fenêtre glissante : la dérive lente
  deviendrait invisible (boiling frog).
- `src/drift_sim.py` : 4 perturbations (`cloud`, `seasonal`, `blur`, `band_shift`),
  intensité 0 = identité stricte, monotones, déterministes via `(seed, index)`,
  label-preserving → covariate shift UNIQUEMENT (jamais de concept drift).
- `src/drift_monitor.py` : déclencheur CT. Son rôle est surtout de REFUSER —
  persistance 3 fenêtres, cooldown 12 h, fenêtre non concluante qui ne réinitialise
  rien, `--dispatch` explicite (dry-run par défaut).
- `.github/workflows/` : `ci.yml` (ruff + pytest + build + smoke test du conteneur,
  runner GitHub) et `retrain.yml` (CT, **runner self-hosted obligatoire** : MinIO et
  MLflow sont locaux).

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
- Tests : `python -m pytest -q -rs` (`-rs` OBLIGATOIRE : sans lui les skips sont
  invisibles, et une suite qui saute son filet ressemble à une suite qui l'exécute).
  Stack allumée : 85 passed, 0 skipped, ~6 min. Stack éteinte : 80 passed, 5 skipped,
  ~1 min 20.
- Monitoring (Sprint 4) : Prometheus sur http://localhost:9090. Postgres publié sur
  le port hôte **55433** (pas 5432 : un PostgreSQL natif Windows l'occupe et les
  connexions atteignent silencieusement la mauvaise base — l'erreur remonte en
  `UnicodeDecodeError` trompeur).
- Chaîne de dérive complète :
  `python src/drift_reference.py` (une fois, commité) →
  `python src/drift_traffic.py --kind cloud --intensity 0.6` (trafic tagué) →
  `python src/drift_report.py --source sim:cloud:0.6` →
  `python src/drift_monitor.py --source ...` (dry-run ; `--dispatch` pour armer).
- Toujours taguer le trafic simulé via `X-TerraOps-Source` : mélanger simulé et réel
  corromprait la référence même à laquelle réagit la boucle CT.

## État (projet CLOS, fin Sprint 4, 2026-08-11)

- Sprint 1 : pipeline DVC reproductible (repro bit-à-bit du modèle de référence).
- Sprint 2 : clos. Champion = v1 (backfill Sprint 1, 98.10 % sur jeu figé).
  Campagne de 5 expériences (lr ×3, augmentation, gel, MobileNet) : aucune n'a battu
  le champion ; 3 refus du gate documentés (v2, v3, v4). Détails dans `learning.md`.
- Sprint 3 : clos (2026-08-01). Serving en place : preprocessing partagé (anti-skew),
  API FastAPI chargeant `@champion` par alias avec hot-swap `/reload`, UI Streamlit
  cliente + carte folium, tests unitaires + non-régression, images Docker slim (API
  2.53 Go torch-CPU, UI 784 Mo sans torch). Test d'acceptation validé en conteneurs.
  Détails + 7 questions recruteur AVEC réponses développées dans `learning.md`.
- Sprint 4 : clos. Journalisation des prédictions, Prometheus, Evidently contre la
  référence TRAIN figée, simulateur de dérive, boucle CT, CI en ligne et VÉRIFIÉE
  (runs #1-#3 verts, smoke test inclus ; 80 passed / 5 skipped mesuré sur runner —
  2 skips MLflow, 3 skips données absentes : les deux garde-fous sont indépendants).
  GIF de démo dans `docs/map.gif`. Détails + questions recruteur dans `learning.md`.

### Résultats mesurés — ne pas les re-dériver, ne pas les arrondir

- Dérive détectée EN AVANCE sur les perturbations radiométriques (nuage, saison,
  décalage de bandes) : alerte à 0.2 d'intensité, accuracy encore à 96-98 % à 0.4.
- **Angle mort du flou** : accuracy 97 % → 61 % alors que la part de features en
  dérive reste à **0.00**. STRUCTUREL (11 features sur 12 sont colorimétriques),
  aucun réglage de seuil ne le corrige.
- **Entropie NON MONOTONE** : 0.021 → 0.145 → 0.008 pendant que l'accuracy passe de
  0.986 à 0.098. Le modèle devient CERTAIN en devenant faux → l'entropie n'arme
  volontairement rien.
- L'axe d'intensité est arbitraire : comparer les avances ENTRE perturbations n'a
  pas de sens, seuls le signe et l'ordre à l'intérieur d'une perturbation en ont.

### Dette connue, assumée et documentée (README, section « LIMITS AND PROTOCOL HONESTY »)

- Angle mort du flou non corrigé (piste : features de texture/fréquence ou dérive
  sur embeddings).
- `retrain.yml` non exécutable sur runner GitHub public (stack locale).
- `/reload` manuel (pas de TTL/webhook).
- Images 64×64 natives upscalées à 224 (~12× de calcul). API 2.53 Go
  (piste : `mlflow-skinny`).
- La dérive est SIMULÉE : pas de flux de production, pas de concept drift, EuroSAT
  est eurocentré. Ne jamais présenter les courbes comme une preuve — ce sont des
  éléments de preuve sur une famille de perturbations choisie par l'auteur.
