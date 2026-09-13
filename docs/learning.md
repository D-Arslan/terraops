# TerraOps — Learning Log

> Journal d'apprentissage du projet. On y consigne les concepts vus, les décisions
> prises, et les réponses aux questions de compréhension. Le modèle EuroSAT n'est PAS
> le sujet : le sujet, c'est tout ce qui l'entoure (repro, traçabilité, gouvernance,
> service, monitoring, réentraînement).

---

## Sprint 1 — DVC : versionnage des données + pipeline reproductible

### État des lieux du modèle source (réponses à mes questions de diagnostic)

Point de départ : `D:\Arslan\eurosat-classification` (repo Git, 3 commits, propre).

#### Q1 — À quoi ressemble l'arborescence ?

```
eurosat-classification/
├── src/
│   ├── train.py       # boucle d'entraînement + early stopping + checkpoint
│   ├── evaluate.py    # métriques, matrice de confusion, analyse d'erreurs
│   ├── dataset.py     # chargement EuroSAT + split 70/15/15 + augmentations
│   ├── model.py       # ResNet-18 transfer learning (freeze sauf layer4 + fc)
│   ├── utils.py       # set_seed(), setup_logging(), get_device()
│   ├── data/          # (gitignoré) dataset téléchargé par torchvision
│   ├── outputs/       # (gitignoré) best_model.pth (107 Mo) + png + report
│   └── logs/          # logs d'entraînement horodatés
├── notebooks/         # exploration
├── requirements.txt   # dépendances en >= (NON épinglées) ⚠️
├── environment.yml    # conda : python=3.10 (README dit 3.12 ⚠️ incohérence)
└── README.md
```

Découpage code déjà **modulaire** (data / model / train / eval / utils séparés) →
excellente base pour un `dvc.yaml` en 3 étapes.

#### Q2 — Où vivent les hyperparamètres aujourd'hui ?

Deux endroits, et c'est LE point à corriger au Sprint 1 :

**A. Exposés en `argparse` (train.py) — visibles :**

| Param | Défaut | Fichier |
|-------|--------|---------|
| epochs | 25 | train.py:69 |
| lr | 0.001 | train.py:70 |
| batch_size | 64 | train.py:71 |
| seed | 42 | train.py:72 |
| patience (early stop) | 5 | train.py:73 |

**B. EN DUR dans le code — invisibles, non traçables (⚠️ le vrai danger) :**

| Param en dur | Valeur | Fichier |
|--------------|--------|---------|
| weight_decay (Adam) | 1e-4 | train.py:109 |
| scheduler factor | 0.1 | train.py:114 |
| scheduler patience | 3 | train.py:115 |
| split train/val/test | 0.70 / 0.15 / 0.15 | dataset.py:56-58 |
| image size | 224×224 | dataset.py:25,35 |
| augmentations (flip/rot/jitter) | p=0.5, 15°, 0.2… | dataset.py:26-29 |
| normalisation ImageNet | mean/std fixes | dataset.py:17-18 |
| dropout (tête) | 0.3 | model.py:29 |
| num_classes | 10 | train.py:98 |
| pretrained | True | train.py:98 |

> **Leçon clé :** si un hyperparamètre est en dur, DVC ne peut PAS détecter son
> changement → il ne relancera pas l'étape concernée → le pipeline **mentirait**.
> Objectif Sprint 1 : tout remonter dans `params.yaml`. Aucun nombre magique dans le code.

#### Q3 — Comment le dataset est-il stocké ?

- Téléchargé **automatiquement** par `torchvision.datasets.EuroSAT(download=True)`
  (dataset.py:48) à la 1re exécution.
- Sur disque : `src/data/eurosat/2750/<Classe>/*.jpg` (10 dossiers de classes) +
  `src/data/eurosat/EuroSAT.zip`.
- 27 000 images, 64×64 px, RGB, 10 classes.
- Dossier `data/` **gitignoré** → aujourd'hui les données ne sont ni versionnées ni
  partageables de façon reproductible. C'est exactement ce que DVC + remote MinIO va régler.

> **Point de repro subtil :** le téléchargement se fait dans `load_eurosat()`, appelé
> à la fois par train ET evaluate. Pour un DAG propre, on isolera une étape `prepare`
> qui matérialise les données UNE fois ; train/evaluate consommeront ce résultat sans
> re-télécharger.

#### Q4 — Comment lance-t-on l'entraînement ?

Scripts `.py` (pas de notebook pour l'entraînement) :
```bash
python src/train.py --epochs 25 --lr 0.001 --batch-size 64 --seed 42
python src/evaluate.py --checkpoint outputs/best_model.pth
```
⚠️ Les chemins par défaut (`data`, `outputs`) sont relatifs → les scripts sont lancés
**depuis `src/`**. À garder en tête pour définir le `wdir` du pipeline DVC.

#### Bilan reproductibilité (déjà en place vs. à ajouter)

| Bonne pratique | État actuel | Action Sprint 1 |
|----------------|-------------|-----------------|
| seeds torch/numpy/random | ✅ `set_seed()` utils.py:12 | garder, piloter par params |
| cuDNN deterministic | ✅ utils.py:22-23 | garder |
| split seedé | ✅ generator seedé dataset.py:60 | garder, exposer les ratios |
| versions épinglées | ❌ `>=` partout | épingler en `==` (lockfile) |
| Python cohérent | ❌ 3.10 vs 3.12 | trancher une version |
| hyperparams externalisés | ❌ moitié en dur | → `params.yaml` |
| données versionnées | ❌ gitignoré | → `dvc add` + remote MinIO |
| pipeline reproductible | ❌ commandes manuelles | → `dvc.yaml` (prepare→train→evaluate) |

Modèle actuel : **97,8 % test accuracy** (macro F1 0.9779) — c'est notre référence
gelée. On ne le retouche pas.

---

### Concepts vus (à maîtriser pour l'entretien)

**1. Pourquoi Git ne suffit pas pour données/modèles.**
Git garde tout l'historique de chaque blob binaire → le `.git` explose et devient
inclonable. Git est fait pour du texte diffable, pas pour des Go de `.jpg`/`.pth`.

**2. Ce que DVC met où.**
- Dans **Git** : des pointeurs `.dvc` (texte, ~200 o) = `md5 + size + path`.
- Dans le **remote** (MinIO) : les vrais octets.
- Analogie : Git = l'étiquette de suivi du colis ; MinIO = le colis.
- `dvc push` envoie les octets ; `git clone` + `dvc pull` reconstitue tout.

**3. DAG de pipeline.**
Graphe orienté sans cycle. Chaque *stage* déclare `deps` / `params` / `outs`.
Notre DAG : `prepare → train → evaluate`.

**4. Comment DVC décide de relancer une étape.**
Il hashe `deps`, `params`, `outs` et compare au `dvc.lock` :
- tous les hash identiques → étape **sautée** ;
- un hash changé → étape **relancée** + tout l'aval qui en dépend.

**Alternatives écartées et pourquoi :**
| Option | Verdict |
|--------|---------|
| Git-LFS | versionne les gros fichiers mais reste couplé à Git, **pas de pipeline** |
| MLflow artifacts | trace les *expériences* (Sprint 2), ne versionne pas l'*input data* ni le DAG |
| DVC | pointeurs Git + remote + **DAG reproductible** → les 3 à la fois |
> DVC et MLflow sont **complémentaires**, pas concurrents.

---

### Questions de compréhension (répondues)

1. **Après `git clone`, ai-je les 27 000 images ?**
   Non. Git ne contient que le pointeur (`data/raw.dvc` : md5 + size + path, ~200 o).
   Les octets sont sur MinIO. Commande pour les récupérer : **`dvc pull`**.
2. **Je passe `epochs` de 25 à 30 puis `dvc repro`. Que se passe-t-il ?**
   - `prepare` → **sauté** (`epochs` n'est pas sa dépendance).
   - `train` → **relancé** (`epochs` est un param de `train`).
   - `evaluate` → **relancé** (son entrée `models/best_model.pth` a changé → propagation aval).

---

### Test d'acceptation du Sprint 1 — VALIDÉ (smoke test epochs=1)

- [x] `dvc repro` → les 3 étapes s'exécutent, `dvc.lock` créé, modèle + métriques produits.
- [x] `dvc repro` 2e fois → `Stage X didn't change, skipping` ×3 (idempotent).
- [x] `evaluate.batch_size` 64→128 → **seul `evaluate`** se relance (re-run scopé).
- [x] `dvc push` → 27 003 objets / 287 MiB sur le bucket MinIO `terraops-dvc`.

> **Repro exacte confirmée** (2026-07-21) : `dvc repro` sur `epochs=25` (375 min CPU) a
> reconstruit le modèle **au chiffre près** vs le modèle d'origine → 97,80 % test accuracy,
> macro F1 0,9779, **89/4050** mal classés, best val_loss 0,0552. Même seed + stack épinglée
> + split partagé = même modèle. Poussé sur MinIO + GitHub (commit `328d45a`).
> Argument recruteur : « ma pipeline reproduit le modèle de référence bit-pour-bit ».

---

### Récap des concepts (Sprint 1)

| Concept | L'essentiel |
|---------|-------------|
| Git ≠ données | Git garde tout l'historique binaire → repo inclonable. |
| Pointeurs `.dvc` | Git = étiquette (hash) ; remote = colis (octets). |
| Cache DVC | `.dvc/cache` = magasin local content-addressable, jamais dans Git. |
| DAG | `prepare → train → evaluate`, deps/params/outs hashés dans `dvc.lock`. |
| Re-run | hash inchangé → sauté ; changé → relancé + aval. |
| Scoping params | déclarer par étape *seulement* ce qui influence sa sortie. |
| Repro | seeds + versions épinglées (`==`) + cuDNN deterministic + zéro nombre magique. |
| Gouvernance secrets | adresse du remote dans Git, identifiants dans `.dvc/config.local` gitignoré. |
| DVC vs alternatives | Git-LFS (pas de pipeline) ; MLflow (expériences, Sprint 2) ; DVC = les 3. |

## Sprint 2 — MLflow : tracking, registry, lineage, gate de promotion

### Ce qu'on a construit

- **Stack** (docker-compose) : PostgreSQL = *backend store* MLflow (runs, params,
  métriques, registry — relationnel, requêtable) ; MinIO bucket `terraops-mlflow` =
  *artifact store* (courbes, matrices, modèles — des blobs). Mode **proxy**
  (`--serve-artifacts`) : le serveur écrit dans MinIO, les clients n'ont besoin que de
  `MLFLOW_TRACKING_URI` — zéro credential S3 côté client.
- **train.py instrumenté** : 1 exécution = 1 run (params aplatis, métriques par epoch
  avec `step`, courbes + matrice de confusion val en artefacts, meilleur checkpoint
  rechargé puis loggé). **Tags de lineage posés AVANT l'entraînement** : un run qui
  crashe reste traçable.
- **Lineage** = 2 tags par run : `git_commit` (le code) + `dvc_data_hash` (les octets
  de données, lu dans `dvc.lock`). Chaîne d'audit : registry → run → tags →
  `git checkout` + `dvc pull`. Le commit seul ne suffit pas : Git ne contient que le
  *pointeur* vers les données.
- **model.py paramétré** : `arch` (resnet18 | mobilenet_v3_small) et `unfreeze`
  (none | last_block | all) → les expériences d'architecture/gel passent par params.yaml.
- **Gate (`promote.py`)** : duel candidat vs `@champion` sur un **jeu figé**
  (`gate/frozen_val.json`, 4 050 indices commités, estampillés du hash DVC — le gate
  refuse de juger si les données ont changé). Trois règles : plancher absolu (90 %),
  marge ≥ 0.3 pt (au-dessus du bruit), aucune classe ne perd > 3 pts de rappel.
  Refus = exit 1 + version taguée `gate_result: refused` (décision archivée, pas effacée).

### Concepts clés (à maîtriser pour l'entretien)

| Concept | L'essentiel |
|---------|-------------|
| Tracking vs Registry | Cahier de labo (tous les runs, immuables) vs catalogue de gouvernance (modèles nommés, versions, alias). Analogie : commits vs tags de release. |
| Artefact | Fichier produit par un run (blob). Pas dans Git : binaire, lourd, non-diffable, reproductible depuis les sources. Frontière Git : petit + diffable + utile en revue (metrics.json ✅, PNG ❌). |
| Champion/challenger | Le titulaire garde le titre tant qu'un challenger ne gagne pas NETTEMENT sur le jeu figé. Promotion = décision scriptée, reproductible, auditable. |
| Pourquoi un seuil | ±0.1 pt sur 4 050 images ≈ 4 images = bruit de seed. Une victoire dans la bande de bruit n'est pas une victoire. |
| Jeu figé | Mêmes questions d'examen pour tous les duels, jamais vues à l'entraînement, versionnées. Sinon : fuite, ou terrain qui bouge. |
| MLflow + DVC | DVC versionne les ENTRÉES et le procédé (reconstruire) ; MLflow enregistre les SORTIES des exécutions (comparer, gouverner). Le lineage les relie. |
| Backfill | Quand le tracking arrive après un modèle existant, on logge le titulaire rétroactivement avec son VRAI lineage (le commit qui l'a entraîné, pas HEAD). |

### La campagne (budget 6 epochs, CPU) — résultats et leçons

| Run | val_acc pic | Leçon |
|-----|------------|-------|
| champion v1 (25 ep, backfill) | **98.10 %** | le budget de calcul est un hyperparamètre |
| lr 0.001, aug OFF | 97.65 % | même pic que l'exp 2, mais overfitting dès l'ep. 4 (courbes qui divergent, early stop) — l'augmentation ne monte pas le pic, elle REPOUSSE l'overfitting |
| lr 0.0001 | 97.63 % | en fine-tuning, les petits pas gagnent : les poids pré-entraînés partent déjà près du but |
| lr 0.01 (12 ep) | 96.52 % | LR trop grand = oscillation sans fin de course ; le scheduler ReduceLROnPlateau n'a jamais déclenché (il faut patience+1 mauvais epochs CONSÉCUTIFS) |
| mobilenet_v3_small | 95.93 % | entraîné en 25 min (vs ~95) — le trade-off vitesse/précision, chiffré |
| unfreeze none | 88.62 % | dégeler le dernier bloc est LE levier du transfer learning (+9 pts) |

Verdicts du gate : v1 promu (bootstrap) ; v2 (non entraîné, 11.5 %) refusé sur les 3
règles ; v3 (lr 0.01) refusé marge + 2 classes en régression ; v4 (meilleur challenger,
97.65 %) refusé sur la marge SEULE (−0.44 pt) — le refus le plus instructif : « presque
pareil » ≠ « meilleur ».

Piège vécu : `git_dirty` déclenchait sur `dvc.lock`/`metrics/` réécrits par `dvc repro`
lui-même → le check ignore désormais les SORTIES du pipeline ; seule la dérive des
ENTRÉES (code, params) invalide le lineage.

### Questions de compréhension (réponses modèles)

1. **30 runs dont 12 meilleurs que le champion : combien vont au registry ?**
   Zéro ou un. Le registry n'est pas un classement par val_acc ; on y entre par
   DÉCISION (gate passé, usage visé), pas par métrique. Les autres restent au tracking.
2. **« Le PNG de 40 Ko, commitons-le comme metrics.json » ?**
   La frontière n'est pas la taille : petit + DIFFABLE + utile en revue. Le PNG est
   non-diffable et MLflow le compare mieux. Le modèle échoue sur tout (lourd, binaire,
   produit reproductible).
3. **MobileNet +0.2 pt : deux raisons de refuser ?**
   (a) marge sous le bruit (~8 images sur 4 050) ; (b) régression possible sur une
   classe minoritaire masquée par le global. (Bonus : duel valide seulement sur le MÊME
   jeu figé.)
4. **Prouver les données d'entraînement de la v2 ?**
   Registry → run_id → tags `git_commit` + `dvc_data_hash` → `git checkout` (code,
   params, dvc.lock) → `dvc pull` (les octets exacts). Le commit seul ne contient pas
   les données, seulement le pointeur.
5. **lr 0.0001 : forme de courbe et risque symétrique ?**
   Descente lente et régulière ; risque symétrique = sous-convergence (budget épuisé
   avant le plateau). VÉCU : le risque ne s'est pas matérialisé car fine-tuning de poids
   pré-entraînés → petits pas suffisants (97.63 % en 6 ep). Le grain de vérité : reste
   sous le champion 25 epochs.
6. **Sans augmentation : train_acc vs val_acc, et le pic ?**
   train_acc finit par DÉPASSER val_acc (croisement = début de l'overfitting, ep. 4) ;
   pic quasi identique à l'exp 2 mais atteint plus vite, puis val_loss qui remonte
   (0.0716 → 0.1390) pendant que train_loss descend = LA signature à savoir pointer.
   Nuance mesure : train_acc est une moyenne SUR l'epoch (modèle en cours
   d'amélioration), val_acc est mesurée en FIN d'epoch → train sous-estimée.
7. **Tête seule : pic attendu ?**
   Bien plus bas (88.6 %) : un classifieur linéaire sur des features ImageNet gelées ne
   peut pas adapter les représentations aux images satellites.

### Test d'acceptation du Sprint 2 — VALIDÉ

- [x] Lineage 30 s : depuis `models:/terraops-eurosat@champion` → run → tags →
  commandes `git checkout` + `dvc pull` exactes.
- [x] Le gate refuse un modèle moins bon : démontré 3 fois (grossier v2, net v3, serré v4).

## Sprint 3 — Serving : de la gouvernance à la production

Objet du sprint : servir le champion gouverné, sans réintroduire les bugs que
Sprints 1-2 ont bannis. Le modèle ne change pas ; l'outillage autour du serving,
oui.

### Concept n°1 — Train/serving skew (le bug silencieux n°1 en ML prod)

Un modèle n'est pas `model.pth`, c'est `model(preprocessing(input))`. Le skew =
`preprocessing_train ≠ preprocessing_serving`. Il ne CRIE pas : shape valide,
forward OK, une classe sort avec une belle proba — juste fausse plus souvent. Et
en prod on n'a pas les labels, donc on ne mesure pas la chute d'accuracy en direct.
Coût de détection énorme.

Sources concrètes sur images (à savoir citer) : **ordre des canaux** (PIL=RGB vs
cv2=BGR), **normalisation** oubliée (`[0,1]` au lieu des stats ImageNet),
**interpolation** du resize (bilinear vs bicubic — deux personnes « resize en 224 »
ont raison toutes les deux et produisent des tenseurs différents), **canal alpha**
(RGBA→4 canaux), **rotation EXIF** non appliquée.

**La parade structurelle** : UN module `src/preprocessing.py` importé par
`train.py` (via `dataset.py`), par le gate ET par l'API. Il n'y a plus deux
fonctions à synchroniser, il n'y en a qu'une → le skew devient IMPOSSIBLE par
construction, pas « évité par vigilance ». Vérifié : `torch.equal(ancienne_transform,
build_eval_transform) == True` → zéro skew introduit contre le champion v1.

**Le piège que j'ai compris** : la garantie du module s'arrête à sa FRONTIÈRE
d'entrée. Si l'API décode elle-même en BGR avant d'appeler le module, le skew est
né AVANT. → le module doit posséder AUSSI le décodage (bytes → PIL RGB), pas
seulement les transforms. D'où `decode_image` : `convert("RGB")` (tue BGR/alpha/
grayscale) + `exif_transpose` (no-op sur EuroSAT donc zéro skew, mais robuste sur
uploads réels) + interpolation bilinear ÉPINGLÉE (le défaut torchvision a dérivé
entre versions).

### Concept n°2 — Charger par ALIAS depuis le registry

`models:/terraops-eurosat@champion` est un POINTEUR résolu à l'exécution, pas un
chemin. Conséquence le jour où on change de modèle en prod :
- chemin `.pth` en dur → modifier le code, rebuild, redéployer = **déploiement de
  code** ;
- alias → `promote.py` déplace `@champion`, l'API re-résout = **acte de
  gouvernance**, découplé du code. + rollback instantané (repointer l'alias) +
  traçabilité (le registry sait qui est champion, depuis quand, quel run/commit/data).

**Piège de fraîcheur** : l'alias est résolu au `load_model`, donc UNE fois au
startup. Promouvoir à 14h ne notifie pas un process lancé → il sert l'ancien.
Parade : `POST /reload` re-résout et hot-swap SI la version a changé (on compare la
version avant de payer un `load_model` — pas de check registry sur le chemin chaud
`/predict`). C'est ce qui rend vrai « promouvoir → l'API sert le nouveau sans
toucher au code ».

### Concept n°3 — Test de non-régression de MODÈLE

Diffère d'un test unitaire : l'unitaire teste du code déterministe (sortie exacte
connue) ; la non-régression teste une PROPRIÉTÉ STATISTIQUE au-dessus d'un seuil
(accuracy globale, recall PAR CLASSE, invariances, budget de latence). Le point
contre-intuitif : **il peut être ROUGE alors que le code est correct** — car il
surveille le comportement émergent (code + poids + dépendances), pas la logique. Un
`pip install` qui change l'interpolation de Pillow → skew → accuracy du champion qui
chute → test rouge, code inchangé. C'est un détecteur de skew contre notre propre
champion.

Le seuil global seul NE SUFFIT PAS : il noie l'effondrement d'une classe minoritaire
(Highway 94→60 % pendant que la moyenne bouge de 0.3 pt). D'où le plancher PAR
CLASSE — même logique que `max_class_recall_drop` du gate. Ces tests sont la version
pytest/CI de `promote.py` : mêmes seuils (`params.yaml`), même jeu figé.

### Décisions de design (et leurs justifications)

1. **Démarrage dégradé, pas fail-fast.** Dans `docker compose up`, l'API et MLflow
   démarrent ensemble ; fail-fast ferait crash-looper l'API parce que MLflow a booté
   2 s plus tard, ou qu'aucun champion n'est promu. Dégradé : `/health` répond 200
   (liveness) avec `model_loaded: false`, `/predict` et `/model-info` renvoient 503
   (readiness). Récupération par `/reload` sans redémarrage. Healthcheck MLflow dans
   le compose → l'API ne boote qu'une fois le registry prêt (évite le hang de 120 s
   du timeout HTTP MLflow par défaut).
2. **UI = client léger de l'API, JAMAIS de modèle en direct.** Sinon 3e copie du
   modèle + nouvelle surface de skew. Conséquence : l'image UI est SANS torch — elle
   connaît le modèle uniquement par le JSON de l'API (classes = strings). L'UI affiche
   toujours la VERSION servie (traçabilité jusqu'à l'utilisateur).
3. **Image API slim.** `requirements-api.txt` séparé : torch/torchvision **+cpu**
   (index PyTorch CPU, zéro payload CUDA = le plus gros levier), SANS matplotlib/
   seaborn/sklearn/dvc. Mode proxied-artifacts → l'API n'a besoin que de
   `MLFLOW_TRACKING_URI`, AUCUNE credential S3 (le serveur MLflow proxy les artefacts).
   Résultat : API 2.53 Go (mlflow tire pandas/scipy), UI 784 Mo — bien sous les 5 Go.
4. **Démo du hot-swap sur alias JETABLE, jamais `@champion` à la main.** La règle
   « jamais d'alias posé à la main » est de gouvernance. Pour prouver `/reload` sans
   la violer : alias `reload_test` créé → déplacé v1→v2 → `/reload` détecte
   (`reloaded:true, version:2`) → alias supprimé, `@champion` toujours v1.

### Test d'acceptation du Sprint 3 — VALIDÉ (en conteneurs)

- [x] `docker compose up` → API répond, `model_loaded: true`, sert `@champion` v1
  chargé par alias depuis le registry conteneurisé (proxied artifacts, zéro cred S3).
- [x] `/predict` sur vraie tuile EuroSAT (AnnualCrop → AnnualCrop 0.97), `/predict/batch`,
  chaque réponse porte `model_version`.
- [x] UI up (`:8501` → 200), carte folium colorée par usage du sol.
- [x] Hot-swap : alias déplacé → `/reload` sert la nouvelle version SANS toucher au
  code ni redémarrer (prouvé sur alias jetable).
- [x] Tests : 13 passed (preprocessing + contrat API dégradé) + 5 non-régression verts
  contre le champion v1 réel (accuracy > baseline 0.9810, recall par classe, invariances
  hflip/JPEG, latence).

### Dette / pistes Sprint 4

- `/reload` manuel : pas de TTL auto ni de webhook registry (fraîcheur à la demande).
- Image API 2.53 Go : `mlflow-skinny` + flavor PyTorch seul pourrait réduire encore.
- Pas encore de CI qui rejoue le gate/non-régression sur PR (les tests lents ~5 min →
  nightly vs bloquant à décider). Monitoring de drift en prod : non commencé.

---

## Sprint 4 — Monitoring, dérive et Continuous Training

Objet du sprint : rendre le système capable de dire **quand il n'est plus fiable**.
C'est le sprint qui différencie le projet, parce que c'est celui où l'on mesure au
lieu de supposer.

### Concept n°1 — Data drift vs concept drift (et lequel est détectable)

Décomposition : `P(X, Y) = P(X) × P(Y|X)`.

- **Data drift** (covariate shift) : `P(X)` change, `P(Y|X)` stable. Nouveau
  capteur, autre saison, voile nuageux. « Une forêt reste une forêt » — la règle
  n'a pas bougé, l'entrée si. Le modèle se dégrade en extrapolant hors domaine.
- **Concept drift** : `P(Y|X)` change. Les mêmes pixels changent d'étiquette
  (parcelle agricole urbanisée, nomenclature révisée). Même un modèle parfaitement
  calibré sur l'ancien monde a tort.
- **Label shift** : `P(Y)` change, `P(X|Y)` stable. Le mix de classes entrant
  bascule. Se traite par recalibration des priors, pas par réentraînement.

**Le point fondamental, et c'est LA raison d'être du sprint** : en production on
observe `X` (les images) et `Ŷ` (les prédictions). On n'observe **pas** `Y`. Donc :

| grandeur | observable en prod | détecte |
|---|---|---|
| `P(X)` | oui, immédiatement | data drift |
| `P(Ŷ)`, confiance, entropie | oui | proxy de dérive |
| `Y`, accuracy | **non** (ou tard, ou cher) | — |

→ **Le data drift est détectable sans labels ; le concept drift ne l'est pas.**
Tout le monitoring ML sérieux surveille donc un proxy observable en PARIANT qu'il
corrèle avec une dégradation invisible. Ce pari n'est presque jamais vérifié : le
seuil est copié d'un blog et le dashboard est cru parce qu'il est vert.

Ici il est vérifiable, parce que la dérive est simulée et les labels connus.

### Concept n°2 — Les tests statistiques, et pourquoi le volume les casse

- **KS** : écart vertical max entre les CDF empiriques. Non paramétrique, univarié,
  sensible au centre plus qu'aux queues. Sort une p-value.
- **PSI** : divergence KL symétrisée sur des buckets (10 déciles). Pas de p-value,
  seuils conventionnels 0.1 / 0.25. Avantage décisif : **la taille d'échantillon
  n'entre pas dans la formule**.
- **Wasserstein** : coût de transport minimal ; en 1D, l'AIRE entre les CDF (KS en
  prend le max). Sensible à l'AMPLEUR du déplacement, pas seulement à sa
  détectabilité. À normaliser par l'écart-type de la référence pour comparer entre
  features.

**Pourquoi les faux positifs explosent au volume** — la phrase à savoir sortir :

> Un test d'hypothèse ne mesure pas « à quel point c'est différent », il mesure
> « à quel point je suis sûr que ce n'est pas identique ».

H₀ (« distributions identiques ») est **toujours fausse** entre deux échantillons
réels. Ce n'est donc qu'une question de puissance : `D_crit ≈ c(α)·√(2/n)`.
À n = 1 000 → 0.061 (il faut un vrai décalage). À n = 1 000 000 → 0.0019 : un écart
de 0.2 % entre CDF déclenche l'alarme. Significatif, sans conséquence, et l'alerte
hurle tous les jours jusqu'à ce qu'on la coupe — fatigue d'alerte.

Parades appliquées : effect size plutôt que p-value (`monitor.stattest: wasserstein`),
fenêtre **plafonnée** (`current_window: 2000`), **plancher** de refus
(`min_current_rows: 200`), persistance sur 3 fenêtres, seuil calibré empiriquement
par l'expérience — pas le 0.25 conventionnel.

### Concept n°3 — CI/CD vs CT

| | déclencheur | entrée | sortie | décideur |
|---|---|---|---|---|
| CI | commit | code | tests + image | tests (déterministe) |
| CD | merge/tag | image + config | déploiement | déterministe + approbation |
| **CT** | **la donnée** (dérive, calendrier) | données + code figé | un **candidat** | **le gate** |

> **CI/CD réagit à un changement de CODE. CT réagit à un changement du MONDE.**

La propriété qui n'existe dans aucun autre logiciel : **la valeur d'un artefact ML
décroît avec le temps même à code constant**. Et : CI/CD produit un artefact
DÉPLOYABLE, CT produit un artefact **CANDIDAT**.

**Pourquoi le réentraînement automatique sans gate est dangereux** (5 arguments) :

1. **Le drift peut être une PANNE, pas une évolution.** Capteur HS → images
   corrompues → on réentraîne dessus → on apprend au modèle que la corruption est
   normale. On blanchit la panne dans les poids, et le détecteur, recalibré sur les
   données pourries, ne signale plus rien. Le monitoring s'auto-neutralise.
2. **L'entraînement est stochastique.** Preuve empirique dans ce repo : 5
   expériences Sprint 2, 0 amélioration, **3 refus du gate**. Un CT sans gate aurait
   déployé les trois.
3. **Boucle de rétroaction dégénérative** : chaque cycle réentraîne sur les données
   du précédent. `gate/frozen_val.json` commité est l'ancre qui casse la boucle.
4. **L'accuracy globale masque les régressions par classe** (`max_class_recall_drop`).
5. **Sans gate, pas de rollback propre** — l'alias `@champion` EST la version prod.

Et le point que presque personne ne dit : **un gate qui refuse ne referme pas
l'incident.** Ça veut dire « la dérive est réelle et le réentraînement ne suffit
pas » → escalade humaine. Le CT automatise le TRAVAIL, pas la DÉCISION.

### L'expérience centrale — la courbe dérive ↔ accuracy

`src/drift_experiment.py` : 4 perturbations × 8 intensités × 500 images du jeu figé.
Baseline intensité 0 : **98.60 %** (champion gaté à 98.10 % sur les 4050 → cohérent
au bruit d'échantillonnage près, donc baseline auditable).

| perturbation | détection | effondrement | avance | verdict |
|---|---|---|---|---|
| voile nuageux | 0.1 | 0.2 | **+0.1** | alerte précoce |
| saisonnier | 0.2 | 0.6 | **+0.4** | alerte précoce |
| décalage de bandes | 0.2 | 0.6 | **+0.4** | alerte précoce |
| **flou** | 0.6 | 0.3 | **−0.3** | **TARDIF — angle mort** |
| tout combiné | 0.2 | 0.1 | **−0.1** | tardif (hérite du flou) |

**Trois résultats, dont deux inconfortables :**

1. La dérive **radiométrique** est vue tôt. Vraie marge de manœuvre.

2. **Le flou est un angle mort STRUCTUREL.** L'accuracy tombe 97 → 81 → 61 %
   (0.2 → 0.3 → 0.4) alors que la part en dérive reste à **0.00** jusqu'à 0.3 et
   n'atteint que **0.42** à 0.4, sous le seuil de 0.5 ; l'alerte tombe à 0.6,
   accuracy déjà à 27 %. 11 features sur 12 décrivent la couleur ; la seule feature
   de texture, `sharpness`, n'est jamais la plus déplacée dans
   `experiments/drift_curve/results.json`. Ce qui finit par déclencher, ce sont des
   statistiques de couleur du second ordre (`saturation`, puis `std_b`), ce qui est
   cohérent avec un lissage qui écrase la dispersion des canaux (non mesuré par
   feature : rapports Evidently non versionnés). Abaisser le seuil de part à n'importe
   quelle valeur donnerait au mieux une alerte à 0.4, encore après le début de la
   chute. **Aucune valeur du seuil ne corrige ça** — c'est une conséquence de
   conception, maintenant mesurée et non plus soupçonnée.

3. **Le modèle devient confiant ET faux.** Sous voile nuageux total : accuracy
   0.098 (= hasard sur 10 classes), entropie moyenne **0.008** contre 0.021 au
   repos, 100 % des prédictions sur une seule classe. L'entropie monte dans la zone
   de confusion puis **redescend SOUS la baseline**.
   → Conséquence appliquée : **l'entropie n'est PAS un déclencheur CT ici.** Une
   alerte « confiance basse = problème » serait verte au pire moment.

### Erreurs commises (les plus instructives)

**1. J'ai failli publier un faux angle mort.** Premier balayage : 200 images, toutes
les fenêtres revenaient `insufficient_data` (plancher à 200) — et mon agrégation
traitait « pas de verdict » comme « pas de dérive ». J'allais conclure « le
détecteur ne voit rien » alors qu'il n'avait jamais été interrogé. C'est
exactement le mode de défaillance que le sprint combat : **un silence pris pour une
bonne nouvelle.** Corrections : statut à trois valeurs `ok / drift /
insufficient_data`, refus explicite au démarrage si `sample < min_current_rows`, et
un test qui épingle qu'une fenêtre non concluante ne compte jamais comme « sans
dérive ».

**2. Deux perturbations opposées se compensent.** Le voile éclaircit, l'hiver
assombrit : le composite est PLUS PROCHE de l'original en distance pixel moyenne
que le nuage seul. Mon test « le composite domine ses parties » échouait à raison.
Leçon : **une distance agrégée unique peut DIMINUER pendant que la situation
empire.** La comparaison par feature atténue, n'élimine pas.

**3. Le mauvais Postgres.** Un PostgreSQL Windows natif occupait 5432 ; le conteneur
publiait le port mais toute connexion hôte atteignait l'AUTRE base. L'échec
d'authentification revenait en codepage ANSI → `UnicodeDecodeError`, une exception
qui nomme un problème d'encodage et ne dit rien de la cause. Corrigé par un port
inhabituel + un diagnostic explicite dans `_describe_error`.

### Test d'acceptation de bout en bout (validé)

520 requêtes réelles via HTTP contre l'API **conteneurisée**
(`drift_traffic.py`), 0 échec. Les deux envois utilisent le **même échantillonnage
équilibré** sur les 10 classes : seule la perturbation diffère.

| source | n | p95 serveur | confiance | entropie | classes prédites | classe majoritaire |
|---|---|---|---|---|---|---|
| `v2:baseline` | 260 | 851 ms | 0.987 | 0.019 | **10/10** | 0.12 |
| `v2:cloud:0.6` | 260 | 1098 ms | 0.861 | 0.156 | **7/10** | `SeaLake` **0.63** |

- Rapport baseline : **pas de dérive**, part **0.00**.
- Rapport nuage 0.6 : **DÉRIVE**, part 0.92, top `mean_b` (1.44).
- Monitor : streak 1/3 → 2/3 → 3/3 puis dry-run de dispatch, les DEUX détecteurs
  allumés. Repasser sur la source baseline **remet le compteur à 0**.

L'entrée étant échantillonnée à l'identique dans les deux lignes, l'effondrement de
classes est imputable au MODÈLE et non au trafic — c'est la seule condition qui rend
ce chiffre interprétable (voir l'erreur n°4).

Note d'honnêteté sur les latences : p95 client ~1000-1300 ms contre 851 ms serveur —
l'écart est l'encodage PNG, HTTP et le client Python. Les deux valeurs serveur
dépassent largement le budget de 400 ms de `params.yaml:nonreg`, mais **elles ne
mesurent pas la même chose** : ce budget couvre un forward pass en processus, ici on
mesure décodage + features + logging sous un client unique qui sature, sur un poste
de dev. Ce n'est pas une régression ; ce n'est pas non plus un chiffre à citer comme
latence de production.

**4. Un signal de monitoring validé contre une baseline confondue.** Mon
générateur de trafic prenait un pas régulier sur le dataset (pour couvrir les 10
classes), mais `--offset` **découpait la queue** de la liste échantillonnée au lieu
d'en décaler la phase. Résultat : `--offset 300 --count 260` n'envoyait que les
indices 14400-26880, soit les cinq dernières classes. Le premier test d'acceptation
annonçait donc un « effondrement de classes à 0.68 » qui mesurait en partie **quelles
tuiles j'avais envoyées**, pas le comportement du modèle. Repéré par un smoke test
qui a renvoyé 19 `SeaLake` sur 20.

Après correction (décalage de phase, spectre complet des classes dans les deux
envois), le résultat est plus FORT et surtout propre : à entrée équilibrée
identique, la baseline prédit 10 classes sur 10 avec une majorité à 0.12, et le
nuage 0.6 n'en prédit plus que 7 avec `SeaLake` à 0.63.

La leçon dépasse le bug : **un détecteur ne vaut que par la comparabilité de sa
baseline.** Un chiffre de monitoring qui bouge peut toujours être expliqué par un
changement de l'échantillon plutôt que par un changement du système — c'est le même
piège que le label shift confondu avec le concept drift, rencontré ici dans mon
propre outillage. Toujours se demander : « qu'est-ce qui a changé D'AUTRE entre les
deux mesures ? »

### Dette connue à la clôture

- Angle mort du flou **non corrigé** : il faudrait des features de texture/fréquence
  ou une dérive sur embeddings.
- `retrain.yml` exige un runner self-hosted (MinIO et MLflow sont locaux) : la
  boucle CT n'est pas démontrable sur un runner GitHub public.
- `/reload` toujours manuel.
- L'axe d'intensité est arbitraire : les avances ne se comparent pas ENTRE
  perturbations, seulement en signe et en ordre à l'intérieur d'une perturbation.

### Clôture — la CI vérifiée, pas seulement verte

Trois runs verts dès la mise en ligne, les deux jobs, smoke test du conteneur inclus.
Mais « vert » n'est pas une information tant qu'on ne sait pas **ce qui a été
exécuté**. La vérification, faite en prédisant d'abord le résultat puis en le
comparant :

| contexte | résultat | durée |
|---|---|---|
| local, stack allumée | 85 passed, **0 skipped** | ~6 min |
| local, stack éteinte | 80 passed, **5 skipped** | 1 min 17 |
| runner GitHub | 80 passed, **5 skipped** | 15,8 s |

Même commande, même code, trois résultats — et les trois sont corrects. C'est la
stratégie à deux niveaux qui se voit : les tests unitaires tournent partout et sont
bloquants ; les tests de non-régression du champion servi exigent une infrastructure
et **se désactivent proprement au lieu d'échouer**. Une CI rouge par défaut faute
d'infra est supprimée en trois semaines ; une CI qui prétendrait les avoir passés
serait un mensonge.

Détail plus intéressant que le total : la RAISON des skips diffère. En local (données
présentes, MLflow éteint) les 5 skips invoquent MLflow. Sur le runner, 2 invoquent
MLflow et 3 invoquent l'absence de données DVC. Les deux garde-fous sont
**indépendants** et nomment précisément ce qui manque — un lecteur du log sait quoi
faire pour les réactiver. C'est la différence entre un skip exploitable et un skip
qui masque.

D'où le `-rs`, non négociable : sans lui les skips sont invisibles, et une suite qui
saute son filet de sécurité est indistinguable d'une suite qui l'exécute.

### Deux incidents de fin de sprint, gardés parce qu'ils sont instructifs

**1. Un workflow qui aurait échoué après avoir réussi.** `retrain.yml` appelait
`gh pr create` sans condition, sur un runner self-hosted — c'est-à-dire ma propre
machine, où `gh` n'est pas installé. Un réentraînement de ~6 h ayant passé le gate,
déplacé l'alias `@champion` et rechargé l'API se serait donc terminé en ROUGE parce
qu'un outil de confort manquait. Un run rouge qui a réussi est le pire signal qu'un
pipeline puisse émettre : il apprend aux gens à ignorer la couleur. Corrigé — la
branche est toujours poussée, la PR n'est ouverte que si `gh` existe, l'URL de
comparaison est écrite dans le résumé dans tous les cas. À noter : `drift_monitor.py`
n'avait pas ce défaut, il avait déjà un repli REST. Le workflow, lui, n'en avait pas.

**2. Un jeton OAuth imprimé en clair.** Un en-tête `Authorization` mal échappé dans
un `curl` a fait interpréter le jeton comme un nom d'hôte, affiché intégralement dans
le message d'erreur. Règle appliquée sans discussion : **un secret qui apparaît dans
un log est compromis**, indépendamment de l'évaluation du risque. Révocation
immédiate via Settings → Applications → Authorized OAuth Apps. On ne peut pas
dé-imprimer un secret, et « probablement sans conséquence » n'est pas un état de
sécurité mais un pari. Coût réel de la révocation : une reconnexion.
