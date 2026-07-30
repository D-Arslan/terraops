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

### Questions type recruteur (Sprint 1)

1. **« Pourquoi DVC et pas Git-LFS ou juste les artifacts MLflow ? »**
   Git-LFS versionne mais n'orchestre pas de pipeline ; MLflow trace les expériences mais
   ne versionne pas l'input data ni le DAG. DVC fait pointeurs + remote + DAG reproductible.
2. **« Comment DVC sait qu'une étape doit être relancée ? »**
   Il hashe deps + params + outs, compare au `dvc.lock` ; un hash différent → étape sale →
   relance, et propagation à tout l'aval qui en dépend.
3. **« Où sont stockés le dataset et les secrets, et pourquoi cette séparation ? »**
   Octets → MinIO (remote S3) via `dvc push` ; pointeurs + métriques → Git ; identifiants →
   `.dvc/config.local` gitignoré. But : reproductibilité + aucun secret dans l'historique Git.

---

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

### Questions type recruteur (Sprint 2)

1. **« Tracking vs Registry ? »** Cahier de labo immuable vs catalogue de production
   avec versions et alias — on ne « release » pas chaque commit.
2. **« Comment garantissez-vous la traçabilité modèle → données ? »** Tags
   `git_commit` + `dvc_data_hash` posés avant l'entraînement ; chaîne registry → run →
   checkout + pull. Testé en conditions réelles.
3. **« Un modèle gagne de 0.1 pt, vous le promouvez ? »** Non : bande de bruit, jeu
   figé, non-régression par classe, coût d'inférence visible. Décision scriptée.
4. **« Pourquoi le champion a-t-il eu plus d'epochs que les challengers ? »** Le gate
   compare des MODÈLES finis, pas des protocoles. Un challenger peut prendre le même
   budget ; seul le résultat sur le jeu figé compte.
5. **« Une fois où le process a contredit votre intuition ? »** Duel final 97.65 vs
   98.10 : « presque pareil » à l'œil, refus sur la marge au gate. Et une hypothèse
   scheduler réfutée par l'historique du LR loggé — on ne devine pas, on logge.
6. **« Quelle dette assumez-vous ? »** EuroSAT 64×64 natif upscalé à 224 (~12× de
   calcul). Changement reporté sciemment : modifier les règles en pleine campagne
   détruit la comparabilité (et le duel du gate).
