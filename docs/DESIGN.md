# TerraOps — design decisions and limits

The README keeps five bullets per topic. This file holds the reasoning. Every
number here is traceable to a versioned file; where it is not, it says so.

## 1. Design decisions, and why

**One preprocessing module, imported by training, the gate and the API.**
`src/preprocessing.py` owns the whole chain from raw bytes to tensor, decoding
included, because channel order, alpha, and EXIF rotation are the classic
silent-skew sources and they happen *before* the transforms. Train/serving skew
is impossible by construction rather than discouraged by documentation.
`src/image_features.py` plays the same role for monitoring: `FEATURE_NAMES` is
the single source of truth for the Postgres columns and the Evidently mapping,
so the API, the reference builder and the report can never measure the gap
between two extractors instead of the gap between two distributions.

**The API loads the model by ALIAS, never from a path.**
`models:/terraops-eurosat@champion`. Changing what production serves is a
governance action (move the alias through the gate), not a code deploy.
`POST /reload` re-resolves the alias in process, so a promotion reaches
production without a restart. The API boots without a model and answers 503 on
`/predict` and `/model-info` until one loads; `/health` is a liveness probe and
stays 200.

**A promotion gate that can say no.**
Absolute accuracy floor, a minimum delta above seed noise, **and a per-class
recall guard**: a challenger that gains overall while losing more than 3 points
of recall on one class is refused. The three rules are a pure function
(`gate_reasons` in `src/promote.py`) covered by `tests/test_promote.py`.
Over the sprint-2 campaign (5 experiments on a 6-epoch CPU budget: learning
rate 0.0001 and 0.01, augmentation off, backbone frozen, MobileNetV3-small),
none beat the champion. The registry keeps
**two real refusals** (v3 at 96.52%, v4 at 97.65% against a 98.10% champion)
plus v2, an untrained model at 11.51% that documents the floor rule rather than
a genuine attempt. Refused versions stay in the registry with
`gate_result: refused`: a refusal is a recorded decision, not a deletion.

**A frozen validation set, committed.**
`gate/frozen_val.json` is the anchor every version is measured against. Its
input-side twin is `monitoring/reference.json`, the drift reference, built from
the **train** split **without augmentation**: augmentation is a regulariser,
not a description of the world; including it would inflate the reference
variance and blind the detector to exactly the radiometric drift it exists to
catch. Neither anchor is ever replaced by a sliding window: slow drift would
become invisible.

**Effect sizes, not p-values.**
The drift test is pinned to normalized Wasserstein distance in `params.yaml`.
A p-value test answers "am I sure they differ?", and at large volume the answer
is always yes: with the Kolmogorov–Smirnov critical value scaling as
`sqrt(2/n)`, a 0.2% CDF difference is significant at n = 500 000 (a textbook
consequence, not a measurement made here). Such a test fires daily until
someone mutes it. The window is capped for the same reason, and a floor makes
the report **refuse to conclude** on thin traffic rather than emit a confident
verdict.

**Three-valued drift status.** `ok / drift / insufficient_data`. "No data" must
never read as "no drift"; that is the silent-failure shape this whole stack
exists to prevent. The CT trigger treats an inconclusive window as evidence in
neither direction: it neither confirms nor resets the streak.

**Non-blocking prediction logging.** Every prediction is written to Postgres
through a bounded queue drained by a background thread. If the queue saturates,
rows are **dropped and counted**, and the counter is exposed on
`/monitoring/status` and `/metrics`. A monitoring pipeline must never be able
to take down the service it observes; losing rows silently would be worse,
because a drift report computed over a lossy window is wrong without saying so.

**Prometheus and Postgres do different jobs.** Prometheus answers "is the
service healthy now?" (pre-aggregated, seconds, days of retention). Postgres
answers "what exactly did production see?" (row-level, joinable, months). Doing
drift detection in Prometheus would mean per-image feature values as labels:
unbounded cardinality, the canonical way to kill a Prometheus server.

**A CT loop that mostly refuses.** Persistence (3 consecutive windows), a
12-hour cooldown, inconclusive windows that neither confirm nor reset, and an
explicit `--dispatch` flag (dry run by default). Drift caused by a broken
sensor does not go away when you retrain, so without a cooldown the loop would
retrain forever on increasingly corrupted data. The decision logic is pure and
covered by `tests/test_drift_monitor.py`.

**Simulated traffic is always tagged.** Every request carries an
`X-TerraOps-Source` header; the drift report and the trigger filter on it.
Mixing simulated and real traffic would corrupt the very window the CT loop
reacts to.

## 2. What was measured, and what it does and does not establish

Source: `experiments/drift_curve/results.json` (500 images of the frozen set,
4 perturbations × 8 intensities, champion v1) and `experiments/acceptance/`.

**Radiometric drift is caught early.** For the cloud veil, the seasonal shift
and the band shift, the detector fires at intensity 0.1–0.2 while accuracy is
still between 95% and 98% at 0.4. The lead is real but its size is not
comparable across perturbations (see the intensity-axis caveat below).

**Blur is a structural blind spot.** Accuracy falls 97% → 81% → 61% (intensity
0.2 → 0.3 → 0.4) while the drifted feature share stays at 0.00 up to 0.3 and
reaches only 0.42 at 0.4, under the 0.5 threshold. The detector fires at 0.6,
when accuracy is already at 27%. Eleven of the twelve monitored features
describe colour; the only texture feature, `sharpness`, is never the
top-drifting one. What eventually trips the alert is second-order colour
statistics (`saturation`, then `std_b`), consistent with smoothing collapsing
channel spread, though this is not measured per feature. Lowering the share
threshold to any value would at best move the alert to 0.4, still after the
drop began. Fixing it means texture/frequency features or embedding-based
drift: designed, not built.

**The model becomes confident and wrong.** Under a full cloud veil, accuracy
reaches 9.8% (chance level over ten classes), mean entropy 0.008 against 0.021
at rest, and 100% of predictions land on one class. Entropy rises through the
confusion zone (0.145 at 0.6) and then falls back below baseline. Consequence:
mean entropy is **not** a CT trigger. The class-collapse detector is kept as a
catastrophe backstop, not an early warning: under blur it only reaches its
threshold at 0.6, when accuracy is already at 27%.

**Two real drifts can cancel on a monitored feature.** The cloud veil brightens
while the winter shift darkens, so their composite is closer to the reference
in aggregate pixel distance than the cloud alone; the band shift's asymmetric
gains raise saturation, cancelling the drop from cloud and winter. Feature-level
comparison mitigates this; it does not eliminate it.

### Caveats on the measurement

- **The drift is simulated. There is no production stream.** The perturbations
  are parametric functions with amplitudes chosen by the author
  (`params.yaml: drift_sim`). A detector calibrated on that family is calibrated
  on that family. The curves are evidence, not proof.
- **Only covariate shift, never concept drift.** The perturbations are
  label-preserving by construction: `P(Y|X)` is untouched. Nothing here
  demonstrates detection of concept drift, which is not detectable without
  labels at all.
- **The intensity axis is arbitrary.** `0.4` is 40% of an amplitude the author
  picked. Only the sign and the ordering of the two events within one
  perturbation are meaningful; leads are not comparable across perturbations.
- **The lead values are grid bounds, not crossing points.** Measured on
  `[0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]`, so a lead of +0.1 is compatible
  with a true lead near zero.
- **±2 points of sampling noise** on a 500-image sweep; "collapse" is defined
  as a 5-point drop for that reason.
- **EuroSAT is Eurocentric and small.** 27 000 Sentinel-2 tiles over Europe,
  64×64 pixels, ten coarse classes, cloud-free by curation. The 97.8% is a
  number about this benchmark.

### End-to-end acceptance test (2026-08-08)

520 real HTTP requests through the containerized API, 0 failures, same balanced
sampling (26 tiles × 10 classes) for both sources. Figures re-derived from the
prediction log and versioned in `experiments/acceptance/summary.json`; the two
Evidently summaries sit alongside.

| source | n | server p95 | mean confidence | classes predicted | majority class | drift |
|---|---|---|---|---|---|---|
| `v2:baseline` | 260 | 851 ms | 0.987 | 10/10 | 0.12 | no (share 0.00) |
| `v2:cloud:0.6` | 260 | 1098 ms | 0.861 | 7/10 | `SeaLake` 0.63 | **yes** (share 0.92, top `mean_b` 1.44) |

The CT monitor then went `streak 1/3 → 2/3 → 3/3 → would dispatch`, and
switching back to the baseline source reset the streak. Latency note: both p95
values are far above the 400 ms in-process budget of `params.yaml: nonreg`, but
they measure decode + features + logging under a saturating single client on a
laptop, not a forward pass. Client-side p95 (~1000–1300 ms) was read live and
is not reproducible.

## 3. Known debt

- **Blur blind spot** not fixed (see above).
- **`retrain.yml` cannot run on a GitHub-hosted runner**: it needs the local
  MinIO and MLflow. It targets a self-hosted runner and no green badge is
  claimed for it. Its Postgres URI points at the host-published port 55433.
- **`/reload` is manual**: no TTL, no webhook. A promoted champion nobody
  reloads keeps the old version in production; the version is visible on
  `/model-info` and as a Prometheus label, but nothing enforces it.
- **64×64 tiles upscaled to 224**: roughly 12× the compute for no extra
  information, inherited from the ImageNet backbone. Kept so comparisons across
  the campaign stay valid.
- **API image 2.53 GB** (torch CPU + full MLflow client). `mlflow-skinny` is
  the obvious next cut.
- **A fresh clone cannot `dvc pull`**: the DVC remote is a local MinIO. The
  weights of the champion are reproducible (`dvc repro`, ~6 h CPU) but not
  downloadable from the repository.
- **Dev credentials** (`minioadmin`, `mlflow`) are in `docker-compose.yml` in
  clear, on purpose: this is a laptop stack, and `.env.example` says so.

## 4. Provenance of the champion

The served champion is registry version 1, alias `@champion`, logged from
commit `328d45a` with `git_dirty: false`. Its weights are the file DVC tracks
at `models/best_model.pth` (md5 `fc6bea98…`); on 2026-09-13 the registry
artifact and the DVC-tracked file were compared tensor for tensor and found
identical, and the `evaluate` stage was re-run under the current code: 97.80%,
89 misclassified out of 4050, byte-identical confusion matrix. The `train`
stage was not re-run at that date (the lock was re-recorded on the restored
weights); a full `dvc repro` reproduces it in about 6 CPU-hours.
