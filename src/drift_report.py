"""Data-drift report: the committed reference vs a window of production traffic.

Reads: monitoring/reference.json (frozen train statistics, built by
drift_reference.py) and the monitoring.predictions table written by the API.
Produces: an Evidently HTML report for humans, and a compact JSON summary that
the CT trigger can act on without parsing HTML.

Why the statistical test is PINNED in params.yaml
--------------------------------------------------
Evidently picks a test automatically based on sample size and column type. That
default is sensible, but "sensible and invisible" is the wrong property for the
number that decides whether a retraining pipeline fires. So monitor.stattest is
explicit, and it is `wasserstein` — an EFFECT SIZE, normalized by the reference
standard deviation, not a p-value.

The reason is the failure mode that kills most drift dashboards: a p-value test
answers "am I sure the distributions differ?", and with a large enough window the
answer is always yes, because two real samples are never exactly co-distributed.
At n = 500k a KS test rejects on a 0.2% difference between CDFs — statistically
significant, practically irrelevant, and the alert fires every day until someone
mutes it. Wasserstein answers "by how much", in units of the reference's own
spread, and that number does not inflate with volume.

The window is capped (monitor.current_window) for the same reason, and a floor
(monitor.min_current_rows) makes the report REFUSE to conclude on thin traffic
rather than emit a confident verdict built on eighty rows.

What this report can and cannot say
-----------------------------------
It compares INPUT distributions. It cannot measure accuracy, because production
has no labels — that is the whole premise of the sprint. A drifted verdict means
"the incoming data no longer resembles what the model learned from", which is a
reason to investigate and possibly retrain; it is never, by itself, proof that
predictions got worse. The experiment in experiments/ is what calibrates how
much one implies the other.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from drift_reference import load_reference
from image_features import FEATURE_NAMES
from prediction_log import DB_URI, TABLE
from utils import REPO_ROOT, load_params


def rows_to_frame(rows: List[Dict]) -> pd.DataFrame:
    """Feature dicts -> a DataFrame with exactly FEATURE_NAMES columns, in order.

    Reindexing on FEATURE_NAMES rather than trusting dict order guarantees the
    reference and the current frame are aligned column-for-column: a silent
    column mismatch would compare brightness against sharpness and report
    spectacular, meaningless drift.
    """
    frame = pd.DataFrame(rows)
    missing = [c for c in FEATURE_NAMES if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return frame[FEATURE_NAMES].astype(float)


# --- reading the production window --------------------------------------------

def fetch_current(params: dict, source: Optional[str] = None,
                  since_minutes: Optional[int] = None,
                  limit: Optional[int] = None,
                  db_uri: Optional[str] = None) -> Tuple[List[Dict], Dict]:
    """Pull the most recent prediction rows from the log.

    Ordered by ts DESC and capped: this is the sampling step that keeps a
    statistical test from being decided by volume. `source` filters on the
    X-TerraOps-Source tag, which is what keeps simulated traffic from being
    mixed into a report about real traffic (or the reverse).
    """
    import psycopg2

    monitor_cfg = params["monitor"]
    limit = limit or monitor_cfg["current_window"]
    columns = FEATURE_NAMES + ["predicted_class", "confidence", "entropy",
                               "latency_ms", "model_version", "ts"]

    where, args = [], []
    if source:
        where.append("source = %s")
        args.append(source)
    if since_minutes:
        where.append("ts >= now() - make_interval(mins => %s)")
        args.append(since_minutes)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    query = (f"SELECT {', '.join(columns)} FROM {TABLE} {clause} "
             f"ORDER BY ts DESC LIMIT %s")
    args.append(limit)

    conn = psycopg2.connect(db_uri or DB_URI, connect_timeout=5)
    try:
        with conn.cursor() as cur:
            cur.execute(query, args)
            fetched = cur.fetchall()
    finally:
        conn.close()

    rows = [dict(zip(columns, record, strict=True)) for record in fetched]
    meta = {
        "source_filter": source,
        "since_minutes": since_minutes,
        "limit": limit,
        "n_fetched": len(rows),
        "newest_ts": str(rows[0]["ts"]) if rows else None,
        "oldest_ts": str(rows[-1]["ts"]) if rows else None,
    }
    return rows, meta


# --- the drift computation -----------------------------------------------------

def compute_drift(reference_rows: List[Dict], current_rows: List[Dict],
                  params: dict) -> Tuple[Dict, Optional[object]]:
    """Run Evidently and flatten its result into a decision-ready summary.

    Returns (summary, snapshot). The snapshot carries the HTML; the summary is
    what the CT trigger reads — a machine consuming a rendered report would be
    parsing a presentation format for a control decision.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    monitor_cfg = params["monitor"]
    method = monitor_cfg["stattest"]
    threshold = float(monitor_cfg["stattest_threshold"])
    share_threshold = float(monitor_cfg["drift_share_threshold"])

    if len(current_rows) < int(monitor_cfg["min_current_rows"]):
        # Refusing to conclude is a RESULT, not an error. A verdict computed on
        # a handful of rows would be noise dressed as a signal, and the CT loop
        # must be able to tell "no drift" from "not enough evidence".
        return {
            "status": "insufficient_data",
            "n_reference": len(reference_rows),
            "n_current": len(current_rows),
            "min_current_rows": int(monitor_cfg["min_current_rows"]),
            "dataset_drift": False,
            "message": (f"only {len(current_rows)} rows in the window; "
                        f"{monitor_cfg['min_current_rows']} required"),
        }, None

    reference = rows_to_frame(reference_rows)
    current = rows_to_frame(current_rows)

    definition = DataDefinition(numerical_columns=list(FEATURE_NAMES))
    report = Report(metrics=[DataDriftPreset(
        num_method=method,
        num_threshold=threshold,
        drift_share=share_threshold,
    )])
    snapshot = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )

    features: Dict[str, Dict[str, float]] = {}
    drifted_count = drifted_share = None
    for metric in snapshot.dict().get("metrics", []):
        config = metric.get("config", {})
        kind = config.get("type", "")
        if kind.endswith("ValueDrift"):
            column = config["column"]
            distance = float(metric["value"])
            features[column] = {
                "distance": distance,
                # Recomputed from the same threshold Evidently was configured
                # with, so the flag and the number can never disagree.
                "drifted": distance > threshold,
            }
        elif kind.endswith("DriftedColumnsCount"):
            drifted_count = int(metric["value"]["count"])
            drifted_share = float(metric["value"]["share"])

    ranked = sorted(features.items(), key=lambda kv: kv[1]["distance"], reverse=True)

    return {
        "status": "ok",
        "n_reference": len(reference_rows),
        "n_current": len(current_rows),
        "stattest": method,
        "stattest_threshold": threshold,
        "drift_share_threshold": share_threshold,
        "drifted_count": drifted_count,
        "drifted_share": drifted_share,
        # THE decision bit the CT loop reads. Note it is a SHARE of features, not
        # a single feature: one moving column is weather, half of them is a
        # regime change. That choice is what keeps the trigger from firing on a
        # single noisy dimension.
        "dataset_drift": bool(drifted_share is not None
                              and drifted_share >= share_threshold),
        "features": features,
        "top_drifted": [{"feature": name, **values} for name, values in ranked[:5]],
    }, snapshot


def summarize_predictions(current_rows: List[Dict]) -> Dict:
    """Model-side signals over the same window — informational, never decisive.

    The predicted-class mix is NOT a label distribution: comparing it to the
    training class distribution shows how the model's OUTPUT moved, which drifts
    both when the input mix changes and when the model degrades. It is a
    conversation starter, not evidence, and it is labelled as such in the JSON.
    """
    if not current_rows:
        return {}
    counts: Dict[str, int] = {}
    for row in current_rows:
        name = row.get("predicted_class")
        if name:
            counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values()) or 1

    def mean_of(key):
        values = [row[key] for row in current_rows if row.get(key) is not None]
        return float(sum(values) / len(values)) if values else None

    return {
        "note": "predicted classes, NOT ground-truth labels — production has none",
        "predicted_class_share": {k: v / total for k, v in sorted(counts.items())},
        "mean_confidence": mean_of("confidence"),
        "mean_entropy": mean_of("entropy"),
        "mean_latency_ms": mean_of("latency_ms"),
        "model_versions": sorted({str(row.get("model_version"))
                                  for row in current_rows}),
    }


# --- CLI -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a window of the prediction log against the frozen "
                    "training reference and emit an Evidently report.")
    parser.add_argument("--source", default=None,
                        help="filter on the X-TerraOps-Source tag (e.g. sim:cloud:0.4)")
    parser.add_argument("--since-minutes", type=int, default=None)
    parser.add_argument("--window", type=int, default=None,
                        help="override monitor.current_window")
    parser.add_argument("--db-uri", default=os.environ.get("TERRAOPS_DB_URI"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--name", default=None, help="report file stem")
    parser.add_argument("--exit-code", action="store_true",
                        help="exit 2 when drift is detected (for the CT trigger); "
                             "exit 3 when there is not enough data to conclude")
    args = parser.parse_args()

    params = load_params()
    reference = load_reference(params)
    current_rows, window_meta = fetch_current(
        params, source=args.source, since_minutes=args.since_minutes,
        limit=args.window, db_uri=args.db_uri)

    summary, snapshot = compute_drift(reference["rows"], current_rows, params)
    summary["window"] = window_meta
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["reference"] = {
        "created_at": reference.get("created_at"),
        "git_commit": reference.get("git_commit"),
        "dvc_data_hash": reference.get("dvc_data_hash"),
        "split": reference.get("split"),
        "n_samples": reference.get("n_samples"),
    }
    summary["predictions"] = summarize_predictions(current_rows)

    out_dir = args.out_dir or (REPO_ROOT / params["monitor"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"drift_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {json_path}")

    if snapshot is not None:
        html_path = out_dir / f"{stem}.html"
        snapshot.save_html(str(html_path))
        print(f"report  -> {html_path}")

    if summary["status"] == "insufficient_data":
        print(f"INCONCLUSIVE: {summary['message']}")
        raise SystemExit(3 if args.exit_code else 0)

    verdict = "DRIFT" if summary["dataset_drift"] else "no drift"
    print(f"{verdict}: {summary['drifted_count']}/{len(FEATURE_NAMES)} features "
          f"above {summary['stattest_threshold']} "
          f"({summary['stattest']}), share={summary['drifted_share']:.2f} "
          f"vs threshold {summary['drift_share_threshold']}")
    for item in summary["top_drifted"]:
        flag = "*" if item["drifted"] else " "
        print(f"  {flag} {item['feature']:<12} {item['distance']:.4f}")

    if args.exit_code and summary["dataset_drift"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
