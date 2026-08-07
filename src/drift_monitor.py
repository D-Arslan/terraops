"""The CT trigger: from a drift verdict to a retraining workflow.

This is the file where monitoring stops describing and starts acting, so it is
also the file where an error costs the most. Its job is mostly to REFUSE:

  * persistence  — a threshold crossed on one window is noise. It must hold for
                   monitor.trigger.consecutive_windows reports in a row. Same
                   reasoning as the `for:` clause on every Prometheus alert.
  * cooldown     — at most one dispatch per monitor.trigger.cooldown_minutes.
                   Drift caused by a broken sensor does not go away when you
                   retrain, so without a cooldown the loop would retrain
                   forever, each time on more corrupted data.
  * inconclusive — a window below monitor.min_current_rows produces no verdict.
                   It resets nothing and triggers nothing, and it is reported
                   distinctly, because "no data" must never read as "no drift".
  * explicit     — dispatching requires --dispatch. The default is a dry run.
                   A monitoring script that fires a pipeline as a side effect of
                   being run is a script nobody dares to run.

Two detectors, OR-ed
--------------------
1. drift share over the frozen training reference (the primary, anchored signal)
2. predicted-class collapse (one class taking most of the traffic)

The second is a CATASTROPHE BACKSTOP, and the experiment is explicit about what
it does and does not buy. It catches the case where the model funnels everything
into one class (under a full cloud veil: 100% of predictions on a single class,
9.8% accuracy — chance level for ten classes). It does NOT fix the drift share's
measured blind spot: under blur, accuracy falls 97% -> 61% while the drift share
is still 0.00 and the majority share is only 0.28. Neither detector warns in
time there, and the README says so rather than implying the OR covers everything.

Mean prediction entropy is NOT a trigger. The same experiment showed it rising
and then falling back BELOW its baseline as the model became confidently wrong
(entropy 0.021 at rest, 0.145 mid-degradation, 0.008 at total collapse). A
threshold on it would report green at the worst possible moment.

What a trigger does NOT mean
---------------------------
It does not mean "the model is stale and retraining will fix it". It means "the
incoming data no longer resembles the training data". The cause may be a sensor
fault, an ingestion bug, or a genuinely new world — and only the third is fixed
by retraining. That is exactly why the workflow this dispatches ends at the
promotion gate and not at a deployment: see .github/workflows/retrain.yml.
"""

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from drift_reference import load_reference
from drift_report import compute_drift, fetch_current, summarize_predictions
from utils import REPO_ROOT, load_params

# --- persistent state ----------------------------------------------------------

def load_state(path: Path) -> Dict:
    if not path.exists():
        return {"consecutive_drift": 0, "last_dispatch": None, "history": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the tail only: this file is a decision log, not an archive. The full
    # reports live in monitoring/reports.
    state["history"] = state.get("history", [])[-20:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# --- the decision --------------------------------------------------------------

def evaluate(summary: Dict, predictions: Dict, params: dict) -> Dict:
    """Turn one report into a per-window verdict (does NOT consider history)."""
    cfg = params["monitor"]["trigger"]

    if summary.get("status") != "ok":
        return {
            "conclusive": False,
            "drift_detected": False,
            "reasons": [],
            "note": summary.get("message", "window could not be evaluated"),
        }

    reasons = []
    if summary.get("dataset_drift"):
        reasons.append(
            f"drift share {summary['drifted_share']:.2f} >= "
            f"{summary['drift_share_threshold']} "
            f"(top: {(summary.get('top_drifted') or [{}])[0].get('feature')})")

    shares = (predictions or {}).get("predicted_class_share") or {}
    if shares:
        top_class, top_share = max(shares.items(), key=lambda kv: kv[1])
        if top_share > float(cfg["majority_class_share_max"]):
            reasons.append(
                f"predicted-class collapse: {top_class} takes {top_share:.2f} "
                f"of traffic (> {cfg['majority_class_share_max']})")

    return {
        "conclusive": True,
        "drift_detected": bool(reasons),
        "reasons": reasons,
        "note": None,
    }


def decide(verdict: Dict, state: Dict, params: dict,
           now: datetime) -> Tuple[bool, str, Dict]:
    """Apply persistence and cooldown to a per-window verdict.

    Returns (should_dispatch, explanation, new_state). The state is returned
    rather than mutated in place so a dry run can show what WOULD happen without
    committing to it.
    """
    cfg = params["monitor"]["trigger"]
    needed = int(cfg["consecutive_windows"])
    cooldown = timedelta(minutes=int(cfg["cooldown_minutes"]))
    new_state = dict(state)

    if not verdict["conclusive"]:
        # Neither confirm nor reset: an unevaluable window is not evidence in
        # either direction. Resetting here would let a trickle of thin windows
        # silently disarm a real, persistent drift.
        return False, f"inconclusive window, streak unchanged ({verdict['note']})", new_state

    if not verdict["drift_detected"]:
        new_state["consecutive_drift"] = 0
        return False, "no drift in this window, streak reset to 0", new_state

    streak = int(state.get("consecutive_drift", 0)) + 1
    new_state["consecutive_drift"] = streak

    if streak < needed:
        return False, f"drift detected, streak {streak}/{needed} — not yet", new_state

    last = state.get("last_dispatch")
    if last:
        elapsed = now - datetime.fromisoformat(last)
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return False, (f"streak {streak}/{needed} reached but cooldown "
                           f"active ({remaining} left since {last})"), new_state

    return True, f"streak {streak}/{needed} reached and cooldown clear", new_state


# --- dispatch ------------------------------------------------------------------

def dispatch_workflow(params: dict, reason: str, repo: Optional[str] = None,
                      token: Optional[str] = None) -> str:
    """Fire the retraining workflow via workflow_dispatch.

    Tries the gh CLI first (it carries the user's own auth, which keeps the
    dispatch attributable to a person or a machine account rather than to an
    anonymous token), then falls back to the REST API with GITHUB_TOKEN.
    """
    cfg = params["monitor"]["trigger"]
    workflow, ref = cfg["workflow"], cfg["ref"]

    try:
        subprocess.run(
            ["gh", "workflow", "run", workflow, "--ref", ref,
             "-f", f"reason={reason}"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        return f"dispatched {workflow} on {ref} via gh"
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gh workflow run failed: {exc.stderr.strip()}") from exc

    token = token or os.environ.get("GITHUB_TOKEN")
    repo = repo or os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise RuntimeError(
            "no gh CLI available and GITHUB_TOKEN/GITHUB_REPOSITORY are unset — "
            "cannot dispatch. Run with --dispatch from an authenticated "
            "environment, or trigger the workflow manually.")

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
        data=json.dumps({"ref": ref, "inputs": {"reason": reason}}).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status not in (200, 204):
                raise RuntimeError(f"dispatch returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"dispatch failed: HTTP {exc.code} {exc.read()[:200]}") from exc
    return f"dispatched {workflow} on {ref} via REST"


# --- CLI -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=None,
                        help="X-TerraOps-Source filter (e.g. 'ui')")
    parser.add_argument("--since-minutes", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--db-uri", default=os.environ.get("TERRAOPS_DB_URI"))
    parser.add_argument("--dispatch", action="store_true",
                        help="actually trigger the retraining workflow "
                             "(default: dry run, decide and report only)")
    parser.add_argument("--state-path", type=Path, default=None)
    args = parser.parse_args()

    params = load_params()
    now = datetime.now(timezone.utc)
    state_path = args.state_path or (REPO_ROOT
                                     / params["monitor"]["trigger"]["state_path"])
    state = load_state(state_path)

    reference = load_reference(params)
    rows, window_meta = fetch_current(params, source=args.source,
                                      since_minutes=args.since_minutes,
                                      limit=args.window, db_uri=args.db_uri)
    summary, _ = compute_drift(reference["rows"], rows, params)
    predictions = summarize_predictions(rows)

    verdict = evaluate(summary, predictions, params)
    should_dispatch, explanation, new_state = decide(verdict, state, params, now)

    print(f"window: {window_meta['n_fetched']} rows "
          f"(source={window_meta['source_filter']})")
    print(f"verdict: conclusive={verdict['conclusive']} "
          f"drift={verdict['drift_detected']}")
    for reason in verdict["reasons"]:
        print(f"  - {reason}")
    if verdict["note"]:
        print(f"  note: {verdict['note']}")
    print(f"decision: {explanation}")

    entry = {
        "at": now.isoformat(),
        "n_rows": window_meta["n_fetched"],
        "conclusive": verdict["conclusive"],
        "drift_detected": verdict["drift_detected"],
        "reasons": verdict["reasons"],
        "decision": explanation,
        "dispatched": False,
    }

    if should_dispatch:
        reason = "; ".join(verdict["reasons"])
        if args.dispatch:
            message = dispatch_workflow(params, reason)
            print(message)
            new_state["last_dispatch"] = now.isoformat()
            # The streak resets ONLY on a real dispatch: otherwise a cooldown
            # window would quietly erase the evidence that drift is persistent.
            new_state["consecutive_drift"] = 0
            entry["dispatched"] = True
        else:
            print(f"DRY RUN — would dispatch retraining. Reason: {reason}")
            print("Re-run with --dispatch to fire it.")

    new_state.setdefault("history", []).append(entry)
    save_state(state_path, new_state)
    print(f"state -> {state_path}")

    # Exit codes for schedulers: 0 nothing to do, 2 drift acted on / actionable,
    # 3 could not conclude. A cron job that only ever returns 0 is a cron job
    # nobody notices has broken.
    if not verdict["conclusive"]:
        raise SystemExit(3)
    if verdict["drift_detected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
