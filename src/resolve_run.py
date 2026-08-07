"""Resolve the MLflow run produced by a given git commit.

The CT workflow needs to hand promote.py a run id after `dvc repro`, and there
are two ways to get one. Taking "the most recent run in the experiment" is the
obvious one and it is wrong: on a shared tracking server, someone else's run —
or a leftover from a crashed attempt — can be more recent than the retrain that
just finished, and the gate would then evaluate a model nobody asked about.

Selecting by the `git_commit` tag that train.py sets BEFORE training turns the
question into "the run produced by exactly this code", which is the lineage
claim the whole project is built on. Ambiguity (several runs for one commit) is
reported rather than silently resolved by recency.

Also refuses a run tagged git_dirty unless explicitly allowed: a dirty run's
commit does not identify what actually executed, so promoting from it would put
an unidentifiable model behind the @champion alias.
"""

import argparse
import subprocess
import sys

import mlflow

from utils import REPO_ROOT, load_params


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=REPO_ROOT, text=True).strip()


def resolve(commit: str, experiment: str, allow_dirty: bool = False,
            tracking_uri: str = None) -> str:
    """Return the single run id tagged with `commit`, or exit with a reason."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.git_commit = '{commit}'",
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    finished = [r for r in runs if r.info.status == "FINISHED"]
    if not finished:
        sys.exit(f"no FINISHED run tagged git_commit={commit[:8]} in "
                 f"experiment '{experiment}'. Did the training stage run?")

    if len(finished) > 1:
        # Ambiguous on purpose rather than "newest wins": two runs for one commit
        # means the pipeline ran twice, and picking one silently would hide that.
        ids = ", ".join(r.info.run_id for r in finished)
        sys.exit(f"{len(finished)} finished runs share git_commit={commit[:8]} "
                 f"({ids}). Refusing to guess which one to gate.")

    run = finished[0]
    dirty = str(run.data.tags.get("git_dirty", "False")).lower() == "true"
    if dirty and not allow_dirty:
        sys.exit(f"run {run.info.run_id} is tagged git_dirty=True: its commit "
                 f"does not identify what ran, so it must not be promoted. "
                 f"Commit the working tree and re-run, or pass --allow-dirty.")
    return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--commit", default=None, help="defaults to HEAD")
    parser.add_argument("--experiment", default=None,
                        help="defaults to the training experiment name")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    load_params()  # fail early if params.yaml is unreadable
    experiment = args.experiment or "terraops-eurosat"
    print(resolve(args.commit or current_commit(), experiment,
                  args.allow_dirty, args.tracking_uri))


if __name__ == "__main__":
    main()
