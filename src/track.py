"""Sync every training fold and leaderboard submission into a local MLflow store.

Training runs on Kaggle, so this reads what each training kernel saved (downloaded into
data/runs/<kernel>/ with `kaggle kernels output`) and the submission list from the Kaggle API:

  experiment "rsna-knee/training"     one run per kernel+fold: all train.py arguments as params,
                                      loss / val AUC / gold AUC per epoch, per-finding gold AUC,
                                      the gold predictions and log as artifacts
  experiment "rsna-knee/submissions"  one run per submission: kernel, description, public LB score

Runs are keyed by tags, so re-running only adds what is new (and fills in late LB scores).

    python src/track.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
"""
import json
import re
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-knee-abnormality-detection"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

# What each training kernel was for, shown as a run description in the UI
NOTES = {
    "k03b-effv2s": "v1: 224 px, lr 3e-4, 8 epochs",
    "k06a-effv2s-r256": "v2: 256 px, lr 3e-4, 14 epochs; diverged in epochs 1-2",
    "k06b-effv2s-r256": "v2: 256 px, lr 3e-4, 14 epochs; fold 3 diverged, fold 2 fine",
    "k07a-effv2s-r256-lr15": "v3: 256 px, lr 1.5e-4, 1-epoch warm-up, 12 epochs",
    "k07b-effv2s-r256-lr15": "v3: 256 px, lr 1.5e-4, 1-epoch warm-up, 12 epochs",
}


def existing(client, exp_id, key):
    """Map tag value -> run for runs in an experiment that carry tag `key`."""
    runs = client.search_runs([exp_id], max_results=5000)
    return {r.data.tags[key]: r for r in runs if key in r.data.tags}


def sync_training(client, gold):
    exp_id = mlflow.set_experiment("rsna-knee/training").experiment_id
    done = existing(client, exp_id, "run_key")
    added = 0
    for log_path in sorted((ROOT / "data" / "runs").glob("*/log_*.json")):
        kernel, tag = log_path.parent.name, log_path.stem[4:]
        key = f"{kernel}/{tag}"
        if key in done:
            continue
        info = json.loads(log_path.read_text())
        args, log = info["args"], info["log"]
        with mlflow.start_run(run_name=key, description=NOTES.get(kernel, "")):
            mlflow.set_tags({"run_key": key, "kernel": kernel, "fold": args["fold"]})
            args.setdefault("cache", 256)  # runs before the 384 px cache existed
            args.setdefault("warmup", 0.5)
            mlflow.log_params({k: v for k, v in args.items() if k not in ("out", "workers", "limit")})
            for row in log:
                mlflow.log_metrics({"loss": row["loss"], "val_auc_vs_reports": row["val_auc_vs_reports"],
                                    "gold_auc": row["gold_auc"], "epoch_minutes": row["minutes"]},
                                   step=row["epoch"])
            gold_csv = log_path.with_name(f"gold_{tag}.csv")
            if gold_csv.exists():
                pred = pd.read_csv(gold_csv).set_index("StudyInstanceUID").loc[gold.index]
                per = {c: roc_auc_score(gold[c], pred[c]) for c in gold.columns}
                mlflow.log_metrics({f"gold_{re.sub(r'[^A-Za-z]+', '_', c).strip('_')}": v for c, v in per.items()})
                mlflow.log_metric("final_gold_auc", sum(per.values()) / len(per))
                mlflow.log_artifact(str(gold_csv))
            mlflow.log_metric("final_val_auc_vs_reports", log[-1]["val_auc_vs_reports"])
            mlflow.log_artifact(str(log_path))
        added += 1
    return added


def sync_submissions(client):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions(COMP)
    exp_id = mlflow.set_experiment("rsna-knee/submissions").experiment_id
    done = existing(client, exp_id, "submission_ref")
    added = 0
    for s in subs:
        ref, desc = str(s.ref), s.description or ""
        score = s.public_score
        score = float(score) if score not in (None, "") else None
        run = done.get(ref)
        if run is not None:  # fill in a score that arrived after the last sync
            if score is not None and "public_lb" not in run.data.metrics:
                client.log_metric(run.info.run_id, "public_lb", score)
            continue
        kernel = desc.split(":")[0].strip() if ":" in desc else ""
        with mlflow.start_run(run_name=kernel or ref, description=desc):
            mlflow.set_tags({"submission_ref": ref, "kernel": kernel, "status": str(s.status)})
            mlflow.log_param("submitted_at", str(s.date))
            if score is not None:
                mlflow.log_metric("public_lb", score)
        added += 1
    return added


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    labels = list(train.columns[2:])
    gold = train.dropna(subset=labels).set_index("StudyInstanceUID")[labels].astype(int)
    print("training runs added:", sync_training(client, gold))
    print("submissions added:", sync_submissions(client))
    print(f"view with: mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main()
