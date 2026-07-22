"""Training script for EuroSAT classification, instrumented with MLflow.

Hyperparameters are read from params.yaml (no argparse): the DVC 'train' stage
runs this as `python src/train.py` and DVC tracks params.yaml as a dependency.

Every execution becomes one MLflow run:
  - params: the flattened params.yaml sections (model / train / data)
  - metrics: train/val loss & acc per epoch, plus end-of-run summary
  - artifacts: learning curves, validation confusion matrix, the best model
  - LINEAGE tags: git commit + DVC data hash, so any run can be traced back to
    the exact code AND the exact data bytes that produced it.
"""

import os
import time
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import mlflow
import mlflow.pytorch
import matplotlib
matplotlib.use("Agg")  # no display needed; we only save PNGs for MLflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset import load_eurosat, EUROSAT_CLASSES
from model import build_model
from utils import (load_params, set_seed, setup_logging, get_device,
                   get_git_commit, get_dvc_data_hash, flatten_params)


# Pipeline output path — must match the 'outs' declared for this stage in dvc.yaml.
MODEL_PATH = "models/best_model.pth"

# Env-var override so the same script works locally and in CI without edits.
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "terraops-eurosat"


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model, return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def predict(model, loader, device):
    """Collect predictions and labels over a loader (for the confusion matrix)."""
    model.eval()
    preds, labels_all = [], []
    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        outputs = model(images.to(device))
        preds.extend(outputs.argmax(1).cpu().numpy())
        labels_all.extend(labels.numpy())
    return np.array(labels_all), np.array(preds)


def plot_learning_curves(history, path):
    """Loss + accuracy curves, train vs val — the overfitting X-ray of a run."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("epoch"), ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss"), ax_loss.legend()
    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_xlabel("epoch"), ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy"), ax_acc.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_val_confusion_matrix(y_true, y_pred, path):
    """Confusion matrix on the VALIDATION split (test stays untouched here)."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EUROSAT_CLASSES, yticklabels=EUROSAT_CLASSES)
    plt.xlabel("Predicted"), plt.ylabel("True")
    plt.title("Confusion Matrix — validation split (best checkpoint)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    params = load_params()
    seed = params["seed"]
    tcfg = params["train"]
    mcfg = params["model"]

    # Reproducibility
    set_seed(seed)

    # Setup
    logger = setup_logging()
    device = get_device()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"{mcfg['arch']}-lr{tcfg['lr']}-{mcfg['unfreeze']}"

    with mlflow.start_run(run_name=run_name):
        # --- LINEAGE FIRST: even if training crashes, the run is traceable ---
        commit, dirty = get_git_commit()
        data_hash = get_dvc_data_hash()
        mlflow.set_tags({
            "git_commit": commit,
            "git_dirty": str(dirty).lower(),
            "dvc_data_hash": data_hash,
            "arch": mcfg["arch"],
        })
        if dirty:
            logger.warning(
                "Working tree is DIRTY: git_commit does not fully identify "
                "this code. Commit before real experiments."
            )
        logger.info(f"Lineage: commit={commit[:8]} dirty={dirty} data={data_hash}")

        # Everything tunable, flattened: train.lr, model.arch, data.augment.hflip_p...
        mlflow.log_params({"seed": seed, **flatten_params(
            {"model": mcfg, "train": tcfg, "data": params["data"]}
        )})

        logger.info(f"Config: epochs={tcfg['epochs']}, lr={tcfg['lr']}, "
                    f"batch_size={tcfg['batch_size']}, seed={seed}, "
                    f"patience={tcfg['patience']}")
        logger.info(f"Device: {device}")

        # Data (batch_size is the train stage's own param)
        train_loader, val_loader, _ = load_eurosat(params, batch_size=tcfg["batch_size"])

        # Model
        model = build_model(
            num_classes=mcfg["num_classes"],
            pretrained=mcfg["pretrained"],
            dropout=mcfg["dropout"],
            arch=mcfg["arch"],
            unfreeze=mcfg["unfreeze"],
        ).to(device)

        # Loss: CrossEntropyLoss (standard for multi-class classification)
        criterion = nn.CrossEntropyLoss()

        # Optimizer: Adam with L2 regularization (weight_decay)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )

        # Scheduler: reduce LR on validation loss plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=tcfg["scheduler_factor"],
            patience=tcfg["scheduler_patience"],
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        start_time = time.time()

        for epoch in range(1, tcfg["epochs"] + 1):
            logger.info(f"Epoch {epoch}/{tcfg['epochs']}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
                history[key].append(value)
            # step=epoch gives MLflow the x-axis for its metric charts
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                }, MODEL_PATH)
                logger.info(f"Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{tcfg['patience']})")
                if patience_counter >= tcfg["patience"]:
                    logger.info("Early stopping triggered.")
                    break

        elapsed = time.time() - start_time

        # Summary metrics: what you sort/filter runs by in the MLflow UI
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "train_minutes": elapsed / 60,
        })

        # Reload the BEST checkpoint: the last epoch may be a post-peak state —
        # what we archive and (maybe) promote is the best one, not the last one.
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        with tempfile.TemporaryDirectory() as tmp:
            curves = os.path.join(tmp, "learning_curves.png")
            plot_learning_curves(history, curves)
            mlflow.log_artifact(curves)

            y_true, y_pred = predict(model, val_loader, device)
            cm_path = os.path.join(tmp, "confusion_matrix_val.png")
            plot_val_confusion_matrix(y_true, y_pred, cm_path)
            mlflow.log_artifact(cm_path)

        # The model itself, as an MLflow artifact -> loadable via
        # "runs:/<run_id>/model" and registrable in the Model Registry.
        mlflow.pytorch.log_model(model, name="model")

        logger.info(f"Training complete in {elapsed / 60:.1f} min")
        logger.info(f"Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}, "
                    f"val_acc={best_val_acc:.4f}")
        logger.info(f"Checkpoint: {MODEL_PATH}")
        logger.info(f"MLflow run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
