"""Training script for EuroSAT classification with ResNet-18.

Hyperparameters are read from params.yaml (no argparse): the DVC 'train' stage
runs this as `python src/train.py` and DVC tracks params.yaml as a dependency.
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import load_eurosat
from model import build_model
from utils import load_params, set_seed, setup_logging, get_device


# Pipeline output path — must match the 'outs' declared for this stage in dvc.yaml.
MODEL_PATH = "models/best_model.pth"


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
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, tcfg["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{tcfg['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    logger.info(f"Training complete in {elapsed / 60:.1f} min")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint: {MODEL_PATH}")


if __name__ == "__main__":
    main()
