"""Evaluation script: metrics, confusion matrix, and error analysis.

Reads hyperparameters from params.yaml and the model produced by the 'train'
stage. Writes a machine-readable metrics.json (tracked by DVC) plus plots.
"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dataset import load_eurosat, EUROSAT_CLASSES
from model import build_model
from utils import load_params, set_seed, get_device


# Pipeline paths — must match the deps/outs declared for this stage in dvc.yaml.
MODEL_PATH = "models/best_model.pth"
METRICS_DIR = "metrics"
METRICS_JSON = os.path.join(METRICS_DIR, "metrics.json")


def denormalize(tensor, mean, std):
    """Reverse normalization for visualization (mean/std from params)."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor.cpu() * std + mean


@torch.no_grad()
def evaluate(model, loader, device):
    """Run evaluation, return predictions and ground truth."""
    model.eval()
    all_preds, all_labels, all_probs, all_images = [], [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_images.extend(images.cpu())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_images


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EUROSAT_CLASSES, yticklabels=EUROSAT_CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — EuroSAT Classification")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")


def plot_misclassified(images, y_true, y_pred, probs, output_dir, mean, std, n=9):
    """Show most-confident misclassified examples for error analysis."""
    misclassified_idx = np.where(y_true != y_pred)[0]
    if len(misclassified_idx) == 0:
        print("No misclassified examples!")
        return

    confidences = [probs[i][y_pred[i]] for i in misclassified_idx]
    sorted_idx = np.argsort(confidences)[::-1]
    selected = misclassified_idx[sorted_idx[:n]]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, idx in zip(axes.flat, selected):
        img = denormalize(images[idx], mean, std).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(
            f"True: {EUROSAT_CLASSES[y_true[idx]]}\n"
            f"Pred: {EUROSAT_CLASSES[y_pred[idx]]} "
            f"({probs[idx][y_pred[idx]]:.2%})",
            fontsize=9,
        )
        ax.axis("off")

    plt.suptitle("Most Confident Misclassifications", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "misclassified.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved misclassified examples to {path}")


def main():
    params = load_params()
    seed = params["seed"]
    mcfg = params["model"]
    mean, std = params["data"]["norm_mean"], params["data"]["norm_std"]

    set_seed(seed)
    device = get_device()
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Test set — batch_size is the evaluate stage's own param.
    _, _, test_loader = load_eurosat(params, batch_size=params["evaluate"]["batch_size"])

    # Rebuild identical architecture, then load trained weights.
    model = build_model(
        num_classes=mcfg["num_classes"], pretrained=False, dropout=mcfg["dropout"]
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f}")

    y_pred, y_true, probs, images = evaluate(model, test_loader, device)

    # Human-readable report
    report = classification_report(y_true, y_pred, target_names=EUROSAT_CLASSES, digits=4)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    with open(os.path.join(METRICS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, METRICS_DIR)
    plot_misclassified(images, y_true, y_pred, probs, METRICS_DIR, mean, std)

    # Machine-readable metrics for DVC (small, versioned in Git, diffable).
    metrics = {
        "test_accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "n_test": int(len(y_true)),
        "n_misclassified": int(np.sum(y_true != y_pred)),
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTest Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Metrics written to {METRICS_JSON}")


if __name__ == "__main__":
    main()
