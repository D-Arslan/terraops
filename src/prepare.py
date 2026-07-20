"""Prepare stage: materialize the raw EuroSAT dataset once.

Why a dedicated stage: the original code downloaded EuroSAT inside load_eurosat(),
which is called by BOTH train and evaluate. Isolating the download into its own
DAG node means:
  - it runs once, is cached by DVC, and never re-triggers train/evaluate;
  - train/evaluate declare data/raw as a *dependency* and only consume it.
This is the boundary that makes `prepare -> train -> evaluate` a clean DAG.
"""

import os

from torchvision import datasets

from utils import load_params


def main():
    params = load_params()
    data_dir = params["prepare"]["data_dir"]

    os.makedirs(data_dir, exist_ok=True)

    # download=True is idempotent: torchvision verifies the archive and skips
    # the download if the files are already present. Materializes
    # <data_dir>/eurosat/2750/<Class>/*.jpg
    datasets.EuroSAT(root=data_dir, download=True)

    n_images = sum(len(files) for _, _, files in os.walk(data_dir))
    print(f"EuroSAT ready under '{data_dir}' ({n_images} files on disk)")


if __name__ == "__main__":
    main()
