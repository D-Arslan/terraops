"""EuroSAT dataset loading and preprocessing."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from preprocessing import build_eval_transform

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


def get_transforms(data_cfg: dict, train: bool = True):
    """Build transforms from params. Augmentation applies to the train split only.

    The EVAL branch (train=False) is NOT defined here anymore: it is the shared
    inference contract, owned by preprocessing.build_eval_transform and imported
    by train.py, the promotion gate, AND the serving API. One definition, so
    train/serving skew cannot exist. Augmentation stays train-only and local.
    """
    if not train:
        return build_eval_transform(data_cfg)

    size = data_cfg["image_size"]
    mean, std = data_cfg["norm_mean"], data_cfg["norm_std"]
    aug = data_cfg["augment"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=aug["hflip_p"]),
        transforms.RandomVerticalFlip(p=aug["vflip_p"]),
        transforms.RandomRotation(aug["rotation_deg"]),
        transforms.ColorJitter(
            brightness=aug["jitter_brightness"],
            contrast=aug["jitter_contrast"],
            saturation=aug["jitter_saturation"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_eurosat(params: dict, batch_size: int):
    """Load EuroSAT with a seeded train/val/test split.

    Reads the already-materialized dataset from params['prepare']['data_dir']
    (download=False: fetching is the 'prepare' stage's job). Split ratios and
    seed come from params, so DVC can detect any change to them.
    """
    data_dir = params["prepare"]["data_dir"]
    seed = params["seed"]
    data_cfg = params["data"]
    val_split = data_cfg["val_split"]
    test_split = data_cfg["test_split"]
    num_workers = data_cfg["num_workers"]

    # download=False: the 'prepare' stage already materialized the data.
    full_dataset = datasets.EuroSAT(root=data_dir, download=False, transform=None)

    # Split ratios derived from params; arithmetic matches the original run
    # (n_train = int(0.70 * n)) so the partition — and the 97.8% — reproduces.
    n_total = len(full_dataset)
    n_train = int((1 - val_split - test_split) * n_total)
    n_val = int(val_split * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    train_set = TransformSubset(train_set, get_transforms(data_cfg, train=True))
    val_set = TransformSubset(val_set, get_transforms(data_cfg, train=False))
    test_set = TransformSubset(test_set, get_transforms(data_cfg, train=False))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Dataset split: train={n_train}, val={n_val}, test={n_test}")
    return train_loader, val_loader, test_loader


class TransformSubset(torch.utils.data.Dataset):
    """Apply transforms to a Subset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)
