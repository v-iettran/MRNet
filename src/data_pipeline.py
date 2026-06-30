"""Data pipeline: loading, preprocessing, augmentation, DataLoaders.

Owner: Albee + Noma

Dataset
---------------------------------------------------
MRNet (Stanford, Bien et al. 2018): 1,370 knee MRI exams, each with three planes
(axial / coronal / sagittal). This project uses the **sagittal** plane and the
**ACL-tear** binary task (positive prevalence ~23%, hence the class weighting).
Each exam is stored as a single ``.npy`` volume of shape ``(slices, H, W)`` with
a VARIABLE number of slices per exam (~17-61), which is why the DataLoader uses
``batch_size=1`` and the model pools over the slice dimension. Labels come from
``{split}_{task}.csv``. Preprocessing: center-crop/pad to a fixed size (256 for
CNNs, 224 for MedViT), z-score intensity normalisation, optional augmentation,
and channel-repeat to 3 channels for ImageNet-pretrained backbones. The external
test set (Rijeka KneeMRI) uses the SAME ``MRNetTransform`` (see
``for-gpu/eval_external.py``).

This mirrors the team's ``data_processing.py``. Compared to a plain loader it:
  * reads MRNet label CSVs (tolerant to a couple of common formats),
  * center-crops/pads each exam to a fixed size and normalizes intensities,
  * applies the SAME sampled augmentation to every slice of an exam,
  * optionally repeats the single MRI channel to 3 channels (for ImageNet
    pretrained backbones such as MedViT),
  * builds train/valid DataLoaders (batch = 1 exam, variable #slices),
  * exposes ``get_pos_weight`` for class-imbalance handling via
    ``BCEWithLogitsLoss(pos_weight=...)`` (NOT per-sample weights).

Each dataset item is ``(image, label)`` where:
  * image : float tensor ``(slices, 3, H, W)`` (3 channels if repeat_channels),
  * label : float tensor ``(1,)`` (0/1).

``root_dir`` defaults to ``config.DATA_DIR`` so notebooks can just call
``build_dataloaders(task=..., plane=...)``.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from . import config

VALID_TASKS = config.TASKS
VALID_PLANES = config.PLANES


def _resolve_root(root_dir):
    """Default ``root_dir`` to ``config.DATA_DIR`` when not provided."""
    return str(config.DATA_DIR) if root_dir is None else str(root_dir)


AUGMENTATION_PRESETS = {
    "none": {
        "rotation": 0,
        "translate": 0.0,
        "hflip": 0.0,
        "vflip": 0.0,
        "scale": (1.0, 1.0),
        "noise_std": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
    },
    "light": {
        "rotation": 10,
        "translate": 0.04,
        "hflip": 0.5,
        "vflip": 0.0,
        "scale": (0.97, 1.03),
        "noise_std": 0.01,
        "brightness": 0.05,
        "contrast": 0.05,
    },
    "medium": {
        "rotation": 20,
        "translate": 0.08,
        "hflip": 0.5,
        "vflip": 0.0,
        "scale": (0.92, 1.08),
        "noise_std": 0.02,
        "brightness": 0.08,
        "contrast": 0.08,
    },
    "strong": {
        "rotation": 30,
        "translate": 0.12,
        "hflip": 0.5,
        "vflip": 0.1,
        "scale": (0.88, 1.12),
        "noise_std": 0.03,
        "brightness": 0.12,
        "contrast": 0.12,
    },
}


def pad_id(patient_id):
    return str(patient_id).zfill(4)


def read_mrnet_records(csv_path):
    raw_records = pd.read_csv(csv_path, header=None)

    if raw_records.shape[1] >= 2:
        records = raw_records.iloc[:, :2].copy()
        records.columns = ["id", "label"]
    else:
        # Some copied MRNet label files store each row like "0001_0"
        # instead of two comma-separated columns. This supports both.
        rows = raw_records.iloc[:, 0].astype(str).str.strip()
        extracted = rows.str.extract(r"^(.*?)[_,\s]+([01])$")
        if extracted.isnull().any().any():
            raise ValueError(
                "Could not read labels from {}. Expected rows like "
                "'0001,0' or '0001_0'.".format(csv_path)
            )
        records = extracted
        records.columns = ["id", "label"]

    records["id"] = (
        records["id"]
        .astype(str)
        .str.strip()
        .str.replace(".npy", "", regex=False)
        .str.extract(r"(\d+)")[0]
        .map(pad_id)
    )
    records["label"] = (
        records["label"]
        .astype(str)
        .str.strip()
        .str.replace(r"^_+", "", regex=True)
        .str.extract(r"([01])")[0]
        .astype(int)
    )

    return records


def get_pos_weight(root_dir=None, task="acl", train=True):
    """Return ``pos_weight = neg/pos`` for ``BCEWithLogitsLoss`` (per task).

    Computed from the TRAIN split by default. Use this for class-imbalance
    handling instead of per-sample weights.
    """
    root_dir = _resolve_root(root_dir)
    split = "train" if train else "valid"
    csv_path = os.path.join(root_dir, "{}_{}.csv".format(split, task))
    records = read_mrnet_records(csv_path)

    pos = float(records["label"].sum())
    neg = float(len(records) - pos)
    if pos == 0:
        return torch.FloatTensor([1.0])
    return torch.FloatTensor([neg / pos])


def get_expected_paths(root_dir=None, task="acl", plane="sagittal", train=True):
    root_dir = _resolve_root(root_dir)
    split = "train" if train else "valid"
    csv_path = os.path.join(root_dir, "{}_{}.csv".format(split, task))
    folder_path = os.path.join(root_dir, split, plane)
    records = read_mrnet_records(csv_path)
    paths = [
        os.path.join(folder_path, patient_id + ".npy")
        for patient_id in records["id"].tolist()
    ]
    return records, paths


def check_dataset_files(root_dir=None, task="acl", plane="sagittal", train=True,
                        max_show=10):
    root_dir = _resolve_root(root_dir)
    split = "train" if train else "valid"
    csv_path = os.path.join(root_dir, "{}_{}.csv".format(split, task))
    folder_path = os.path.join(root_dir, split, plane)

    print("Checking", split, task, plane)
    print("CSV path:", csv_path)
    print("Image folder:", folder_path)
    print("CSV exists:", os.path.exists(csv_path))
    print("Image folder exists:", os.path.isdir(folder_path))

    if not os.path.exists(csv_path) or not os.path.isdir(folder_path):
        return None

    records, paths = get_expected_paths(root_dir, task, plane, train=train)
    missing = [path for path in paths if not os.path.exists(path)]

    print("Rows in CSV:", len(records))
    print("Expected .npy files:", len(paths))
    print("Missing .npy files:", len(missing))

    if missing:
        print("First missing files:")
        for path in missing[:max_show]:
            print("  ", path)
        print("\nFirst files that actually exist in the image folder:")
        try:
            for name in sorted(os.listdir(folder_path))[:max_show]:
                print("  ", name)
        except OSError as error:
            print("Could not list folder:", error)

    return missing


def center_crop_or_pad(array, output_size=256):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    target_h, target_w = output_size
    slices, height, width = array.shape

    crop_top = max((height - target_h) // 2, 0)
    crop_left = max((width - target_w) // 2, 0)
    array = array[
        :,
        crop_top:crop_top + min(height, target_h),
        crop_left:crop_left + min(width, target_w),
    ]

    pad_h = max(target_h - array.shape[1], 0)
    pad_w = max(target_w - array.shape[2], 0)

    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        array = np.pad(
            array,
            ((0, 0), (top, bottom), (left, right)),
            mode="constant",
        )

    return array


def normalize_mri(array, method="zscore", clip_percentiles=(1, 99)):
    array = array.astype(np.float32)

    if clip_percentiles is not None:
        low, high = np.percentile(array, clip_percentiles)
        if high > low:
            array = np.clip(array, low, high)

    if method == "none":
        return array

    if method == "minmax":
        min_value = np.min(array)
        max_value = np.max(array)
        if max_value > min_value:
            array = (array - min_value) / (max_value - min_value)
        return array

    if method == "zscore":
        mean = np.mean(array)
        std = np.std(array)
        if std > 0:
            array = (array - mean) / std
        else:
            array = array - mean
        return array

    raise ValueError("method should be 'zscore', 'minmax', or 'none'")


class MRNetAugmentor:
    def __init__(self, config="light"):
        if isinstance(config, str):
            if config not in AUGMENTATION_PRESETS:
                raise ValueError("Unknown augmentation preset: {}".format(config))
            config = AUGMENTATION_PRESETS[config]

        self.config = dict(AUGMENTATION_PRESETS["none"])
        self.config.update(config)

    def __call__(self, tensor):
        # tensor size is (slices, 1, height, width). The same sampled
        # transform is applied to all slices in one MRI exam.
        slices, channels, height, width = tensor.shape

        if random.random() < self.config["hflip"]:
            tensor = torch.flip(tensor, dims=(-1,))

        if random.random() < self.config["vflip"]:
            tensor = torch.flip(tensor, dims=(-2,))

        rotation = self.config["rotation"]
        translate = self.config["translate"]
        scale_range = self.config["scale"]

        if rotation > 0 or translate > 0 or scale_range != (1.0, 1.0):
            angle = random.uniform(-rotation, rotation)
            max_dx = int(translate * width)
            max_dy = int(translate * height)
            translations = (
                random.randint(-max_dx, max_dx) if max_dx > 0 else 0,
                random.randint(-max_dy, max_dy) if max_dy > 0 else 0,
            )
            scale = random.uniform(scale_range[0], scale_range[1])

            tensor = TF.affine(
                tensor,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

        brightness = self.config["brightness"]
        if brightness > 0:
            tensor = tensor + random.uniform(-brightness, brightness)

        contrast = self.config["contrast"]
        if contrast > 0:
            mean = tensor.mean(dim=(-2, -1), keepdim=True)
            factor = random.uniform(1 - contrast, 1 + contrast)
            tensor = (tensor - mean) * factor + mean

        noise_std = self.config["noise_std"]
        if noise_std > 0:
            tensor = tensor + torch.randn_like(tensor) * noise_std

        return tensor


class MRNetTransform:
    def __init__(self, augment="none", normalize="zscore", output_size=256,
                 repeat_channels=True, clip_percentiles=(1, 99)):
        self.augment = None if augment in [None, "none"] else MRNetAugmentor(augment)
        self.normalize = normalize
        self.output_size = output_size
        self.repeat_channels = repeat_channels
        self.clip_percentiles = clip_percentiles

    def __call__(self, array):
        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)
        if array.ndim != 3:
            raise ValueError("Expected MRI array with shape (slices, height, width)")

        array = center_crop_or_pad(array, self.output_size)
        array = normalize_mri(array, self.normalize, self.clip_percentiles)

        tensor = torch.FloatTensor(array).unsqueeze(1)

        if self.augment is not None:
            tensor = self.augment(tensor)

        if self.repeat_channels:
            tensor = tensor.repeat(1, 3, 1, 1)

        return tensor


# A torch Dataset implementing the required __init__ / __len__ / __getitem__.
# __init__ loads the label CSV
# and the per-exam .npy paths; __len__ returns the exam count; __getitem__ loads
# one volume, applies MRNetTransform, and returns (image, label) tensors. It is
# wrapped by a DataLoader in build_dataloaders() and consumed by the training
# loop in training_utils.run_training (batch_size=1 -> one exam per step).
class MRNetDataset(data.Dataset):
    def __init__(self, root_dir=None, task="acl", plane="sagittal", train=False,
                 transform=None, augment=None, normalize="zscore", output_size=256):
        # __init__: read labels + resolve the .npy path for every exam.
        super().__init__()

        if task not in VALID_TASKS:
            raise ValueError("task should be one of {}".format(VALID_TASKS))
        if plane not in VALID_PLANES:
            raise ValueError("plane should be one of {}".format(VALID_PLANES))

        self.task = task
        self.plane = plane
        self.root_dir = _resolve_root(root_dir)
        self.train = train

        split = "train" if self.train is True else "valid"
        self.folder_path = os.path.join(self.root_dir, split, plane)
        csv_path = os.path.join(self.root_dir, "{}_{}.csv".format(split, task))

        self.records = read_mrnet_records(csv_path)
        self.paths = [
            os.path.join(self.folder_path, filename + ".npy")
            for filename in self.records["id"].tolist()
        ]
        self.labels = self.records["label"].tolist()

        if transform is not None:
            self.transform = transform
        else:
            if augment is None:
                augment = "none"
            self.transform = MRNetTransform(
                augment=augment,
                normalize=normalize,
                output_size=output_size,
            )

    def __len__(self):
        # __len__: number of exams in this split (drives DataLoader iteration).
        return len(self.paths)

    def __getitem__(self, index):
        # __getitem__: load one exam, transform it, return (image, label).
        path = self.paths[index]
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing MRI file: {}. Check that root_dir points to the MRNet "
                "root folder, that plane='{}' is available, and that the CSV "
                "labels match the .npy files.".format(path, self.plane)
            )

        array = np.load(path)
        image = self.transform(array)
        label = torch.FloatTensor([self.labels[index]])

        return image, label


Dataset = MRNetDataset


def build_transform(augment="light", normalize="zscore", output_size=256,
                    repeat_channels=True, clip_percentiles=(1, 99)):
    return MRNetTransform(
        augment=augment,
        normalize=normalize,
        output_size=output_size,
        repeat_channels=repeat_channels,
        clip_percentiles=clip_percentiles,
    )


def build_datasets(root_dir=None, task="acl", plane="sagittal", train_augment="light",
                   valid_augment="none", normalize="zscore", output_size=256):
    train_transform = build_transform(
        augment=train_augment,
        normalize=normalize,
        output_size=output_size,
    )
    valid_transform = build_transform(
        augment=valid_augment,
        normalize=normalize,
        output_size=output_size,
    )

    train_dataset = Dataset(
        root_dir, task, plane, train=True, transform=train_transform
    )
    valid_dataset = Dataset(
        root_dir, task, plane, train=False, transform=valid_transform
    )

    return train_dataset, valid_dataset


def build_dataloaders(root_dir=None, task="acl", plane="sagittal",
                      train_augment="light", batch_size=1, num_workers=2,
                      normalize="zscore", output_size=256, pin_memory=None):
    if batch_size != 1:
        raise ValueError(
            "Use batch_size=1 because each MRNet exam can have a different "
            "number of slices. Use gradient accumulation for a larger "
            "effective batch size."
        )

    # Pinned host memory enables faster, non-blocking host->GPU transfers.
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    # Keep workers warm between epochs (avoids re-spawn overhead) when used.
    persistent = num_workers > 0

    train_dataset, valid_dataset = build_datasets(
        root_dir=root_dir,
        task=task,
        plane=plane,
        train_augment=train_augment,
        normalize=normalize,
        output_size=output_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, valid_loader


def describe_augmentations():
    rows = []
    for name, cfg in AUGMENTATION_PRESETS.items():
        row = {"preset": name}
        row.update(cfg)
        rows.append(row)
    return pd.DataFrame(rows)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
