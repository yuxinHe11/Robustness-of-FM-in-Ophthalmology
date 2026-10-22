from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MinMaxNormalize:
    """Normalize each image channel independently to [0, 1]."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone()
        for channel in range(tensor.shape[0]):
            values = tensor[channel]
            minimum = values.min()
            maximum = values.max()
            if maximum > minimum:
                tensor[channel] = (values - minimum) / (maximum - minimum)
            else:
                tensor[channel].zero_()
        return tensor


def build_mirage_transforms(split: str, image_size: int = 512):
    operations = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        MinMaxNormalize(),
    ]

    if split == "train":
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
        ])

    return transforms.Compose(operations)


class MIRAGEOCTDataset(Dataset):
    """Read image,label,dataset,split manifests for MIRAGE."""

    def __init__(
        self,
        csv_path: str,
        split: str,
        image_root: str = "",
        image_size: int = 512,
    ):
        self.image_root = image_root
        self.split = split.lower()

        df = pd.read_csv(csv_path)
        required = {"image", "label", "dataset", "split"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing columns: {', '.join(sorted(missing))}"
            )

        df = df.copy()
        df["split"] = df["split"].astype(str).str.strip().str.lower()
        df = df[df["split"] == self.split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No samples found for split: {split}")

        all_labels = sorted(pd.read_csv(csv_path)["label"].unique().tolist())
        self.label_map = {label: index for index, label in enumerate(all_labels)}
        self.num_classes = len(all_labels)
        self.df = df
        self.transform = build_mirage_transforms(
            self.split, image_size=image_size
        )

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, image_path: str) -> str:
        image_path = str(image_path)
        if os.path.isabs(image_path):
            return image_path
        return os.path.join(self.image_root, image_path)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self._resolve_image_path(row["image"])
        try:
            image = Image.open(image_path)
            image.load()
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc

        image = self.transform(image)
        label = torch.tensor(
            self.label_map[row["label"]], dtype=torch.long
        )
        return {"image": image, "label": label, "path": image_path}
