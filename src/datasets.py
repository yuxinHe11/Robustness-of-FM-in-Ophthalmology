import os
from typing import Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(split: str, image_size: int = 224):
    """
    参考眼底foundation model benchmark常用设置:
    train: resize 256 -> random resized crop 224 -> flip -> normalize
    val/test: resize 256 -> center crop 224 -> normalize
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize(
                (256, 256),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize(
            (256, 256),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


class FundusDataset(Dataset):
    """
    输入csv格式:
        image,label,dataset,split

    image:
        可以是绝对路径，也可以是相对于 image_root 的相对路径

    label:
        二分类建议提前整理成 0/1
        如果是 2/3 这种，也会自动映射成 0/1
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        image_root: str = "",
        image_size: int = 224,
    ):
        self.csv_path = csv_path
        self.image_root = image_root
        self.split = split

        df = pd.read_csv(csv_path)

        required_cols = ["image", "label", "dataset", "split"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No samples found for split: {split}")

        self.df = df
        self.transform = build_transforms(split, image_size=image_size)

        labels = sorted(pd.read_csv(csv_path)["label"].unique().tolist())
        self.label_map = {label: idx for idx, label in enumerate(labels)}
        self.num_classes = len(labels)

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, image_path: str):
        image_path = str(image_path)

        if os.path.isabs(image_path):
            return image_path

        return os.path.join(self.image_root, image_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self._resolve_image_path(row["image"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {img_path}") from e

        image = self.transform(image)

        raw_label = row["label"]
        label = self.label_map[raw_label]
        label = torch.tensor(label, dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "path": img_path,
        }