import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def import_official_vision_model(flair_repo: str = ""):
    if flair_repo:
        repo = str(Path(flair_repo).expanduser().resolve())
        if not Path(repo).exists():
            raise FileNotFoundError(f"--flair_repo does not exist: {repo}")
        if repo in sys.path:
            sys.path.remove(repo)
        sys.path.insert(0, repo)

    from src.FLAIR.flair.modeling import model as flair_modeling

    if not hasattr(flair_modeling, "VisionModel"):
        location = getattr(flair_modeling, "__file__", "<unknown>")
        raise ImportError(
            "Imported package 'flair.modeling.model' does not expose VisionModel. "
            f"Got {location}. Point --flair_repo to the official jusiro/FLAIR repo."
        )
    return flair_modeling.VisionModel


def torch_load_compat(path: str, map_location="cpu"):
    if str(path).endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "Loading model.safetensors requires the safetensors package. "
                "Install it in the server environment with: pip install safetensors"
            ) from e
        return load_file(path, device=str(map_location))

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "model", "state_dict", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def load_official_flair_vision_weights(vision_model: nn.Module, checkpoint_path: str):
    state = strip_module_prefix(get_state_dict(torch_load_compat(checkpoint_path)))
    vision_state = {}
    for k, v in state.items():
        if k.startswith("flair.vision_model."):
            vision_state[k[len("flair.vision_model."):]] = v
        elif k.startswith("vision_model."):
            vision_state[k[len("vision_model."):]] = v
        elif k.startswith("model.") or k.startswith("projection_head_vision."):
            vision_state[k] = v

    if not vision_state:
        vision_state = state

    model_state = vision_model.state_dict()
    keep = {}
    skipped = []
    for k, v in vision_state.items():
        if k not in model_state:
            continue
        if tuple(v.shape) != tuple(model_state[k].shape):
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        keep[k] = v

    if not any(k.startswith("model.") for k in keep):
        sample_keys = list(state.keys())[:20]
        raise RuntimeError(
            "No official FLAIR visual backbone weights were matched. "
            "Expected keys like 'vision_model.model.*' or 'model.*'. "
            f"First checkpoint keys: {sample_keys}"
        )

    msg = vision_model.load_state_dict(keep, strict=False)
    print(f"Loaded official FLAIR vision weights: {checkpoint_path}")
    print(f"Vision keys loaded={len(keep)}, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    for k, ckpt_shape, model_shape in skipped[:20]:
        print(f"Skipped mismatched key: {k}, ckpt={ckpt_shape}, model={model_shape}")
    return msg


class FLAIRRetinaTransform:
    """Match the official FLAIR preprocessing: scale to [0, 1], resize, pad."""

    def __init__(self, image_size: int = 512, is_train: bool = False, hflip_p: float = 0.5):
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.hflip_p = float(hflip_p)

    def __call__(self, image):
        if self.is_train and torch.rand(()) < self.hflip_p:
            image = TF.hflip(image)

        x = TF.to_tensor(image)
        _, height, width = x.shape
        max_size = max(height, width)
        if max_size <= 0:
            raise ValueError("Invalid image with non-positive size.")

        scale = self.image_size / float(max_size)
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        x = TF.resize(x, [new_h, new_w], antialias=True)

        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        if pad_h < 0 or pad_w < 0:
            x = TF.center_crop(x, [self.image_size, self.image_size])
        else:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        return x.float()


class FLAIRRetinaClassifier(nn.Module):
    """Official retinal FLAIR image encoder plus a supervised classifier head."""

    def __init__(
        self,
        pretrained: str,
        flair_repo: str = "",
        num_classes: int = 2,
        vision_type: str = "resnet_v1",
        image_size: int = 512,
        projection: bool = False,
        norm_features: bool = False,
        mode: str = "finetune",
        unfreeze_last_n_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not pretrained:
            raise ValueError("--pretrained is required for offline FLAIR v2.")
        if not os.path.exists(pretrained):
            raise FileNotFoundError(f"Pretrained FLAIR weights not found: {pretrained}")

        VisionModel = import_official_vision_model(flair_repo)
        self.vision_model = VisionModel(
            vision_type=vision_type,
            pretrained=False,
            proj_dim=512,
            proj_bias=False,
            projection=projection,
            norm=norm_features,
        )
        load_official_flair_vision_weights(self.vision_model, pretrained)

        self.image_size = int(image_size)
        self.preprocess_train = FLAIRRetinaTransform(image_size=image_size, is_train=True)
        self.preprocess_val = FLAIRRetinaTransform(image_size=image_size, is_train=False)
        self.feature_dim = self._infer_feature_dim()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )
        self.set_mode(mode, unfreeze_last_n_blocks)

    def _infer_feature_dim(self) -> int:
        was_training = self.vision_model.training
        device = next(self.vision_model.parameters()).device
        self.vision_model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=device)
            feat = self.encode_image_feature(dummy)
        self.vision_model.train(was_training)
        return int(feat.shape[-1])

    def freeze_all_flair(self):
        for p in self.vision_model.parameters():
            p.requires_grad = False

    def set_mode(self, mode: str, unfreeze_last_n_blocks: int = 1):
        self.freeze_all_flair()

        if mode == "linear_probe":
            pass
        elif mode == "finetune":
            for p in self.vision_model.parameters():
                p.requires_grad = True
        elif mode == "partial_finetune":
            backbone = self.vision_model.model
            children = [m for m in backbone.children() if any(p.numel() for p in m.parameters())]
            n = min(max(int(unfreeze_last_n_blocks), 0), len(children))
            for child in children[-n:]:
                for p in child.parameters():
                    p.requires_grad = True
            if getattr(self.vision_model, "projection_head_vision", None) is not None:
                for p in self.vision_model.projection_head_vision.parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        for p in self.classifier.parameters():
            p.requires_grad = True

    def encode_image_feature(self, image):
        feat = self.vision_model(image)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim == 3:
            feat = feat.mean(dim=1)
        return feat.float()

    def forward(self, image):
        return self.classifier(self.encode_image_feature(image))
