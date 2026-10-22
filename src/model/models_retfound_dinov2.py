import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class RETFoundDINOv2Classifier(nn.Module):
    """
    RETFound-DINOv2 backbone + linear classification head.

    适配:
        RETFound_dinov2_meh.pth
        RETFound_dinov2_shanghai.pth

    已确认:
        checkpoint["teacher"] 是 OrderedDict
        checkpoint key: backbone.xxx
        backbone dim: 1024
        original pos_embed: [1, 1370, 1024] = 1 CLS + 37*37 patches
        original image size: 518, patch size: 14

    本文件支持 image_size=224：
        自动把 pos_embed 从 37x37 插值到 16x16。
    """

    def __init__(
        self,
        arch: str = "vit_large_patch14_dinov2",
        pretrained_path: str = "",
        num_classes: int = 2,
        mode: str = "finetune",
        dropout: float = 0.0,
        image_size: int = 224,
    ):
        super().__init__()

        self.arch = arch
        self.pretrained_path = pretrained_path
        self.image_size = image_size

        self.backbone = timm.create_model(
            arch,
            pretrained=False,
            num_classes=0,
            img_size=image_size,
        )

        self._load_retfound_weights(pretrained_path)

        hidden_dim = self._infer_hidden_dim()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if mode == "linear_probe":
            self.freeze_backbone()
        elif mode == "finetune":
            self.unfreeze_backbone()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _resize_pos_embed_if_needed(self, clean_state_dict):
        if "pos_embed" not in clean_state_dict:
            return clean_state_dict

        ckpt_pos = clean_state_dict["pos_embed"]
        model_pos = self.backbone.state_dict()["pos_embed"]

        if ckpt_pos.shape == model_pos.shape:
            return clean_state_dict

        print(
            f"Resize pos_embed: checkpoint {tuple(ckpt_pos.shape)} "
            f"-> model {tuple(model_pos.shape)}"
        )

        num_prefix_tokens = 1

        ckpt_prefix = ckpt_pos[:, :num_prefix_tokens, :]
        ckpt_grid = ckpt_pos[:, num_prefix_tokens:, :]

        model_grid_len = model_pos.shape[1] - num_prefix_tokens

        old_grid_size = int(math.sqrt(ckpt_grid.shape[1]))
        new_grid_size = int(math.sqrt(model_grid_len))

        if old_grid_size * old_grid_size != ckpt_grid.shape[1]:
            raise ValueError(
                f"Cannot infer old grid size from pos_embed: {ckpt_pos.shape}"
            )

        if new_grid_size * new_grid_size != model_grid_len:
            raise ValueError(
                f"Cannot infer new grid size from model pos_embed: {model_pos.shape}"
            )

        ckpt_grid = ckpt_grid.reshape(
            1,
            old_grid_size,
            old_grid_size,
            ckpt_grid.shape[-1],
        ).permute(0, 3, 1, 2)

        ckpt_grid = F.interpolate(
            ckpt_grid,
            size=(new_grid_size, new_grid_size),
            mode="bicubic",
            align_corners=False,
        )

        ckpt_grid = ckpt_grid.permute(0, 2, 3, 1).reshape(
            1,
            new_grid_size * new_grid_size,
            ckpt_grid.shape[1],
        )

        new_pos_embed = torch.cat([ckpt_prefix, ckpt_grid], dim=1)

        if new_pos_embed.shape != model_pos.shape:
            raise ValueError(
                f"Resized pos_embed shape mismatch: "
                f"{new_pos_embed.shape} vs {model_pos.shape}"
            )

        clean_state_dict["pos_embed"] = new_pos_embed

        return clean_state_dict

    def _load_retfound_weights(self, pretrained_path: str):
        if not pretrained_path:
            print("[WARN] No pretrained_path provided. Using random backbone.")
            return

        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "teacher" in ckpt:
            state_dict = ckpt["teacher"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        clean_state_dict = {}

        for k, v in state_dict.items():
            new_k = k

            for prefix in [
                "module.",
                "backbone.",
                "teacher.",
                "student.",
                "encoder.",
            ]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]

            if new_k == "mask_token":
                continue

            if new_k.startswith("dino_head."):
                continue

            if new_k.startswith("ibot_head."):
                continue

            clean_state_dict[new_k] = v

        clean_state_dict = self._resize_pos_embed_if_needed(clean_state_dict)

        msg = self.backbone.load_state_dict(clean_state_dict, strict=False)

        print(f"Loaded RETFound-DINOv2 checkpoint from: {pretrained_path}")
        print(f"Missing keys: {len(msg.missing_keys)}")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")

        if len(msg.missing_keys) > 0:
            print("First missing keys:")
            for k in msg.missing_keys[:20]:
                print(f"  {k}")

        if len(msg.unexpected_keys) > 0:
            print("First unexpected keys:")
            for k in msg.unexpected_keys[:20]:
                print(f"  {k}")

    def _infer_hidden_dim(self):
        if hasattr(self.backbone, "num_features"):
            return self.backbone.num_features

        if hasattr(self.backbone, "embed_dim"):
            return self.backbone.embed_dim

        if hasattr(self.backbone, "norm") and hasattr(self.backbone.norm, "normalized_shape"):
            return self.backbone.norm.normalized_shape[0]

        raise ValueError("Cannot infer hidden dimension.")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images):
        feat = self.backbone(images)

        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            elif "x_prenorm" in feat:
                feat = feat["x_prenorm"][:, 0]
            else:
                raise ValueError(f"Unknown backbone output keys: {list(feat.keys())}")

        if feat.ndim == 3:
            feat = feat[:, 0]

        logits = self.classifier(feat)
        return logits