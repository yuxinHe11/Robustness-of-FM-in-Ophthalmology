import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Classifier(nn.Module):
    """
    DINOv3 backbone + linear classification head.

    mode:
        linear_probe: freeze backbone, only train classifier
        finetune: train backbone + classifier
    """

    def __init__(
        self,
        model_dir: str,
        num_classes: int,
        mode: str = "finetune",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
        )

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "hidden_dim", None)

        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from DINOv3 config.")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        if mode == "linear_probe":
            self.freeze_backbone()
        elif mode == "finetune":
            self.unfreeze_backbone()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)

        # HF官方示例使用 pooler_output。
        # 为了兼容不同实现，如果没有 pooler_output，就取 CLS token。
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        else:
            feat = outputs.last_hidden_state[:, 0]

        logits = self.classifier(feat)

        return logits