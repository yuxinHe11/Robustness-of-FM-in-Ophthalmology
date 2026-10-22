import os
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_score,
    args,
):
    ensure_dir(os.path.dirname(path))

    ckpt = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(ckpt, path)


def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return ckpt


def load_checkpoint_dinov2(path, model, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return ckpt

def load_dinov3_checkpoint_flexible(path, model, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model_state = model.state_dict()
    new_state = {}

    for k, v in state.items():
        nk = k

        # 旧checkpoint: backbone.layer.x
        # 新模型: backbone.model.layer.x
        if nk.startswith("backbone.layer."):
            nk = nk.replace("backbone.layer.", "backbone.model.layer.", 1)

        if nk.startswith("backbone.embeddings."):
            nk = nk.replace("backbone.embeddings.", "backbone.model.embeddings.", 1)

        if nk.startswith("backbone.norm."):
            nk = nk.replace("backbone.norm.", "backbone.model.norm.", 1)

        if nk.startswith("backbone.pooler."):
            nk = nk.replace("backbone.pooler.", "backbone.model.pooler.", 1)

        new_state[nk] = v

    msg = model.load_state_dict(new_state, strict=False)

    print(f"Loaded checkpoint: {path}")
    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    # 如果只有classifier或者少量pooler不匹配，一般不影响embedding
    if len(msg.missing_keys) > 0:
        print("First missing keys:")
        for k in msg.missing_keys[:20]:
            print(" ", k)

    if len(msg.unexpected_keys) > 0:
        print("First unexpected keys:")
        for k in msg.unexpected_keys[:20]:
            print(" ", k)

    return ckpt