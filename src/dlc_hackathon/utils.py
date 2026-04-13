"""Helpers for locating and updating DeepLabCut PyTorch train configs."""

from pathlib import Path
from typing import Any

import numpy as np
import deeplabcut.pose_estimation_pytorch as dlc_torch
from deeplabcut.core.engine import Engine
from deeplabcut.utils import auxiliaryfunctions


def get_pytorch_config_path(
    project_cfg_path: str | Path,
    *,
    shuffle: int,
    trainsetindex: int,
) -> Path:
    """Absolute path to ``train/pytorch_config.yaml`` for a shuffle."""
    project_cfg = dlc_torch.config.read_config_as_dict(project_cfg_path)
    train_fraction = project_cfg["TrainingFraction"][trainsetindex]
    rel = auxiliaryfunctions.get_model_folder(
        train_fraction,
        shuffle,
        project_cfg,
        engine=Engine.PYTORCH,
    )
    return Path(project_cfg_path).parent / rel / "train" / Engine.PYTORCH.pose_cfg_name


def update_pytorch_config_file(
    pytorch_config_path: Path | str,
    updates: dict[str, Any] = None,
) -> None:
    """Merge ``updates`` into ``pytorch_config.yaml`` and write it back."""
    if updates is None:
        updates = {}
    if not updates:
        return
    cfg = dlc_torch.config.read_config_as_dict(pytorch_config_path)
    merged = dlc_torch.config.update_config(cfg, updates)
    dlc_torch.config.write_config(pytorch_config_path, merged)


def to_jsonable(data: Any) -> Any:
    """Recursively convert nested structures with numpy values to JSON-serializable Python values."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, dict):
        return {k: to_jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_jsonable(item) for item in data]
    return data