from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataTransformationConfig(StrictBaseModel):
    resize: dict | None = None
    longest_max_size: int | dict | None = None
    hflip: bool | float | dict | None = None
    affine: dict | None = None
    random_bbox_transform: dict | None = None
    crop_sampling: dict | None = None
    hist_eq: bool | dict | None = False
    motion_blur: bool | dict | None = False
    covering: bool | dict | None = None
    elastic_transform: bool | dict | None = None
    grayscale: bool | dict | None = None
    gaussian_noise: bool | float | int | dict | None = None
    auto_padding: dict | None = None
    normalize_images: bool | dict | None = True
    scale_to_unit_range: bool | dict | None = False
    top_down_crop: dict | None = None
    collate: dict | None = None


class DataConfig(StrictBaseModel):
    bbox_margin: int = 25
    colormode: str = "RGB"
    inference: DataTransformationConfig | None = None
    train: DataTransformationConfig | None = None


class OptimizerConfig(StrictBaseModel):
    type: str = ""
    params: dict[str, Any] | None = None


class SchedulerConfig(StrictBaseModel):
    type: str = ""
    params: dict[str, Any] | None = None


class SnapshotCheckpointConfig(StrictBaseModel):
    max_snapshots: int = 5
    save_epochs: int = 25
    save_optimizer_state: bool = False


class RunnerConfig(StrictBaseModel):
    type: str = "PoseTrainingRunner"
    gpus: Any | None = None
    device: str = "auto"
    key_metric: str = "test.mAP"
    key_metric_asc: bool = True
    eval_interval: int = 10
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | None = None
    snapshots: SnapshotCheckpointConfig | None = None
    resume_training_from: str | None = None
    load_weights_only: bool | None = None
