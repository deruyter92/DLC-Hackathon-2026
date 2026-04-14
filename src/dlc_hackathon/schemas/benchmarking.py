from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from dlc_hackathon.schemas.dlc_overrides import DataConfig, RunnerConfig

if TYPE_CHECKING:
    from deeplabcut.pose_estimation_pytorch.task import Task  # type: ignore


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class DataSubsetConfig(StrictBaseModel):
    name: str
    root: Path
    shuffle: int
    trainsetindex: int

    @property
    def project_config_path(self) -> Path:
        """Path to the DeepLabCut project config corresponding to this dataset."""
        return self.root / "config.yaml"


class ModelType(Enum):
    DETECTION = "detection"
    POSE_ESTIMATION = "pose_estimation"

    @property
    def dlc_task(self) -> "Task":  # type: ignore
        """Lazy import of ``Task`` from DLC."""
        from deeplabcut.pose_estimation_pytorch.task import Task

        return {
            ModelType.POSE_ESTIMATION: Task.TOP_DOWN,
            ModelType.DETECTION: Task.DETECT,
        }[self]


class ModelConfig(StrictBaseModel):
    type: ModelType
    name: str
    net_type: str
    shuffle: int
    trainsetindex: int


class WandbLoggerConfig(StrictBaseModel):
    type: Literal["WandbLogger"] = "WandbLogger"
    project_name: str
    run_name: str
    group: str | None = None
    tags: tuple[str] | None = None
    notes: str | None = None

    @classmethod
    def from_model_and_dataset(
        cls,
        model_cfg: ModelConfig,
        dataset_cfg: DataSubsetConfig,
        project_name: str = "DLC-Hackathon-2026",
        **kwargs,
    ) -> "WandbLoggerConfig":
        return cls(
            project_name=project_name,
            run_name=f"{dataset_cfg.name}-{model_cfg.name}-{model_cfg.shuffle}",
            **kwargs,
        )


class TrainSettingsConfig(StrictBaseModel):
    batch_size: int = 4
    epochs: int = 300
    seed: int = 42
    dataloader_workers: int = 8
    dataloader_pin_memory: bool = False
    display_iters: int = 1000


class DLCOverridesConfig(StrictBaseModel):
    data: DataConfig
    runner: RunnerConfig

    def to_dict(self) -> dict[str, Any]:
        """Overrides dict should omit None values to use the DLC defaults."""
        return {
            "data": self.data.model_dump(exclude_none=True),
            "runner": self.runner.model_dump(exclude_none=True),
        }


class BenchMarkTrainConfig(StrictBaseModel):
    model: ModelConfig
    dataset: DataSubsetConfig
    logger: WandbLoggerConfig
    train_settings: TrainSettingsConfig
    overrides: DLCOverridesConfig | None = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "BenchMarkTrainConfig":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, yaml_path: Path) -> None:
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f)


class EvalSettingsConfig(StrictBaseModel):
    batch_size: int = 4
    dataloader_workers: int = 8
    dataloader_pin_memory: bool = False
    display_iters: int = 1000


class BenchMarkEvalConfig(StrictBaseModel):
    model: ModelConfig
    dataset: DataSubsetConfig
    eval_settings: EvalSettingsConfig

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "BenchMarkEvalConfig":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, yaml_path: Path) -> None:
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f)
