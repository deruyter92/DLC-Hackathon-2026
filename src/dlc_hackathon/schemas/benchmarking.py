import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from dlc_hackathon.paths import REPO_ROOT
from dlc_hackathon.schemas.dlc_overrides import DataConfig, RunnerConfig
from dlc_hackathon.schemas.types import DETECTOR_NET_TYPES, TOP_DOWN_NET_TYPES, NetType

if TYPE_CHECKING:
    from deeplabcut.pose_estimation_pytorch.task import Task  # type: ignore

logger = logging.getLogger(__name__)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class DataSubsetConfig(StrictBaseModel):
    name: str
    root: Path
    shuffle: int
    trainsetindex: int

    @model_validator(mode="after")
    def _resolve_root(self) -> "DataSubsetConfig":
        """Resolve relative root paths to absolute so DLC doesn't choke on them."""
        if not self.root.is_absolute():
            try:
                resolved = self.root.resolve()
                if resolved.exists():
                    self.root = resolved
                elif (REPO_ROOT / self.root).exists():
                    self.root = (REPO_ROOT / self.root).resolve()
                else:
                    logger.warning(
                        "Could not resolve relative root path '%s' from CWD or repo root.",
                        self.root,
                    )
            except OSError:
                logger.warning("Failed to resolve relative root path '%s'.", self.root)
        return self

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
    net_type: NetType
    shuffle: int
    trainsetindex: int

    @model_validator(mode="after")
    def _validate_net_type_for_model_type(self) -> "ModelConfig":
        if self.type is ModelType.DETECTION and not self.net_type.is_detector:
            raise ValueError(
                f"Invalid detector net_type '{self.net_type.value}'. Must be one of: {sorted(DETECTOR_NET_TYPES)}"
            )

        if self.type is ModelType.POSE_ESTIMATION and not self.net_type.is_top_down:
            raise ValueError(
                f"Invalid pose net_type '{self.net_type.value}'. Must be one of: {sorted(TOP_DOWN_NET_TYPES)}"
            )

        return self


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
