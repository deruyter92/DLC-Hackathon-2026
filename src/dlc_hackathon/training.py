"""Training helpers for DeepLabCut experiments."""

import logging
from pathlib import Path

import deeplabcut
import deeplabcut.pose_estimation_pytorch as dlc_torch
from deeplabcut.core.engine import Engine

from dlc_hackathon.schemas.benchmarking import BenchMarkTrainConfig, ModelType
from dlc_hackathon.utils import get_pytorch_config_path, update_pytorch_config_file

logger = logging.getLogger(__name__)


def prepare_weight_init():
    raise NotImplementedError("Weight initialization is not implemented yet")


def prepare_train_shuffle(config: BenchMarkTrainConfig) -> Path:
    """Create training shuffle(s) and update ``pytorch_config.yaml`` immediately."""

    pytorch_config_path = get_pytorch_config_path(
        project_cfg_path=config.dataset.project_config_path,
        shuffle=config.model.shuffle,
        trainsetindex=config.model.trainsetindex,
    )
    if pytorch_config_path.exists():
        logger.info(
            "Shuffle %s already exists at %s; skipping shuffle creation.",
            config.model.shuffle,
            pytorch_config_path,
        )
        return pytorch_config_path

    if config.model.type == ModelType.POSE_ESTIMATION:
        # Create placeholder for detector type; we are only training
        # the pose estimation models
        detector_net_type = "fasterrcnn_mobilenet_v3_large_fpn"
        # pose_estimation_net_type = f"top_down_{config.model.net_type}"
        pose_estimation_net_type = config.model.net_type
    elif config.model.type == ModelType.DETECTION:
        # Create placeholder for pose estimation net type; we are only training
        # the detection models
        pose_estimation_net_type = "top_down_resnet_50"
        detector_net_type = config.model.net_type
    else:
        raise ValueError(f"Unsupported model type: {config.model.type!r}")

    deeplabcut.create_training_dataset_from_existing_split(
        config=str(config.dataset.project_config_path),
        from_shuffle=config.dataset.shuffle,
        from_trainsetindex=config.dataset.trainsetindex,
        shuffles=[config.model.shuffle],
        net_type=pose_estimation_net_type,
        detector_type=detector_net_type,
        engine=Engine.PYTORCH,
    )

    updates = config.overrides.to_dict() if config.overrides is not None else {}
    updates["train_settings"] = config.train_settings.to_dict()
    # For detection models, the updates are nested under the "detector" key
    if config.model.type == ModelType.DETECTION:
        updates = {"detector": updates}
    updates["logger"] = config.logger.to_dict()
    update_pytorch_config_file(pytorch_config_path, updates)
    return pytorch_config_path


def get_loader(config: BenchMarkTrainConfig) -> dlc_torch.DLCLoader:
    loader = dlc_torch.DLCLoader(
        config=config.dataset.project_config_path,
        shuffle=config.model.shuffle,
        trainset_index=config.model.trainsetindex,
    )
    return loader


def train_model(config: BenchMarkTrainConfig, device: str) -> None:
    loader = get_loader(config)

    # The pythorch_config.yaml is the training run config for pose estimation.
    # For detection models, the run config is nested under the "detector" key.
    run_config = loader.model_cfg
    if config.model.type == ModelType.DETECTION:
        run_config = run_config["detector"]

    # fix seed for reproducibility
    dlc_torch.utils.fix_seeds(config.train_settings.seed)

    dlc_torch.train(
        loader=loader,
        run_config=run_config,
        task=config.model.type.dlc_task,
        device=device,
        # logger_config=config.logger.to_dict(),
        # snapshot_path=config.train_settings.snapshot_path,
        # max_snapshots_to_keep=config.train_settings.max_snapshots_to_keep,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/example_train_pose.yaml")
    config = BenchMarkTrainConfig.from_yaml(config_path)
    prepare_train_shuffle(config)
    train_model(config, "cuda")
