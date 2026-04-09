from pathlib import Path

import deeplabcut.pose_estimation_pytorch as dlc_torch

from dlc_hackathon.schemas.benchmarking import BenchMarkEvalConfig
from dlc_hackathon.schemas.types import (
    BBoxes,
    EvalMetrics,
)


def get_ground_truth_bboxes(loader: dlc_torch.DLCLoader) -> BBoxes:
    raise NotImplementedError("Not implemented yet")
    return BBoxes(train=[], test=[])


def get_predicted_bboxes(loader: dlc_torch.DLCLoader, device: str = "auto") -> BBoxes:
    """Gets the context for top-down pose estimation models"""
    # If bboxes are already cached in jsonfile, return them
    json_file = loader.evaluation_folder / "predicted_bboxes.json"
    bboxes = BBoxes.from_file(
        json_file,
        missing_ok=True,
    )
    if bboxes and bboxes.test:
        return bboxes

    det_snapshot = dlc_torch.apis.utils.get_model_snapshots(
        "best",
        loader.model_folder,
        dlc_torch.Task.DETECT,
    )[0]

    runner = dlc_torch.apis.utils.get_detector_inference_runner(loader.model_cfg, det_snapshot.path, device=device)
    bboxes = BBoxes(train=[], test=[])
    bboxes.test = runner.inference(loader.image_filenames("test"))
    bboxes.dump_json(json_file)
    bboxes.train = runner.inference(loader.image_filenames("train"))
    bboxes.dump_json(json_file)
    return bboxes


def evaluate_detector(config: BenchMarkEvalConfig, device: str) -> tuple[BBoxes, EvalMetrics]:
    """Evaluates the best trained model. Uploads results to WandB"""

    detector_loader = dlc_torch.DLCLoader(
        config=config.dataset.project_config_path,
        shuffle=config.detector_model.shuffle,
        trainset_index=config.detector_model.trainsetindex,
    )

    bboxes = get_predicted_bboxes(detector_loader, device)

    # TODO: compute metrics
    # metrics = compute_metrics(...)
    metrics = None

    return bboxes, metrics


def evaluate_pose_estimation(
    config: BenchMarkEvalConfig,
    bboxes: BBoxes | None = None,
    device: str = "auto",
) -> None:
    """Evaluates the pose estimation model"""
    if config.pose_estimation_model is None:
        raise ValueError("Pose estimation model is not set")

    pose_estimation_loader = dlc_torch.DLCLoader(
        config=config.dataset.project_config_path,
        shuffle=config.pose_estimation_model.shuffle,
        trainset_index=config.pose_estimation_model.trainsetindex,
    )

    if bboxes is None:
        bboxes = get_ground_truth_bboxes(pose_estimation_loader)

    pose_estimation_snapshot = dlc_torch.apis.utils.get_model_snapshots(
        "best", pose_estimation_loader.model_folder, config.pose_estimation_model.type.dlc_task
    )[0]
    runner = dlc_torch.utils.get_pose_inference_runner(pose_estimation_loader.model_cfg, pose_estimation_snapshot.path)
    for mode in ["train", "test"]:
        images = pose_estimation_loader.image_filenames(mode)
        mode_images_with_context = bboxes.to_images_with_context(images, mode)
        predictions = runner.inference(images=mode_images_with_context)
        # TODO: collect and score predictions
        _ = predictions

    # TODO compute metrics

    return None


def evaluate(config: BenchMarkEvalConfig, device: str = "auto") -> None:
    bboxes, _ = evaluate_detector(config, device=device)
    if config.pose_estimation_model is not None:
        evaluate_pose_estimation(config, bboxes=bboxes, device=device)


if __name__ == "__main__":
    config = BenchMarkEvalConfig.from_yaml(Path("configs/example_eval.yaml"))
    evaluate(config, device="cuda")
