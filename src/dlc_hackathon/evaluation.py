from pathlib import Path

import numpy as np
from collections import defaultdict
import deeplabcut.core.metrics as metrics
import deeplabcut.pose_estimation_pytorch as dlc_torch

from dlc_hackathon.schemas.benchmarking import BenchMarkEvalConfig
from dlc_hackathon.schemas.types import (
    BBoxEntry,
    BBoxes,
    PosePredictionEntry,
    PosePredictions,
    BBoxEvalMetrics,
    PoseEstimationEvalMetrics,
    BBoxTrainTestMetrics,
    PoseTrainTestMetrics,
)
from dlc_hackathon.metrics import calculate_bbox_metrics, calculate_pose_estimation_metrics


def get_ground_truth_bboxes(loader: dlc_torch.DLCLoader) -> BBoxes:
    coco_dict = dlc_torch.DLCLoader.to_coco(
        project_root=loader.project_path,
        df=loader.df,
        parameters=loader.get_dataset_parameters(),
    )
    annotations = coco_dict['annotations']
    images = coco_dict['images']
    mode_by_fn = {fn: "train" for fn in loader.image_filenames("train")}
    mode_by_fn.update({fn: "test" for fn in loader.image_filenames("test")})
    anns_by_image = defaultdict(list)
    for an in annotations:
        anns_by_image[an["image_id"]].append(an)
    
    bboxes_by_mode = defaultdict(list)
    for img in images:
        img_annotations = anns_by_image[img["id"]]
        mode = mode_by_fn[img["file_name"]]
        bboxes_per_img = []
        for annotation in img_annotations:
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            bboxes_per_img.append(bbox)
        bboxes_by_mode[mode].append(BBoxEntry(
            bboxes=bboxes_per_img,
            bbox_scores=[1.0 for _ in range(len(bboxes_per_img))],
            bbox_format="xywh",
            image_path=Path(img["file_name"]),
        ))
    return BBoxes(**bboxes_by_mode)


def get_ground_truth_poses(loader: dlc_torch.DLCLoader) -> PosePredictions:
    """Get ground-truth keypoints from a DLCLoader as PosePredictions."""
    has_unique = len(loader.model_cfg["metadata"]["unique_bodyparts"]) >= 1

    result: dict[str, list[PosePredictionEntry]] = {}
    for mode in ("train", "test"):
        gt_dict = loader.ground_truth_keypoints(mode)
        gt_unique_dict = (
            loader.ground_truth_keypoints(mode, unique_bodypart=True)
            if has_unique
            else {}
        )

        entries: list[PosePredictionEntry] = []
        for image_path, keypoints_array in gt_dict.items():
            if keypoints_array.ndim == 2:
                keypoints_array = keypoints_array[None, ...]
            xy = keypoints_array[..., :2].tolist()
            visibility = keypoints_array[..., 2].tolist()

            unique_xy = None
            unique_scores = None
            if image_path in gt_unique_dict:
                ukps = gt_unique_dict[image_path]
                if ukps.ndim == 3:
                    ukps = ukps[0]
                unique_xy = ukps[..., :2].tolist()
                unique_scores = ukps[..., 2].tolist()

            entries.append(PosePredictionEntry(
                keypoints=xy,
                keypoint_scores=visibility,
                unique_keypoints=unique_xy,
                unique_keypoint_scores=unique_scores,
                image_path=Path(image_path),
            ))
        result[mode] = entries
    return PosePredictions(**result)


def get_predicted_bboxes(loader: dlc_torch.DLCLoader, device: str = "auto") -> BBoxes:
    """Gets the context for top-down pose estimation models"""
    # If bboxes are already cached in jsonfile, return them
    json_file = loader.evaluation_folder / "predicted_bboxes.json"
    bboxes = BBoxes.from_file_if_exists(json_file)
    if bboxes and bboxes.test:
        return bboxes

    det_snapshot = dlc_torch.apis.utils.get_model_snapshots(
        "best",
        loader.model_folder,
        dlc_torch.Task.DETECT,
    )[0]

    runner = dlc_torch.apis.utils.get_detector_inference_runner(loader.model_cfg, det_snapshot.path, device=device)
    bboxes_by_mode = {}
    for mode in ["train", "test"]:
        filenames = loader.image_filenames(mode)
        predictions = runner.inference(filenames)
        bboxes_by_mode[mode] = [
            BBoxEntry(
                bboxes=pred['bboxes'].tolist(),
                bbox_scores=pred['bbox_scores'].tolist(),
                bbox_format="xywh",
                image_path=Path(fn),
            )
            for fn, pred in zip(filenames, predictions)
        ]
    bboxes = BBoxes(**bboxes_by_mode)
    bboxes.dump_json(json_file)
    return bboxes


def get_predicted_poses(
    loader: dlc_torch.DLCLoader,
    bboxes: BBoxes | None = None,
    device: str = "auto",
) -> PosePredictions:
    """Run pose inference and return predictions as PosePredictions."""
    if loader.pose_task == dlc_torch.Task.TOP_DOWN and bboxes is None:
        raise ValueError("Bboxes are required for top-down pose estimation")

    snapshot = dlc_torch.apis.utils.get_model_snapshots(
        "best", loader.model_folder, loader.pose_task,
    )[0]
    runner = dlc_torch.apis.utils.get_pose_inference_runner(
        loader.model_cfg,
        snapshot.path, 
        device=device,
    )

    has_unique = len(loader.model_cfg["metadata"]["unique_bodyparts"]) >= 1

    result: dict[str, list[PosePredictionEntry]] = {}
    for mode in ("train", "test"):
        filenames = loader.image_filenames(mode)
        if bboxes is not None:
            # Top-down pose estimation using bboxes
            images_with_ctx = bboxes.to_images_with_context(filenames, mode)
            raw_predictions = runner.inference(images=images_with_ctx)
        else:
            # Bottom-up pose estimation
            raw_predictions = runner.inference(filenames)

        entries: list[PosePredictionEntry] = []
        for fn, pred in zip(filenames, raw_predictions):
            kps = pred["bodyparts"]
            if kps.ndim == 2:
                kps = kps[None, ...]

            unique_xy = None
            unique_scores = None
            if has_unique and "unique_bodyparts" in pred:
                ukps = pred["unique_bodyparts"]
                if ukps.ndim == 3:
                    ukps = ukps[0]
                unique_xy = ukps[..., :2].tolist()
                unique_scores = ukps[..., 2].tolist()

            entries.append(PosePredictionEntry(
                keypoints=kps[..., :2].tolist(),
                keypoint_scores=kps[..., 2].tolist(),
                unique_keypoints=unique_xy,
                unique_keypoint_scores=unique_scores,
                image_path=Path(fn),
            ))
        result[mode] = entries

    return PosePredictions(**result)


def evaluate_detector(config: BenchMarkEvalConfig, device: str) -> tuple[BBoxes, BBoxTrainTestMetrics]:
    """Evaluates the best trained model. Uploads results to WandB"""

    detector_loader = dlc_torch.DLCLoader(
        config=config.dataset.project_config_path,
        shuffle=config.model.shuffle,
        trainset_index=config.model.trainsetindex,
    )

    bboxes: BBoxes = get_predicted_bboxes(detector_loader, device)
    gt_bboxes: BBoxes = get_ground_truth_bboxes(detector_loader)

    test_metrics: BBoxEvalMetrics = calculate_bbox_metrics(gt_bboxes.test, bboxes.test)
    train_metrics: BBoxEvalMetrics = calculate_bbox_metrics(gt_bboxes.train, bboxes.train)

    return bboxes, {
        "test": test_metrics,
        "train": train_metrics,
    }


def evaluate_pose_estimation(
    config: BenchMarkEvalConfig,
    bboxes: BBoxes | None = None,
    device: str = "auto",
) -> PoseTrainTestMetrics:
    """Evaluates the pose estimation model"""
    pose_estimation_loader = dlc_torch.DLCLoader(
        config=config.dataset.project_config_path,
        shuffle=config.model.shuffle,
        trainset_index=config.model.trainsetindex,
    )

    if bboxes is None and pose_estimation_loader.pose_task == dlc_torch.Task.TOP_DOWN:
        bboxes = get_ground_truth_bboxes(pose_estimation_loader)

    pose_predictions: PosePredictions = get_predicted_poses(pose_estimation_loader, bboxes, device)
    pose_gt: PosePredictions = get_ground_truth_poses(pose_estimation_loader)

    test_metrics: PoseEstimationEvalMetrics = calculate_pose_estimation_metrics(pose_gt.test, pose_predictions.test)
    train_metrics: PoseEstimationEvalMetrics = calculate_pose_estimation_metrics(pose_gt.train, pose_predictions.train)
    return {
        "test": test_metrics,
        "train": train_metrics,
    }


if __name__ == "__main__":
    config = BenchMarkEvalConfig.from_yaml(Path("configs/example_eval.yaml"))
    pose_metrics = evaluate_pose_estimation(config, device="cuda")
    print(pose_metrics)
