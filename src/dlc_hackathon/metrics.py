from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from deeplabcut.core.metrics.api import compute_metrics as dlc_compute_metrics

from dlc_hackathon.schemas.types import (
    BBoxEntry,
    BBoxEvalMetrics,
    PoseEstimationEvalMetrics,
    PosePredictionEntry,
)


@dataclass
class _ImagePair:
    gt: BBoxEntry
    pred: BBoxEntry


def _iou_matrix_xyxy(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix for boxes in xyxy format."""
    if gt.size == 0 or pred.size == 0:
        return np.zeros((gt.shape[0], pred.shape[0]), dtype=np.float32)

    # gt: [G, 4], pred: [P, 4]
    gt_x1, gt_y1, gt_x2, gt_y2 = gt[:, 0][:, None], gt[:, 1][:, None], gt[:, 2][:, None], gt[:, 3][:, None]
    pr_x1, pr_y1, pr_x2, pr_y2 = pred[:, 0][None, :], pred[:, 1][None, :], pred[:, 2][None, :], pred[:, 3][None, :]

    inter_x1 = np.maximum(gt_x1, pr_x1)
    inter_y1 = np.maximum(gt_y1, pr_y1)
    inter_x2 = np.minimum(gt_x2, pr_x2)
    inter_y2 = np.minimum(gt_y2, pr_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    gt_area = np.maximum(0.0, (gt_x2 - gt_x1) * (gt_y2 - gt_y1))
    pr_area = np.maximum(0.0, (pr_x2 - pr_x1) * (pr_y2 - pr_y1))
    union = gt_area + pr_area - inter

    out = np.zeros_like(inter, dtype=np.float32)
    valid = union > 0
    out[valid] = inter[valid] / union[valid]
    return out


def _match_detections(
    gt_xyxy: np.ndarray,
    pred_xyxy: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Greedy score-ordered matching for one image.

    Returns:
      tp_flags: [num_pred] bool in score-sorted order
      sorted_scores: [num_pred] float in descending order
      num_gt: int
      sum_iou_for_true_positives: float
    """
    num_gt = gt_xyxy.shape[0]
    num_pred = pred_xyxy.shape[0]

    if num_pred == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float32), num_gt, 0.0

    order = np.argsort(-pred_scores)
    pred_xyxy = pred_xyxy[order]
    sorted_scores = pred_scores[order]

    ious = _iou_matrix_xyxy(gt_xyxy, pred_xyxy)  # [G, P]
    gt_used = np.zeros(num_gt, dtype=bool)
    tp_flags = np.zeros(num_pred, dtype=bool)

    tp_iou_sum = 0.0
    for j in range(num_pred):
        if num_gt == 0:
            break
        iou_col = ious[:, j]
        # mask out used GT
        iou_col = np.where(gt_used, -1.0, iou_col)
        best_gt = int(np.argmax(iou_col))
        best_iou = float(iou_col[best_gt])

        if best_iou >= iou_threshold:
            tp_flags[j] = True
            gt_used[best_gt] = True
            tp_iou_sum += best_iou

    return tp_flags, sorted_scores, num_gt, tp_iou_sum


def _average_precision(tp_flags: np.ndarray, scores: np.ndarray, num_gt: int) -> float:
    """Compute AP from ranked detections (continuous integral PR-AUC)."""
    if num_gt == 0:
        return 0.0
    if tp_flags.size == 0:
        return 0.0

    order = np.argsort(-scores)
    tp = tp_flags[order].astype(np.float32)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / num_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # Precision envelope
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def _align_image_pairs(
    ground_truth_bboxes: Sequence[BBoxEntry],
    predicted_bboxes: Sequence[BBoxEntry],
) -> list[_ImagePair]:
    """
    Align GT and predictions by image_path when available, otherwise by index.
    """
    gt_have_paths = all(x.image_path is not None for x in ground_truth_bboxes)
    pr_have_paths = all(x.image_path is not None for x in predicted_bboxes)

    if gt_have_paths and pr_have_paths:
        gt_map = {str(x.image_path): x for x in ground_truth_bboxes}
        pr_map = {str(x.image_path): x for x in predicted_bboxes}
        if set(gt_map) != set(pr_map):
            missing_pred = sorted(set(gt_map) - set(pr_map))
            missing_gt = sorted(set(pr_map) - set(gt_map))
            raise ValueError(
                f"GT/pred image keys do not match. Missing in predictions: {missing_pred}. Missing in GT: {missing_gt}."
            )
        return [_ImagePair(gt=gt_map[k], pred=pr_map[k]) for k in sorted(gt_map)]

    if len(ground_truth_bboxes) != len(predicted_bboxes):
        raise ValueError("When image_path is missing, GT and predictions must have equal lengths to align by index.")
    return [_ImagePair(gt=g, pred=p) for g, p in zip(ground_truth_bboxes, predicted_bboxes, strict=False)]


def calculate_bbox_metrics(
    ground_truth_bboxes: list[BBoxEntry],
    predicted_bboxes: list[BBoxEntry],
) -> BBoxEvalMetrics:
    """
    Compute detector bbox metrics without DLC/COCO helper.
    Returns AP/AR at IoU=.50:.05:.95 plus @0.50 and @0.75.
    """
    pairs = _align_image_pairs(ground_truth_bboxes, predicted_bboxes)
    thresholds = [round(float(x), 2) for x in np.arange(0.50, 0.96, 0.05)]

    ap_by_thr: dict[float, float] = {}
    ar_by_thr: dict[float, float] = {}
    mean_iou_50_num = 0.0
    mean_iou_50_den = 0

    num_gt_total = int(sum(len(p.gt.bboxes) for p in pairs))
    num_pred_total = int(sum(len(p.pred.bboxes) for p in pairs))

    for thr in thresholds:
        all_tp_flags: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        total_gt = 0
        total_tp = 0

        for pair in pairs:
            gt_xyxy = pair.gt.to_xyxy(dtype=np.float32)
            pred_xyxy = pair.pred.to_xyxy(dtype=np.float32)
            pred_scores = np.asarray(pair.pred.bbox_scores, dtype=np.float32)

            tp_flags, sorted_scores, num_gt, tp_iou_sum = _match_detections(
                gt_xyxy=gt_xyxy,
                pred_xyxy=pred_xyxy,
                pred_scores=pred_scores,
                iou_threshold=thr,
            )

            all_tp_flags.append(tp_flags)
            all_scores.append(sorted_scores)
            total_gt += num_gt
            total_tp += int(tp_flags.sum())

            if thr == 0.50:
                mean_iou_50_num += tp_iou_sum
                mean_iou_50_den += int(tp_flags.sum())

        tp_concat = np.concatenate(all_tp_flags) if all_tp_flags else np.zeros((0,), dtype=bool)
        sc_concat = np.concatenate(all_scores) if all_scores else np.zeros((0,), dtype=np.float32)

        ap_by_thr[thr] = _average_precision(tp_concat, sc_concat, total_gt)
        ar_by_thr[thr] = float(total_tp / total_gt) if total_gt > 0 else 0.0

    mAP_50_95 = float(np.mean(list(ap_by_thr.values()))) * 100.0
    mAR_50_95 = float(np.mean(list(ar_by_thr.values()))) * 100.0
    mAP_50 = ap_by_thr[0.5] * 100.0
    mAP_75 = ap_by_thr[0.75] * 100.0
    mAR_50 = ar_by_thr[0.5] * 100.0
    mAR_75 = ar_by_thr[0.75] * 100.0
    mean_iou_matched_50 = (mean_iou_50_num / mean_iou_50_den) if mean_iou_50_den > 0 else 0.0

    return {
        "mAP_50_95": mAP_50_95,
        "mAP_50": mAP_50,
        "mAP_75": mAP_75,
        "mAR_50_95": mAR_50_95,
        "mAR_50": mAR_50,
        "mAR_75": mAR_75,
        "mean_iou_matched_50": float(mean_iou_matched_50),
        "num_images": len(pairs),
        "num_gt_boxes": num_gt_total,
        "num_pred_boxes": num_pred_total,
    }


def calculate_pose_estimation_metrics(
    ground_truth_keypoints: list[PosePredictionEntry],
    predicted_keypoints: list[PosePredictionEntry],
) -> PoseEstimationEvalMetrics:
    """Calculate pose-estimation metrics using paired-by-order entries."""
    if len(ground_truth_keypoints) != len(predicted_keypoints):
        raise ValueError("Ground-truth and predicted keypoint lists must have equal lengths.")

    gt_dict: dict[str, np.ndarray] = {}
    pred_dict: dict[str, np.ndarray] = {}
    for idx, (gt_entry, pred_entry) in enumerate(zip(ground_truth_keypoints, predicted_keypoints, strict=False)):
        key = str(idx)
        gt_dict[key] = gt_entry.to_pose_array()
        pred_dict[key] = pred_entry.to_pose_array()

    raw_metrics = dlc_compute_metrics(
        ground_truth=gt_dict,
        predictions=pred_dict,
        single_animal=False,
        pcutoff=0.6,
    )
    return cast(PoseEstimationEvalMetrics, raw_metrics)
