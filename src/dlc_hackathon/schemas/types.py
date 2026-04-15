from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class NetType(str, Enum):
    # Detector net types (DeepLabCut cy/h-detectors)
    FASTERRCNN_MOBILENET_V3_LARGE_FPN = "fasterrcnn_mobilenet_v3_large_fpn"
    FASTERRCNN_RESNET50_FPN_V2 = "fasterrcnn_resnet50_fpn_v2"
    SSDLITE = "ssdlite"

    # Top-down net types (DeepLabCut cy/h-detectors)
    ANIMALTOKENPOSE_BASE = "animaltokenpose_base"
    TOP_DOWN_ANIMALTOKENPOSE_BASE = "top_down_animaltokenpose_base"
    RTMPOSE_M = "rtmpose_m"
    TOP_DOWN_RTMPOSE_M = "top_down_rtmpose_m"
    RTMPOSE_S = "rtmpose_s"
    TOP_DOWN_RTMPOSE_S = "top_down_rtmpose_s"
    RTMPOSE_X = "rtmpose_x"
    TOP_DOWN_RTMPOSE_X = "top_down_rtmpose_x"
    TOP_DOWN_CSPNEXT_M = "top_down_cspnext_m"
    TOP_DOWN_CSPNEXT_S = "top_down_cspnext_s"
    TOP_DOWN_CSPNEXT_X = "top_down_cspnext_x"
    TOP_DOWN_HRNET_W18 = "top_down_hrnet_w18"
    TOP_DOWN_HRNET_W32 = "top_down_hrnet_w32"
    TOP_DOWN_HRNET_W48 = "top_down_hrnet_w48"
    TOP_DOWN_RESNET_101 = "top_down_resnet_101"
    TOP_DOWN_RESNET_50 = "top_down_resnet_50"

    @property
    def is_detector(self) -> bool:
        return self.value in DETECTOR_NET_TYPES

    @property
    def is_top_down(self) -> bool:
        return self.value in TOP_DOWN_NET_TYPES


DETECTOR_NET_TYPES = {
    NetType.FASTERRCNN_MOBILENET_V3_LARGE_FPN.value,
    NetType.FASTERRCNN_RESNET50_FPN_V2.value,
    NetType.SSDLITE.value,
}

TOP_DOWN_NET_TYPES = {
    NetType.ANIMALTOKENPOSE_BASE.value,
    NetType.TOP_DOWN_ANIMALTOKENPOSE_BASE.value,
    NetType.RTMPOSE_M.value,
    NetType.TOP_DOWN_RTMPOSE_M.value,
    NetType.RTMPOSE_S.value,
    NetType.TOP_DOWN_RTMPOSE_S.value,
    NetType.RTMPOSE_X.value,
    NetType.TOP_DOWN_RTMPOSE_X.value,
    NetType.TOP_DOWN_CSPNEXT_M.value,
    NetType.TOP_DOWN_CSPNEXT_S.value,
    NetType.TOP_DOWN_CSPNEXT_X.value,
    NetType.TOP_DOWN_HRNET_W18.value,
    NetType.TOP_DOWN_HRNET_W32.value,
    NetType.TOP_DOWN_HRNET_W48.value,
    NetType.TOP_DOWN_RESNET_101.value,
    NetType.TOP_DOWN_RESNET_50.value,
}


def _numpy_to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_jsonable(x) for x in obj]
    return obj


class BBoxEvalMetrics(TypedDict):
    mAP_50_95: float
    mAP_50: float
    mAP_75: float
    mAR_50_95: float
    mAR_50: float
    mAR_75: float
    mean_iou_matched_50: float
    num_images: int
    num_gt_boxes: int
    num_pred_boxes: int


class BBoxTrainTestMetrics(TypedDict):
    train: BBoxEvalMetrics
    test: BBoxEvalMetrics


class PoseEstimationEvalMetrics(TypedDict, total=False):
    rmse: float
    rmse_pcutoff: float
    mAP: float
    mAR: float
    rmse_detections: float
    rmse_detections_pcutoff: float
    keypoint_rmse: list[float]
    keypoint_rmse_cutoff: list[float]
    unique_keypoint_rmse: list[float]
    unique_keypoint_rmse_cutoff: list[float]


class PoseTrainTestMetrics(TypedDict):
    train: PoseEstimationEvalMetrics
    test: PoseEstimationEvalMetrics


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PosePredictionEntry(StrictBaseModel):
    """Per-image pose data.

    Always stored in 3D format: (num_individuals, num_keypoints, ...).
    For single-animal data, num_individuals is 1.

    - keypoints: shape (N, K, 2) — [x, y] per keypoint per individual.
    - keypoint_scores: shape (N, K) — confidence/visibility per keypoint.

    Unique bodyparts (scene-level, always N=1) are stored as 2D:
    - unique_keypoints: shape (K_unique, 2)
    - unique_keypoint_scores: shape (K_unique,)
    """

    keypoints: list[list[list[float]]]
    keypoint_scores: list[list[float]]
    unique_keypoints: list[list[float]] | None = None
    unique_keypoint_scores: list[float] | None = None
    image_path: Path | None = None

    def to_keypoints_array(self, *, dtype=np.float32) -> np.ndarray:
        """Returns shape (N, K, 2)."""
        return np.asarray(self.keypoints, dtype=dtype)

    def to_scores_array(self, *, dtype=np.float32) -> np.ndarray:
        """Returns shape (N, K)."""
        return np.asarray(self.keypoint_scores, dtype=dtype)

    def to_pose_array(self, *, dtype=np.float32) -> np.ndarray:
        """Returns shape (N, K, 3) with [x, y, score]."""
        xy = self.to_keypoints_array(dtype=dtype)
        scores = self.to_scores_array(dtype=dtype)[..., None]
        return np.concatenate([xy, scores], axis=-1)

    def to_unique_pose_array(self, *, dtype=np.float32) -> np.ndarray | None:
        """Returns shape (1, K_unique, 3) with [x, y, score], or None."""
        if self.unique_keypoints is None:
            return None
        xy = np.asarray(self.unique_keypoints, dtype=dtype)
        scores = np.asarray(self.unique_keypoint_scores, dtype=dtype)
        return np.concatenate([xy, scores[..., None]], axis=-1)[None, ...]


class PosePredictions(StrictBaseModel):
    train: list[PosePredictionEntry]
    test: list[PosePredictionEntry]


DetectorContext: TypeAlias = dict[str, np.ndarray]
ImageWithContext: TypeAlias = tuple[Path, DetectorContext]
EvalMode: TypeAlias = Literal["train", "test"]
ImagesWithContext: TypeAlias = list[ImageWithContext]
BBoxFormat = Literal["xyxy", "xywh"]


class BBoxEntry(StrictBaseModel):
    """
    Detector output for one image.

    `bbox_scores` aligns one-to-one with `bboxes`.
    `bboxes` are pixel coordinates, with format given by `bbox_format`.
    """

    bboxes: list[tuple[float, float, float, float]]
    bbox_scores: list[float]
    bbox_format: BBoxFormat = "xyxy"
    image_path: Path | None = None

    def to_array(self, *, dtype=np.float32) -> np.ndarray:
        return np.asarray(self.bboxes, dtype=dtype).reshape(-1, 4)

    def to_xywh(self, *, dtype=np.float32) -> np.ndarray:
        boxes = self.to_array(dtype=dtype).copy()
        if self.bbox_format == "xyxy":
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        return boxes

    def to_xyxy(self, *, dtype=np.float32) -> np.ndarray:
        boxes = self.to_array(dtype=dtype).copy()
        if self.bbox_format == "xywh":
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return boxes

    def to_detector_context(
        self,
        *,
        dtype=np.float32,
        target_format: BBoxFormat = "xywh",
    ) -> DetectorContext:
        if target_format == "xywh":
            bboxes = self.to_xywh(dtype=dtype)
        else:
            bboxes = self.to_xyxy(dtype=dtype)

        return {
            "bboxes": bboxes,
            "bbox_scores": np.asarray(self.bbox_scores, dtype=dtype),
        }


class BBoxes(StrictBaseModel):
    train: list[BBoxEntry] = Field(default_factory=list)
    test: list[BBoxEntry] = Field(default_factory=list)

    @classmethod
    def from_file(cls, json_file: Path) -> "BBoxes":
        if not json_file.exists():
            raise FileNotFoundError(f"BBoxes file not found: {json_file}")
        return cls.from_json(json_file.read_text(encoding="utf-8"))

    @classmethod
    def from_file_if_exists(cls, json_file: Path) -> "BBoxes":
        if not json_file.exists():
            return cls()
        return cls.from_file(json_file)

    @classmethod
    def from_json(cls, json_str: str) -> "BBoxes":
        return cls.model_validate_json(json_str)

    def dump_json(self, json_file: Path) -> None:
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        json_file.write_text(self.model_dump_json(indent=4), encoding="utf-8")

    def to_images_with_context(
        self,
        image_paths: list[Path],
        mode: EvalMode,
    ) -> ImagesWithContext:
        """Zip image paths with detector context in DLC expected format."""
        mode_bboxes = getattr(self, mode)
        if len(image_paths) != len(mode_bboxes):
            raise ValueError(f"Got {len(image_paths)} {mode} images but {len(mode_bboxes)} bbox entries.")
        return [
            (image_path, bbox_entry.to_detector_context())
            for image_path, bbox_entry in zip(image_paths, mode_bboxes, strict=False)
        ]
