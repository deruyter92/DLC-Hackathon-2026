from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict
import json
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def _numpy_to_jsonable(obj: Any) -> Any:
  if isinstance(obj, np.ndarray):
      return obj.tolist()
  if isinstance(obj, dict):
      return {k: _numpy_to_jsonable(v) for k, v in obj.items()}
  if isinstance(obj, list):
      return [_numpy_to_jsonable(x) for x in obj]
  return obj



class EvalMetrics(TypedDict):
    pass
    # TODO: add metrics


class PosePredictionEntry(TypedDict):
    keypoints: list[list[float]]
    keypoint_scores: list[list[float]]
    image_path: Path | None = None


class PosePredictions(TypedDict):
    train: list[PosePredictionEntry]
    test: list[PosePredictionEntry]


DetectorContext: TypeAlias = dict[str, np.ndarray]
ImageWithContext: TypeAlias = tuple[Path, DetectorContext]
EvalMode: TypeAlias = Literal["train", "test"]
ImagesWithContext: TypeAlias = list[ImageWithContext]
BBboxFormat: TypeAlias = Literal["xyxy", "xywh"]

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


from typing import Literal

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
    def from_file(cls, json_file: Path, missing_ok: bool = False) -> "BBoxes":
        if not json_file.exists():
            if missing_ok:
                return cls()
            raise FileNotFoundError(f"BBoxes file not found: {json_file}")
        return cls.from_json(json_file.read_text(encoding="utf-8"))

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
            raise ValueError(
                f"Got {len(image_paths)} {mode} images but {len(mode_bboxes)} bbox entries."
            )
        return [
            (image_path, bbox_entry.to_detector_context())
            for image_path, bbox_entry in zip(image_paths, mode_bboxes)
        ]
