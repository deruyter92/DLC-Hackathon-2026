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


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BBoxEntry(StrictBaseModel):
    """Detector output for one image.

    `bboxes` follow xyxy pixel coordinates: (x1, y1, x2, y2).
    `bbox_scores` aligns one-to-one with `bboxes`.
    """

    bboxes: list[tuple[float, float, float, float]]
    bbox_scores: list[float]
    image_path: Path | None = None

    def to_detector_context(
        self, *, dtype: np.dtype[Any] = np.float32
    ) -> DetectorContext:
        """Convert this entry to DLC detector context format."""
        return {
            "bboxes": np.asarray(self.bboxes, dtype=dtype),
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
        serializable = {
            "train": _numpy_to_jsonable(self.train),
            "test": _numpy_to_jsonable(self.test),
        }
        Path(json_file).parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, "w") as f:
            json.dump(serializable,f, indent=4)

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
