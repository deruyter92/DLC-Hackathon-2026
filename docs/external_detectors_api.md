External detectors for top-down DeepLabCut (PyTorch)
====================================================

Overview
--------
This extension adds support for "external" object detectors in the PyTorch
top-down pose pipeline. The intended workflow is:

    image/video frame
      -> detector
      -> bounding boxes / crops
      -> top-down pose model

The design supports two closely related use cases:

1. Live external detector inference
   A foundation model or other pretrained detector is run at inference time
   and its detections are converted into the DeepLabCut detector output format.

2. Offline / precomputed detector outputs
   Detector outputs are run once, saved to disk, and then reused later for
   training or inference of the top-down pose model without rerunning the
   detector.

This is especially useful when:
- detector inference is expensive,
- detector code has heavyweight dependencies,
- detector weights should remain frozen,


Core concepts
-------------
1. **BaseExternalDetector**
   External detectors implement a minimal detector API by subclassing
   BaseExternalDetector and registering themselves in EXTERNAL_DETECTORS.

   A detector must expose:

       predict(images: list[torch.Tensor]) -> list[DetectionResult]

   where each DetectionResult is a dictionary containing at least:

       {
           "boxes":  FloatTensor[N, 4],  # absolute XYXY pixel coordinates
           "scores": FloatTensor[N],
           "labels": LongTensor[N],
       }

   The detector is assumed to be inference-oriented and typically frozen.
   It does not need to implement a training loop.
   Of course it is possible to implement a trainable detector, so feel free to extend the API as you see fit.

2. **Detector inference runner compatibility**
   BaseExternalDetector provides a forward() shim so it can be used by the
   existing DLC inference runner stack. This means an external detector can
   be wrapped by a standard DetectorInferenceRunner and postprocessed into
   DLC-style detector context:

       {
           "bboxes": np.ndarray[N, 4],
           "bbox_scores": np.ndarray[N],
       }

   In the top-down data path, boxes are expected in XYWH format after
   postprocessing.
   Note that other useful fields like "classes" or "labels" are currently accepted but not deeply integrated into the rest of the code, so feel free to extend there too.

3. **PrecomputedDetectorRunner**
   PrecomputedDetectorRunner is an adapter that makes saved bounding boxes
   behave like a detector runner. It implements:

       inference(images, shelf_writer=None) -> list[DetectorContext]

   so it can be plugged directly into dataset creation or other DLC code that
   expects a detector runner. It will load from disk and emit DLC-formatted
   detector outputs, mimicking a live detector.

4. **BBox schema**
   Precomputed detections are stored in a JSON artifact using two schema types:

   - BBoxEntry: detections for one image
   - BBoxes: split-aware container with:
         train: list[BBoxEntry]
         test:  list[BBoxEntry]

   Each BBoxEntry contains:
   - bboxes
   - bbox_scores
   - bbox_format ("xywh" or "xyxy" or "cxcywh", etc.)
   - optional image_path

5. **Bounding-box source selection**
   Dataset creation now owns the decision of where bounding boxes come from.
   For top-down and detection tasks, the bbox source is resolved in this order:

   1. detector_runner is provided
      -> use detection boxes
   2. explicit config value data.bbox_source
   3. loader/task default

   For historical DLC compatibility, DLCLoader defaults to KEYPOINTS for
   top-down and detect tasks unless overridden. This preserves the previous
   behavior for projects that do not use external detectors.

6. **Multi-animal matching**
   For multi-animal top-down training, detector predictions must be matched to
   annotated individuals. This implementation does that by:

   - deriving a reference box for each annotated individual
     (prefer keypoints, otherwise existing bbox),
   - computing pairwise IoU between reference boxes and detector boxes,
   - solving assignment with Hungarian matching,
   - accepting matches above bbox_match_iou_threshold,
   - optionally falling back to the original GT/reference bbox when no match
     is found.

   Notes:
   - only "real individual" annotations are matched (category_id == 1),
     which avoids assigning detector boxes to unique-bodypart-only annotations.
     Note: In DLC "unique bodypart" annotations are meant to be used for e.g. background references or "landmark" keypoints that are not associated with an actual animal instance, such as the corner of a cage, a reward port, etc.
   - in the single-animal case, the highest-scoring detector box is used
     directly.

What this does not do (yet)
--------------------------------------

- There is **no bounding box validity checking or filtering built into the dataset creation process** beyond the IoU-based matching and optional GT fallback. Check for negative crops, NaNs, boxes outside the image, or other edge cases in your detector implementation or in a wrapper before saving precomputed outputs if needed.
- **Handling model configs/reproducibility signals** and other model specific details is up to you.
- If your particular detector outputs require **custom pre- or post-processing** (e.g. custom score thresholding, prompts, class filtering, NMS, etc.), it is currently expected that you handle this logic within your custom detector implementation or in a wrapper around it before saving precomputed outputs. The current code does not provide built-in utilities for these common detection post-processing steps.
- The **current multi-animal matching logic is based on very basic IoU** and does not consider other potential matching signals such as class labels, keypoint visibility patterns, temporal consistency, or other heuristics. This is a simple but effective strategy for many cases, but it may not be optimal for all scenarios. You may want to explore stronger strategies if relevant. This would be critical for robust performance in challenging multi-animal scenarios.

Implementing a new external detector
------------------------------------
To add a detector, create a new module under:

    deeplabcut/pose_estimation_pytorch/models/detectors/external/

and register it in EXTERNAL_DETECTORS.

Minimal example:

    from __future__ import annotations
    import torch
    from .base import EXTERNAL_DETECTORS, BaseExternalDetector

    @EXTERNAL_DETECTORS.register_module
    class MyExternalDetector(BaseExternalDetector):
        def __init__(self, checkpoint=None, score_threshold=0.25):
            super().__init__()
            self.checkpoint = checkpoint
            self.score_threshold = score_threshold
            # load your model here

        def predict(self, images: list[torch.Tensor]):
            outputs = []
            for image in images:
                # image is CHW
                # run detector here and return XYXY absolute pixel coords
                boxes = torch.empty((0, 4), dtype=torch.float32)
                scores = torch.empty((0,), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.long)

                outputs.append(
                    {
                        "boxes": boxes,
                        "scores": scores,
                        "labels": labels,
                    }
                )
            return outputs

Requirements and conventions:
- input images are tensors, usually [C, H, W]
- boxes must be absolute pixel coordinates in XYXY
- scores should align one-to-one with boxes
- labels are currently accepted but not yet deeply used by the top-down path
- detector weights are typically frozen
- forward() is inherited and automatically wraps predict()


Building a live detector runner
-------------------------------
A helper is provided to build an inference runner for external detectors:

    get_external_detector_inference_runner(
        detector_cfg,
        batch_size,
        device,
        max_individuals,
        color_mode,
        transform=None,
        inference_cfg=None,
        min_bbox_score=None,
    )

This helper:
- builds the detector from EXTERNAL_DETECTORS,
- freezes its parameters when possible,
- sets eval() when available,
- builds a standard DLC detector inference runner,
- uses the existing detector postprocessor to emit DLC detector outputs.

This is the recommended path when you want to run a live detector and obtain
DLC-formatted detector outputs.


Saving precomputed detector outputs
-----------------------------------
To run a detector once and save the results for later reuse:

    precompute_detector_bboxes(
        loader,
        detector_runner,
        output_file,
        modes=("train", "test"),
        bbox_format="xywh",
    )

This will:
- iterate over all images in the requested split(s),
- call detector_runner.inference(image_paths),
- convert outputs into BBoxEntry records,
- save them as a BBoxes JSON file.

This artifact can later be reused without any live detector.


Using precomputed detections in training
----------------------------------------
A top-down pose config can be generated with precomputed detector boxes:

    make_pytorch_pose_config(
        project_config=...,
        pose_config_path=...,
        method=...,
        precomputed_bboxes="path/to/precomputed_bboxes.json",
        bbox_source="detection_bbox",   # optional when precomputed_bboxes is set
        external_detector_metadata={...},
        save=True,
    )

When precomputed_bboxes is provided:
- task must be top-down,
- data.bbox_source is automatically set to "detection_bbox",
- data.precomputed_bboxes is saved into the config,
- safe defaults are applied:
    - bbox_match_iou_threshold = 0.1
    - bbox_fallback_to_gt = True
    - bbox_validate_image_paths = False

During train():
- if task == TOP_DOWN, a PrecomputedDetectorRunner is built for train and test
  splits from model_cfg["data"]["precomputed_bboxes"],
- loader.create_dataset(..., detector_runner=...) rewrites annotation bboxes
  using the detector outputs,
- the pose model is then trained on crops derived from those boxes.

Important:
once the dataset is built, the detector is no longer needed for pose training.
Only the pose model parameters are optimized.


Config fields
-------------
Relevant config fields introduced by this extension:
```yaml
data:
  bbox_source: "gt" | "keypoints" | "detection_bbox" | "segmentation_mask"
  precomputed_bboxes: "/path/to/precomputed_bboxes.json"
  bbox_margin: 20
  bbox_match_iou_threshold: 0.1
  bbox_fallback_to_gt: true
  bbox_validate_image_paths: false

metadata:
  external_detector:
    ... free-form detector metadata ...
```

Fields:
- bbox_source
  Selects how bounding boxes are obtained.
- precomputed_bboxes
  Path to a BBoxes JSON artifact.
- bbox_margin
  Margin used when deriving reference boxes from keypoints.
- bbox_match_iou_threshold
  Minimum IoU required to assign a detector box to an annotated individual.
- bbox_fallback_to_gt
  If true, unmatched annotations keep their original/reference bbox.
  If false, unmatched annotations receive an empty bbox and zero area.
- bbox_validate_image_paths
  Whether to verify that saved bbox entries align with requested image paths.
- metadata.external_detector
  Optional provenance information (detector name, version, checkpoint, etc.).

See these in the code for better context.

Precomputed bbox JSON format
-----------------------
Example:

    {
      "train": [
        {
          "bboxes": [
            [15.0, 15.0, 20.0, 20.0],
            [65.0, 15.0, 20.0, 20.0]
          ],
          "bbox_scores": [0.8, 0.9],
          "bbox_format": "xywh",
          "image_path": "img0.png"
        }
      ],
      "test": [
        {
          "bboxes": [
            [15.0, 15.0, 20.0, 20.0],
            [65.0, 15.0, 20.0, 20.0]
          ],
          "bbox_scores": [0.8, 0.9],
          "bbox_format": "xywh",
          "image_path": "img0.png"
        }
      ]
    }

Semantics:
- one BBoxEntry per image,
- each entry may contain zero, one, or many detections,
- detector order does not need to match annotation order,
- in multi-animal training, assignment is recovered by IoU matching. Feel free to add better matching signals if you can !

Please adapt the format and I/O if needed, but clearly document any changes and ensure the PrecomputedDetectorRunner can read your format correctly.

A Lot More Details That AI Seems to Care About
----------------------------------
This integration treats object detection as a **separate stage** for
top-down pose estimation.

Training-time flow sketch:
- annotations already contain keypoints for each individual,
- an external detector supplies candidate boxes,
- these boxes are matched to annotated individuals,
- matched boxes replace or augment the crop source,
- **the pose estimator learns on crops that more closely resemble test-time
  detector proposals.** This is the crucial bit that can improve generalization and reduce train/test mismatch.

Why this matters:
- if a pose model is trained only on GT or keypoint-derived crops, but test-time
  crops come from a detector, there can be a train/test crop-domain mismatch.

Matching strategy reminder:
- reference boxes are built from visible keypoints when possible,
- otherwise existing annotation["bbox"] is used,
- detector boxes are compared to reference boxes using IoU,
- Hungarian assignment finds the globally best one-to-one matching,
- low-IoU assignments are rejected,
- optional fallback preserves robustness when detections are missing.

This is particularly important for multi-animal projects because **detector order
is not guaranteed to correspond to annotation order**.

Format conversions:
- external detectors may emit XYXY or any other format (convert it to XYXY in predict()),
- DLC top-down context typically expects XYWH,
- BBoxEntry can convert between XYXY and XYWH formats and serialize the result.
- Extend the class if needed.

Caching and mutation safety:
- loader.load_data() may be cached,
- dataset creation deep-copies annotations before rewriting bboxes,
- this prevents detector-based bbox replacement from mutating cached raw data.


Practical recommendations
-------------------------
1. **Store detector provenance**:
   Save detector name, checkpoint, prompt/class definition, score thresholds,
   and version info in metadata.external_detector.

2. **Prefer path validation when possible**:
   During development, set bbox_validate_image_paths = true to catch ordering
   mistakes between images and saved detections.

3. **Tune IoU threshold carefully**:
   For imperfect foundation-model boxes, a low threshold such as 0.1 may be
   appropriate; for cleaner detector outputs, a higher threshold may reduce
   accidental mismatches. More robust methods would be both interesting and useful here, plenty exist in the literature.

4. **Decide on fallback behavior explicitly**:
   - bbox_fallback_to_gt = true is safer and more forgiving
   - bbox_fallback_to_gt = false enforces stricter reliance on detector outputs
   - Recommended: start with False to understand detector quality, then consider True once you have a good understanding of why GT fallbacks may be needed and whether it is makes sense for your use case.

5. **Filter or postprocess detector outputs upstream if needed**:
   If your detector predicts multiple classes or a very large number of boxes,
   consider restricting to the target animal class and applying score/NMS
   upstream before saving precomputed outputs to keep the dataset manageable.


Current scope
-------------
This implementation provides:
- an external detector interface,
- a registry for external detectors,
- a live detector runner builder,
- a JSON schema for precomputed detections,
- a precomputed detector runner adapter,
- dataset-level bbox replacement for top-down training,
- multi-animal IoU-based assignment logic.

The most mature path at present is:
- run detector externally or through the external detector runner,
- save detections as a BBoxes JSON artifact,
- train a top-down pose model using precomputed detector boxes.


Testing coverage included here
------------------------------
The current tests cover:
- detector registry/build path,
- end-to-end mock external detector inference,
- schema round-tripping,
- precomputed detector runner contract,
- create_dataset() integration,
- cached annotation immutability,
- multi-animal matching with reversed detector order,
- end-to-end pose training on offline/precomputed boxes.

These tests establish the main contract for using external detector outputs as
the source of crops for top-down pose training.
Since the purpose of the hackathon is to add new detectors we have not tested the actual training-time pose estimation performance when using external detectors, but we will help fix any issues that arise during the hackathon because of this code :)
