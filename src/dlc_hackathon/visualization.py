from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dlc_hackathon.schemas.types import BBoxEntry, PosePredictionEntry


def plot_bbox_entry(
    bbox: BBoxEntry,
    *,
    ax: Axes | None = None,
    color: str = "lime",
    linewidth: float = 2,
    linestyle: str = "-",
    show_scores: bool = True,
    show_image: bool = True,
    label: str | None = None,
) -> tuple[Figure, Axes]:
    """Overlay bounding boxes on an existing or new axes.

    Args:
        bbox: The BBoxEntry to visualize.
        ax: Matplotlib axes to draw on. If None, creates a new figure.
            When ``show_image`` is True and no axes is provided, the image at
            ``bbox.image_path`` is displayed as background.
        color: Edge color for boxes and score text.
        linewidth: Line width for box edges.
        linestyle: Line style for box edges (e.g. "-", "--", ":").
        show_scores: Whether to render confidence scores above each box.
        show_image: Whether to show the image as background (only when creating
            a new figure, i.e. ``ax is None``).
        label: Optional label added to the first box (useful for legend).

    Returns:
        The (figure, axes) tuple so callers can further customize the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        if show_image:
            if bbox.image_path is None:
                raise ValueError("BBoxEntry.image_path is required when show_image=True.")
            image = plt.imread(bbox.image_path)
            ax.imshow(image)
            ax.set_title(str(bbox.image_path))
        ax.axis("off")
    else:
        fig = ax.get_figure()

    boxes_xyxy = bbox.to_xyxy(dtype=np.float32)
    for idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label if idx == 0 else None,
        )
        ax.add_patch(rect)

        if show_scores and idx < len(bbox.bbox_scores):
            ax.text(
                x1,
                max(0, y1 - 2),
                f"{bbox.bbox_scores[idx]:.2f}",
                color=color,
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
            )

    return fig, ax


def plot_pose_prediction(
    entry: PosePredictionEntry,
    *,
    ax: Axes | None = None,
    color: str = "cyan",
    marker: str = "o",
    marker_size: float = 5,
    show_scores: bool = False,
    score_threshold: float = 0.0,
    show_image: bool = True,
    label: str | None = None,
    connections: list[tuple[int, int]] | None = None,
    connection_color: str | None = None,
    connection_linewidth: float = 1.5,
) -> tuple[Figure, Axes]:
    """Overlay predicted keypoints on an existing or new axes.

    Args:
        entry: A PosePredictionEntry with keypoints, scores, and optional image_path.
        ax: Matplotlib axes to draw on. If None, creates a new figure.
        color: Color for keypoint markers.
        marker: Matplotlib marker style.
        marker_size: Size of keypoint markers.
        show_scores: Whether to render the score next to each keypoint.
        score_threshold: Only plot keypoints with score above this value.
        show_image: Whether to show the image as background (only when ``ax is None``).
        label: Optional label for the scatter (useful for legend).
        connections: Optional list of (i, j) index pairs to draw skeleton lines.
        connection_color: Color for skeleton lines. Defaults to ``color``.
        connection_linewidth: Line width for skeleton lines.

    Returns:
        The (figure, axes) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        if show_image:
            image_path = entry.image_path
            if image_path is None:
                raise ValueError("PosePredictionEntry.image_path is required when show_image=True.")
            image = plt.imread(image_path)
            ax.imshow(image)
            ax.set_title(str(image_path))
        ax.axis("off")
    else:
        fig = ax.get_figure()

    keypoints = entry.to_keypoints_array().reshape(-1, 2)
    scores = entry.to_scores_array().ravel()

    if len(scores) < len(keypoints):
        scores = np.ones(len(keypoints), dtype=np.float32)

    visible = scores > score_threshold
    xs = keypoints[visible, 0]
    ys = keypoints[visible, 1]

    ax.scatter(xs, ys, c=color, marker=marker, s=marker_size**2, zorder=3, label=label)

    if show_scores:
        for x, y, s in zip(xs, ys, scores[visible]):
            ax.text(
                x + 2,
                y - 2,
                f"{s:.2f}",
                color=color,
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.4, "pad": 0.5},
            )

    if connections is not None:
        conn_color = connection_color or color
        for i, j in connections:
            if i < len(keypoints) and j < len(keypoints) and visible[i] and visible[j]:
                ax.plot(
                    [keypoints[i, 0], keypoints[j, 0]],
                    [keypoints[i, 1], keypoints[j, 1]],
                    color=conn_color,
                    linewidth=connection_linewidth,
                    zorder=2,
                )

    return fig, ax
