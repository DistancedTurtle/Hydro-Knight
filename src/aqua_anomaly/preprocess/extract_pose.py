"""
Pose extraction (Rung 1): video -> per-swimmer keypoint time-series (Parquet).

Runs YOLO-pose with tracking over a clip. Every detected swimmer in every
frame becomes one row: frame index, track id, box confidence, and the 17
COCO keypoints as (x, y, confidence). That table is the training data — group
by track_id, sort by frame, and you have each swimmer's motion over time.

Non-tiled version (SAHI tiling is a later upgrade). Model is a parameter so
the YOLO version/size is swappable without code changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

# 17 COCO keypoints, in the order YOLO returns them.
KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# Column layout of the output table: metadata first, then x/y/conf per keypoint.
COLUMNS = ["frame", "track_id", "box_conf"] + [
    f"{axis}{i}" for i in range(17) for axis in ("x", "y", "c")
]


def extract(
    video_path: Path,
    out_path: Path,
    model_name: str = "yolo11n-pose.pt",
    imgsz: int = 1280,
    conf: float = 0.20,
    tracker: str = "bytetrack.yaml",
    device: str | None = None,
    max_frames: int | None = None,
) -> int:
    """
    Extract pose keypoints from one clip into a Parquet file.

    Returns the number of rows (swimmer-detections) written.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    # model.track(... stream=True) yields one Results object per frame, while
    # maintaining a persistent track_id per swimmer across frames (the tracker).
    # stream=True processes frames lazily so we don't hold the whole video in RAM.
    results = model.track(
        source=str(video_path),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        tracker=tracker,
        device=device,
        verbose=False,
    )

    rows: list[list[float]] = []
    for frame_idx, result in enumerate(results):
        if max_frames is not None and frame_idx >= max_frames:
            break

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue  # no swimmers this frame

        # Track IDs: present once the tracker has locked on. If a detection has
        # no id yet (e.g. first sighting), fall back to -1.
        ids = boxes.id.int().tolist() if boxes.id is not None else [-1] * len(boxes)
        confs = boxes.conf.tolist()

        # keypoints.data is a tensor (n_people, 17, 3) of (x, y, confidence).
        kpts = result.keypoints.data.cpu().numpy()

        for p in range(len(boxes)):
            row = [frame_idx, ids[p], float(confs[p])]
            row.extend(kpts[p].flatten().tolist())  # 17*(x,y,c) = 51 values
            rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_parquet(out_path, index=False)

    n_tracks = df["track_id"].nunique() if len(df) else 0
    print(f"{video_path.name}: {len(df)} detections across "
          f"{df['frame'].nunique() if len(df) else 0} frames, "
          f"{n_tracks} unique tracks -> {out_path}")
    return len(df)


if __name__ == "__main__":
    import sys
    video = Path(sys.argv[1])
    out = Path("data/keypoints") / f"{video.stem}.parquet"
    extract(video, out)
