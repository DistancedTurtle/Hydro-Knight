"""
Rung 1 spike: compare YOLO-pose vs MediaPipe pose on the same clip.

Samples frames evenly across a video, runs both pose estimators on each,
draws their detected skeletons, and saves side-by-side PNGs so you can
eyeball which one actually finds the swimmers.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MP_MODEL = Path("raw_local/pose_landmarker.task")


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def _resize_h(img, target_h):
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h))


def _draw_mp_pose(img, poses, connections):
    h, w = img.shape[:2]
    for landmarks in poses:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for c in connections:
            cv2.line(img, pts[c.start], pts[c.end], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(img, p, 3, (0, 0, 255), -1)


def compare(video_path: Path, out_dir: Path, n_frames: int = 8, imgsz: int = 640) -> None:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO("yolo11n-pose.pt")

    base = mp_python.BaseOptions(model_asset_path=str(MP_MODEL))
    opts = vision.PoseLandmarkerOptions(base_options=base,
                                        running_mode=vision.RunningMode.IMAGE,
                                        num_poses=10,
                                        min_pose_detection_confidence=0.3)
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total - 1, n_frames, dtype=int)

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        yolo_result = yolo(frame, verbose=False, imgsz=imgsz)[0]
        yolo_img = yolo_result.plot()
        yolo_count = len(yolo_result.boxes)

        mp_img = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = landmarker.detect(mp_image)
        mp_count = len(mp_result.pose_landmarks)
        _draw_mp_pose(mp_img, mp_result.pose_landmarks, connections)

        _label(yolo_img, f"YOLO-pose: {yolo_count} people")
        _label(mp_img, f"MediaPipe: {mp_count} people")

        h = min(yolo_img.shape[0], mp_img.shape[0])
        combined = np.hstack([_resize_h(yolo_img, h), _resize_h(mp_img, h)])

        out_path = out_dir / f"compare_{i:02d}_frame{int(idx)}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"[{i+1}/{len(frame_indices)}] frame {idx}: "
              f"YOLO {yolo_count} | MediaPipe {mp_count} -> {out_path.name}")

    cap.release()
    landmarker.close()
    print(f"\nDone. Open the PNGs in {out_dir} to compare.")


if __name__ == "__main__":
    import sys
    video = Path(sys.argv[1])
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    out = Path("raw_local/pose_spike_out") / f"{video.stem}_imgsz{imgsz}"
    compare(video, out, n_frames=8, imgsz=imgsz)
