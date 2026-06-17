"""
Interactive clip annotator.

Opens each downloaded clip in a window and waits for keypresses to label,
trim, or delete it. Writes changes back to the manifest immediately after
each clip so progress is never lost if you quit mid-session.

Controls
--------
  SPACE       pause / resume playback
  LEFT / RIGHT  seek backward / forward 5 seconds
  I           set in-point  (trim start) at current position
  O           set out-point (trim end)   at current position
  N           label as NORMAL    and advance to next clip
  D           label as DISTRESS  and advance to next clip
  S           label as SUBMERGED and advance to next clip
  F           label as FACE_DOWN and advance to next clip
  R           label as REVIEW    and advance to next clip (watch again later)
  X           delete clip from manifest and advance
  Q           quit the session
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2

from ..ingest.manifest import CameraView, ClipRecord, Label, Manifest
from ..ingest.download import local_path, is_downloaded


# --- Overlay rendering -------------------------------------------------------

def _draw_overlay(
    frame,
    current_sec: float,
    total_sec: float,
    in_point: float | None,
    out_point: float | None,
    paused: bool,
    label: Label,
) -> None:
    """
    Draw a heads-up display on top of the current frame.

    We write directly onto the frame array that OpenCV is about to display.
    This is purely visual — it doesn't affect the video file or any saved data.

    The overlay shows:
    - Current timestamp and total duration
    - Active trim in/out points
    - Current working label
    - Paused indicator
    - Key reference
    """
    h, w = frame.shape[:2]

    # Semi-transparent black bar at the bottom for readability.
    # We draw a filled rectangle and then text on top of it.
    bar_h = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    # Alpha blending: mix the black bar with the original frame at 60% opacity
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def text(msg: str, row: int, col: int = 10, color=(255, 255, 255)):
        cv2.putText(frame, msg, (col, h - bar_h + row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Row 1: timestamp + pause indicator
    pause_str = "  [PAUSED]" if paused else ""
    text(f"{_fmt(current_sec)} / {_fmt(total_sec)}{pause_str}", 22)

    # Row 2: trim points
    in_str  = _fmt(in_point)  if in_point  is not None else "not set"
    out_str = _fmt(out_point) if out_point is not None else "not set"
    text(f"IN: {in_str}   OUT: {out_str}", 44)

    # Row 3: current label
    label_color = {
        Label.NORMAL:    (100, 220, 100),
        Label.DISTRESS:  (60,  60,  220),
        Label.SUBMERGED: (220, 180, 60),
        Label.FACE_DOWN: (60,  200, 220),
        Label.REVIEW:    (180, 180, 180),
        Label.UNLABELED: (180, 180, 180),
    }.get(label, (255, 255, 255))
    text(f"Label: {label.value.upper()}", 66, color=label_color)

    # Row 4: key reference
    text("SPC=pause  I/O=trim  N/D/S/F/R=label  X=delete  Q=quit", 92, color=(180, 180, 180))

    # Progress bar along the bottom edge
    if total_sec > 0:
        bar_y = h - 4
        bar_w = int(w * current_sec / total_sec)
        cv2.rectangle(frame, (0, bar_y - 4), (w, bar_y), (60, 60, 60), -1)
        cv2.rectangle(frame, (0, bar_y - 4), (bar_w, bar_y), (100, 200, 100), -1)

        # Mark in/out points on the progress bar
        if in_point is not None:
            x = int(w * in_point / total_sec)
            cv2.rectangle(frame, (x - 2, bar_y - 8), (x + 2, bar_y), (255, 255, 0), -1)
        if out_point is not None:
            x = int(w * out_point / total_sec)
            cv2.rectangle(frame, (x - 2, bar_y - 8), (x + 2, bar_y), (255, 100, 0), -1)


def _fmt(seconds: float) -> str:
    """Convert a float number of seconds to MM:SS display string."""
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --- Single clip review ------------------------------------------------------

def _review_clip(record: ClipRecord, manifest: Manifest) -> str:
    """
    Open one clip for review. Returns the action taken:
    'next', 'deleted', or 'quit'.
    """
    path = local_path(record)
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print(f"Could not open {path} — skipping.")
        return "next"

    fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps

    # Initialise trim points from whatever is already in the manifest record.
    # If the record has start_sec=0 and end_sec=-1 (the defaults from collect),
    # we treat those as "no trim set yet."
    in_point  = record.start_sec if record.start_sec > 0    else None
    out_point = record.end_sec   if record.end_sec   > 0    else None

    current_label = record.label
    paused = False
    action = "next"

    window_title = f"Annotator — {record.notes[:60] or record.clip_id}"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 780)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Reached the end of the video — pause and wait for a label key.
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()
                if not ret:
                    break

        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        _draw_overlay(frame, current_sec, total_sec, in_point, out_point, paused, current_label)
        cv2.imshow(window_title, frame)

        # waitKey(delay) waits `delay` milliseconds for a keypress.
        # We calculate the delay from the video's frame rate so playback
        # runs at approximately the correct speed.
        # When paused we wait 50ms between polls so the UI stays responsive.
        delay = max(1, int(1000 / fps)) if not paused else 50
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            action = "quit"
            break

        elif key == ord(' '):
            paused = not paused

        elif key == 81 or key == 2:   # left arrow (Linux/Mac codes differ)
            seek_sec = max(0, current_sec - 5)
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000)

        elif key == 83 or key == 3:   # right arrow
            seek_sec = min(total_sec, current_sec + 5)
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000)

        elif key == ord('i'):
            in_point = current_sec
            print(f"  In-point set: {_fmt(in_point)}")

        elif key == ord('o'):
            out_point = current_sec
            print(f"  Out-point set: {_fmt(out_point)}")

        elif key in (ord('n'), ord('d'), ord('s'), ord('f'), ord('r')):
            current_label = {
                ord('n'): Label.NORMAL,
                ord('d'): Label.DISTRESS,
                ord('s'): Label.SUBMERGED,
                ord('f'): Label.FACE_DOWN,
                ord('r'): Label.REVIEW,
            }[key]

            # Build the updated record. dataclasses.replace() creates a new
            # ClipRecord with all the same fields except the ones you specify —
            # cleaner than constructing a whole new object manually.
            updated = dataclasses.replace(
                record,
                label     = current_label,
                start_sec = in_point  if in_point  is not None else record.start_sec,
                end_sec   = out_point if out_point is not None else record.end_sec,
            )
            manifest.update(updated)
            print(f"  Saved: label={current_label.value}  in={_fmt(in_point or 0)}  out={_fmt(out_point or total_sec)}")
            action = "next"
            break

        elif key == ord('x'):
            manifest.delete(record.clip_id)
            print(f"  Deleted {record.clip_id} from manifest.")
            action = "deleted"
            break

    cap.release()
    cv2.destroyWindow(window_title)
    return action


# --- Session loop ------------------------------------------------------------

def annotate(
    manifest_path: Path,
    only_unlabeled: bool = True,
) -> None:
    """
    Run an annotation session over all downloaded clips.

    Parameters
    ----------
    manifest_path :
        Path to the JSONL manifest.
    only_unlabeled :
        If True (default), only show clips with label UNLABELED or REVIEW.
        Set to False to re-review all clips including already-labeled ones.
    """
    manifest = Manifest(manifest_path)
    records  = manifest.load()

    # Filter to clips that are downloaded and need attention.
    queue = [
        r for r in records
        if is_downloaded(r) and (
            not only_unlabeled or r.label in (Label.UNLABELED, Label.REVIEW)
        )
    ]

    if not queue:
        print("No clips to annotate. Run download.py first, or set only_unlabeled=False to re-review.")
        return

    print(f"{len(queue)} clips in queue.")
    print("Controls: SPACE=pause  LEFT/RIGHT=seek  I/O=trim  N/D/S/F/R=label  X=delete  Q=quit\n")

    for i, record in enumerate(queue, start=1):
        print(f"[{i}/{len(queue)}] {record.notes[:70] or record.clip_id}")
        action = _review_clip(record, manifest)
        if action == "quit":
            print("Session ended.")
            break

    print("Annotation session complete.")


if __name__ == "__main__":
    annotate(
        manifest_path=Path("data/manifests/pool_footage.jsonl"),
    )
