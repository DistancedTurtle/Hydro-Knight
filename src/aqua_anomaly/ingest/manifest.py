"""
Manifest schema and read/write logic for clip records.

A manifest is a JSONL file (one JSON object per line) where each line
describes one video clip: where it came from, what conditions it was filmed
in, and what label it has been given.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


# --- Controlled vocabularies -------------------------------------------------
# These Enums define the only legal values for categorical fields.
# Using an Enum instead of a plain string means a typo ("outdooor") becomes
# an immediate crash rather than silent bad data in the manifest.

class CameraView(str, Enum):
    OVERHEAD   = "overhead"    # camera mounted directly above the pool
    ELEVATED   = "elevated"    # camera on a stand or high wall, angled down
    DECK_LEVEL = "deck_level"  # roughly eye-level with the water surface
    UNDERWATER = "underwater"  # below the surface
    UNKNOWN    = "unknown"


class Setting(str, Enum):
    OUTDOOR = "outdoor"
    INDOOR  = "indoor"
    UNKNOWN = "unknown"


class TimeOfDay(str, Enum):
    DAY     = "day"
    DUSK    = "dusk"
    NIGHT   = "night"
    UNKNOWN = "unknown"


class Weather(str, Enum):
    CLEAR    = "clear"
    OVERCAST = "overcast"
    RAIN     = "rain"
    UNKNOWN  = "unknown"


class Label(str, Enum):
    NORMAL    = "normal"    # confirmed normal swimming activity
    DISTRESS  = "distress"  # confirmed drowning / distress event
    SUBMERGED = "submerged" # person submerged but outcome unknown
    FACE_DOWN = "face_down" # prone face-down beyond normal duration
    REVIEW    = "review"    # needs a human to watch before labeling
    UNLABELED = "unlabeled" # not yet looked at


# --- Clip record -------------------------------------------------------------

@dataclass
class ClipRecord:
    """One row in the manifest — represents a single video clip."""

    clip_id:      str        # deterministic hash, computed from source + timestamps
    source_url:   str        # original URL the clip came from
    platform:     str        # e.g. "youtube", "vimeo", "local"
    start_sec:    float      # where in the source video this clip starts (seconds)
    end_sec:      float      # where it ends; use -1.0 to mean "to the end"
    camera_view:  CameraView
    setting:      Setting
    time_of_day:  TimeOfDay
    weather:      Weather
    label:        Label
    notes:        str = ""   # free-text, optional


def make_clip_id(source_url: str, start_sec: float, end_sec: float) -> str:
    """
    Produce a short, stable identifier for a clip.

    We hash the three fields that together uniquely identify a clip (the URL
    it came from and the start/end timestamps). The hash is truncated to 12
    hex characters — long enough to avoid accidental collisions in a dataset
    of thousands of clips, short enough to be readable in filenames.

    Hashing guarantees that running the collect script twice on the same
    search results produces the same IDs, so de-duplication works correctly.
    """
    key = f"{source_url}|{start_sec}|{end_sec}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# --- Manifest reader / writer ------------------------------------------------

class Manifest:
    """
    Reads and writes a JSONL manifest file.

    JSONL ("JSON Lines") means each line of the file is its own complete JSON
    object. This format is append-safe: you can add new clips by writing a new
    line without rewriting the whole file. It also diffs cleanly in git
    because each clip is on its own line.
    """

    def __init__(self, path: Path) -> None:
        # Store the path but don't open the file yet.
        # The file is created lazily on the first write.
        self.path = Path(path)

    def load(self) -> list[ClipRecord]:
        """
        Read all records from the manifest file.

        Returns an empty list if the file does not yet exist — callers don't
        need to check for the file's existence before calling this.
        """
        if not self.path.exists():
            return []

        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # Convert the raw string values back into their Enum types
                # so callers always get ClipRecord objects with Enum fields,
                # never bare strings.
                records.append(ClipRecord(
                    clip_id    = data["clip_id"],
                    source_url = data["source_url"],
                    platform   = data["platform"],
                    start_sec  = data["start_sec"],
                    end_sec    = data["end_sec"],
                    camera_view = CameraView(data["camera_view"]),
                    setting     = Setting(data["setting"]),
                    time_of_day = TimeOfDay(data["time_of_day"]),
                    weather     = Weather(data["weather"]),
                    label       = Label(data["label"]),
                    notes       = data.get("notes", ""),
                ))
        return records

    def append(self, record: ClipRecord) -> bool:
        """
        Write one record to the manifest, skipping it if already present.

        Returns True if the record was written, False if it was a duplicate.

        De-duplication is by clip_id. Because clip_id is a deterministic hash
        of (url, start, end), running the same search twice won't add the same
        clip twice.
        """
        existing_ids = {r.clip_id for r in self.load()}
        if record.clip_id in existing_ids:
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("a", encoding="utf-8") as f:
            # asdict() converts the dataclass to a plain dict.
            # We then convert each Enum to its .value string so JSON can
            # serialize it (JSON doesn't know what an Enum is).
            row = asdict(record)
            row["camera_view"] = record.camera_view.value
            row["setting"]     = record.setting.value
            row["time_of_day"] = record.time_of_day.value
            row["weather"]     = record.weather.value
            row["label"]       = record.label.value
            f.write(json.dumps(row) + "\n")

        return True

    def append_many(self, records: list[ClipRecord]) -> tuple[int, int]:
        """
        Write multiple records, skipping duplicates.

        Returns (written_count, skipped_count).
        """
        written = skipped = 0
        for record in records:
            if self.append(record):
                written += 1
            else:
                skipped += 1
        return written, skipped
