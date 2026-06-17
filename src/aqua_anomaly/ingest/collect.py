"""
Search YouTube for pool footage and register clips in the manifest.

This script never downloads video. It uses yt-dlp in metadata-only mode
to find videos matching a search query, then writes a ClipRecord for each
result into the manifest JSONL file.
"""

from __future__ import annotations

import subprocess
import json
from pathlib import Path

from .blocklist import Blocklist
from .manifest import (
    ClipRecord,
    CameraView,
    Setting,
    TimeOfDay,
    Weather,
    Label,
    Manifest,
    make_clip_id,
)


# --- Default search queries --------------------------------------------------
# Each entry is a search string that biases toward a specific type of normal
# pool footage. Keeping these as a list makes it easy to add new search terms
# without touching the rest of the code.

DEFAULT_QUERIES = [
    "outdoor swimming pool long footage",
    "swimming pool overhead camera",
    "public pool surveillance footage",
    "competitive swimming pool side view",
    "outdoor pool sunny day swimmers",
]


# --- yt-dlp metadata fetch ---------------------------------------------------

def fetch_metadata(query: str, max_results: int = 10) -> list[dict]:
    """
    Ask yt-dlp to search YouTube and return video metadata.

    yt-dlp is run as a subprocess — a separate program launched from Python.
    We pass it flags that tell it: search YouTube, return JSON, don't download
    anything, and stop after max_results videos.

    Returns a list of raw metadata dicts, one per video found.
    """
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",  # "ytsearch10:..." means search YouTube for up to 10 results
        "--dump-json",                      # print metadata as JSON instead of downloading
        "--no-playlist",                    # treat each result as an individual video, not a playlist
        "--quiet",                          # suppress progress bars and status lines
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,   # capture both stdout (the JSON) and stderr (errors)
        text=True,             # return output as a string, not raw bytes
    )

    if result.returncode != 0:
        print(f"yt-dlp error for query '{query}':\n{result.stderr}")
        return []

    # yt-dlp with --dump-json prints one JSON object per line (not a list).
    # We split on newlines and parse each line individually.
    records = []
    for line in result.stdout.strip().splitlines():
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


# --- Metadata → ClipRecord ---------------------------------------------------

def metadata_to_record(meta: dict, guess_setting: Setting, guess_time_of_day: TimeOfDay, guess_weather: Weather) -> ClipRecord:
    """
    Convert a raw yt-dlp metadata dict into a ClipRecord.

    The condition fields (setting, time_of_day, weather) can't be reliably
    determined from metadata alone — they require watching the footage. We
    accept caller-provided guesses (based on the search query's intent) and
    mark the label as UNLABELED so the annotation pass knows to review it.

    These guesses must be verified manually before the clip is used for training.
    """
    url       = meta.get("webpage_url", meta.get("url", ""))
    duration  = float(meta.get("duration") or 0.0)
    platform  = meta.get("extractor", "unknown")  # yt-dlp calls this "extractor", e.g. "youtube"

    # We treat the full video as one clip: start at 0, end at full duration.
    # Later tooling can split long videos into segments if needed.
    start_sec = 0.0
    end_sec   = duration if duration > 0 else -1.0

    clip_id = make_clip_id(url, start_sec, end_sec)

    return ClipRecord(
        clip_id     = clip_id,
        source_url  = url,
        platform    = platform,
        start_sec   = start_sec,
        end_sec     = end_sec,
        camera_view = CameraView.UNKNOWN,   # can't determine from metadata
        setting     = guess_setting,
        time_of_day = guess_time_of_day,
        weather     = guess_weather,
        label       = Label.UNLABELED,      # must be reviewed before use
        notes       = meta.get("title", ""),  # store the video title as a note for the reviewer
    )


# --- Main collection entry point ---------------------------------------------

def collect(
    manifest_path: Path,
    queries: list[str] | None = None,
    max_results_per_query: int = 10,
    guess_setting: Setting = Setting.OUTDOOR,
    guess_time_of_day: TimeOfDay = TimeOfDay.DAY,
    guess_weather: Weather = Weather.UNKNOWN,
) -> None:
    """
    Search for pool footage and register results in the manifest.

    Parameters
    ----------
    manifest_path :
        Path to the JSONL manifest file. Created if it doesn't exist.
    queries :
        List of search strings. Defaults to DEFAULT_QUERIES if not provided.
    max_results_per_query :
        How many YouTube results to fetch per search string.
    guess_setting / guess_time_of_day / guess_weather :
        Condition metadata to attach to every clip found in this run.
        These are guesses based on search intent and must be verified manually.
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    manifest  = Manifest(manifest_path)
    blocklist = Blocklist()
    all_records: list[ClipRecord] = []
    blocked_count = 0

    for query in queries:
        print(f"Searching: '{query}'")
        results = fetch_metadata(query, max_results=max_results_per_query)
        print(f"  Found {len(results)} results")

        for meta in results:
            url = meta.get("webpage_url", meta.get("url", ""))
            if blocklist.contains(url):
                blocked_count += 1
                continue
            record = metadata_to_record(
                meta,
                guess_setting=guess_setting,
                guess_time_of_day=guess_time_of_day,
                guess_weather=guess_weather,
            )
            all_records.append(record)

    written, skipped = manifest.append_many(all_records)
    print(f"\nDone. {written} new clips registered, {skipped} duplicates skipped, {blocked_count} blocklisted.")
    print(f"Manifest: {manifest_path}")


# --- CLI entry point ---------------------------------------------------------
# This block only runs when you execute this file directly:
#   python -m aqua_anomaly.ingest.collect
# It does not run when the file is imported by another module.

if __name__ == "__main__":
    collect(
        manifest_path=Path("data/manifests/pool_footage.jsonl"),
    )
