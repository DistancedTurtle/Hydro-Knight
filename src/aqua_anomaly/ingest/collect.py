"""
Search YouTube for pool footage and register clips in the manifest.

This script never downloads video. It uses yt-dlp in metadata-only mode
to find videos matching a search query, then writes a ClipRecord for each
result into the manifest JSONL file.
"""

from __future__ import annotations

import subprocess
import json
import time
from pathlib import Path

# Seconds to wait between search queries to avoid YouTube rate limiting.
# 18 queries in rapid succession reliably triggers 400 errors.
QUERY_SLEEP_SEC = 3

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


# --- Search configuration ----------------------------------------------------

# Duration filter: only accept videos between these lengths.
# Too short = not enough swimming activity to be useful.
# Too long = primitive tech builds, construction timelapses, livestreams.
MIN_DURATION_SEC = 3 * 60    #  3 minutes
MAX_DURATION_SEC = 45 * 60   # 45 minutes

# Queries are grouped by intent so condition metadata can be guessed
# per-group when collect() is called in batches. Each string is written
# to avoid YouTube's popularity bias — specific, descriptive phrases
# surface recent or niche uploads rather than viral hits.
DEFAULT_QUERIES = [
    # Overhead / wide angle — most useful camera angle for pose detection
    "swimming pool overhead view people swimming",
    "pool deck camera angle swimmers laps",
    "aquatic center wide angle swim practice",

    # Outdoor recreational — the primary training distribution
    "outdoor public pool swimmers summer",
    "community pool open swim session",
    "backyard pool party swimming people",
    "hotel pool guests swimming vacation",

    # Indoor recreational
    "indoor lap pool swimmers training session",
    "ymca pool open swim",
    "leisure centre pool swimming",

    # Competitive — different body positions, useful for stroke variety
    "swim meet 50m pool side view",
    "masters swimming competition pool",
    "age group swim meet outdoor pool",

    # Lifeguard perspective — closest to surveillance camera angle
    "lifeguard stand view pool swimmers",
    "pool safety swim lesson children",

    # Varied conditions
    "outdoor pool cloudy day swimmers",
    "evening outdoor pool swimmers dusk",
    "crowded public pool summer swimmers",
]


# --- yt-dlp metadata fetch ---------------------------------------------------

def fetch_metadata(
    query: str,
    max_results: int = 10,
    min_duration: int = MIN_DURATION_SEC,
    max_duration: int = MAX_DURATION_SEC,
) -> list[dict]:
    """
    Ask yt-dlp to search YouTube and return video metadata.

    We fetch more results than requested (3x) to account for videos that
    will be filtered out by duration — then trim down to max_results after
    filtering. This avoids short queries returning almost nothing after
    duration filtering removes the bulk of results.

    Returns a list of raw metadata dicts, one per video found.
    """
    fetch_n = max_results * 3  # over-fetch to absorb duration filter losses

    cmd = [
        "yt-dlp",
        f"ytsearch{fetch_n}:{query}",
        "--dump-json",
        "--no-playlist",
        "--quiet",
        "--no-warnings",  # suppress nsig/throttling noise — errors still surface via returncode
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"yt-dlp error for query '{query}':\n{result.stderr}")
        return []

    all_results = []
    for line in result.stdout.strip().splitlines():
        if line:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Filter by duration. Videos outside the window are almost always
    # construction timelapses, music videos, or livestream archives.
    filtered = []
    for meta in all_results:
        duration = float(meta.get("duration") or 0.0)
        if min_duration <= duration <= max_duration:
            filtered.append(meta)
        if len(filtered) >= max_results:
            break

    return filtered


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

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(QUERY_SLEEP_SEC)
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
