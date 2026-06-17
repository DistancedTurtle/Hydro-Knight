"""
Download video files for clips registered in the manifest.

Videos are saved to raw_local/ which is gitignored — they never enter the repo.
A sidecar file (<clip_id>.done) is written next to each video on success so
re-running this script skips already-downloaded clips.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .manifest import ClipRecord, Manifest


# Where downloaded videos land. Gitignored.
RAW_LOCAL = Path("raw_local")


def local_path(record: ClipRecord) -> Path:
    """
    Return the expected local file path for a clip.

    We name the file by clip_id so the filename is stable and unique
    regardless of what the original video was called on YouTube.
    The extension is .mp4 — yt-dlp will convert to this format on download.
    """
    return RAW_LOCAL / f"{record.clip_id}.mp4"


def done_marker(record: ClipRecord) -> Path:
    """
    Return the path to the sidecar file that marks a clip as downloaded.

    We use a separate .done file rather than checking whether the .mp4 exists
    because a partial download leaves a real (but broken) .mp4 behind.
    The .done file is only written after yt-dlp exits successfully.
    """
    return RAW_LOCAL / f"{record.clip_id}.done"


def is_downloaded(record: ClipRecord) -> bool:
    return done_marker(record).exists()


def download_clip(record: ClipRecord) -> bool:
    """
    Download one clip to raw_local/.

    If start_sec and end_sec are set, yt-dlp downloads only that segment
    rather than the full video — saves disk space for long source videos.

    Returns True on success, False on failure.
    """
    RAW_LOCAL.mkdir(parents=True, exist_ok=True)
    out_path = local_path(record)

    cmd = [
        "yt-dlp",
        record.source_url,
        "--output", str(out_path),
        "--format", "bestvideo[ext=mp4]/bestvideo/mp4",  # video only, no audio needed

        "--quiet",
        "--no-playlist",
    ]

    # If the clip is a segment of a longer video, add time range flags.
    # -1.0 end_sec means "to the end of the video", so we skip the flag.
    if record.start_sec > 0 or record.end_sec > 0:
        cmd += ["--download-sections", f"*{record.start_sec}-{record.end_sec}"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Failed ({record.clip_id}): {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error'}")
        return False

    # Write the sidecar marker so future runs skip this clip.
    done_marker(record).touch()
    return True


def download(
    manifest_path: Path,
    limit: int | None = None,
) -> None:
    """
    Download all undownloaded clips in the manifest.

    Parameters
    ----------
    manifest_path :
        Path to the JSONL manifest file to read from.
    limit :
        If set, download at most this many clips. Useful for test runs.
    """
    manifest = Manifest(manifest_path)
    records = manifest.load()

    pending = [r for r in records if not is_downloaded(r)]

    if not pending:
        print("All clips already downloaded.")
        return

    if limit is not None:
        pending = pending[:limit]

    print(f"{len(pending)} clips to download.")

    succeeded = failed = 0
    for i, record in enumerate(pending, start=1):
        title = record.notes[:60] if record.notes else record.clip_id
        print(f"[{i}/{len(pending)}] {title}")
        if download_clip(record):
            print(f"  OK → {local_path(record)}")
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone. {succeeded} downloaded, {failed} failed.")


if __name__ == "__main__":
    download(
        manifest_path=Path("data/manifests/pool_footage.jsonl"),
        limit=5,  # only grab 5 clips on a manual run to avoid a huge download
    )
