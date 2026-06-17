"""
Download video files for clips registered in the manifest.

Videos are saved to raw_local/ which is gitignored — they never enter the repo.
A sidecar file (<clip_id>.done) is written next to each video on success so
re-running this script skips already-downloaded clips.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Ensure Homebrew binaries (ffmpeg) are visible to subprocesses even when
# the shell PATH isn't inherited by the venv.
_ENV = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}

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


def download_clip(record: ClipRecord, cookies_file: Path | None = None, cookies_from_browser: str | None = None) -> bool:
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
        "--format", "bestvideo[ext=mp4]/bestvideo", # best quality video-only, any format
        "--quiet",
        "--no-playlist",
    ]
    if cookies_file:
        cmd += ["--cookies", str(cookies_file)]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]

    # Trimming is handled non-destructively in the annotator — the full video
    # is downloaded and start_sec/end_sec in the manifest tell downstream
    # steps (pose extraction) which segment to actually process.

    result = subprocess.run(cmd, capture_output=True, text=True, env=_ENV)

    if result.returncode != 0:
        print(f"  Failed ({record.clip_id}): {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error'}")
        return False

    # Write the sidecar marker so future runs skip this clip.
    done_marker(record).touch()
    return True


def download(
    manifest_path: Path,
    limit: int | None = None,
    cookies_file: Path | None = None,
    cookies_from_browser: str | None = None,
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
        if download_clip(record, cookies_file=cookies_file, cookies_from_browser=cookies_from_browser):
            print(f"  OK → {local_path(record)}")
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone. {succeeded} downloaded, {failed} failed.")


if __name__ == "__main__":
    download(
        manifest_path=Path("data/manifests/pool_footage.jsonl"),
        limit=5,
        cookies_file=Path.home() / "Downloads/www.youtube.com_cookies.txt",
    )
