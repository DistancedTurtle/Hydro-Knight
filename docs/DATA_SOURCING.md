# Data sourcing & ethics

Hydro-Knight is **manifest-driven**: the repo commits *pointers to* data (source
URLs + metadata + annotations), never the video itself. Anyone reproducing the
work re-downloads from the manifest. This mirrors how datasets like Kinetics and
AVA are distributed.

## What is and isn't committed

| Committed (in git) | Never committed (gitignored) |
|---|---|
| `data/manifests/*.jsonl` — clip URLs, trim points, conditions, labels, events | Raw video (`raw_local/`, `*.mp4`, `*.mkv`, …) |
| `data/annotations/*` — annotation records | Extracted keypoint tables (`data/keypoints/`) |
| code, docs | Model weights (`*.pt`), scratch outputs (`scratch/`) |

## Why no video in the repo

- **Copyright** — public visibility on a platform is *not* a redistribution
  license. We may analyze footage we can lawfully access without acquiring the
  right to rehost it.
- **Consent & privacy** — pool footage shows identifiable people, often minors.
  Committing it would republish their images without consent. Even the figures
  in the README are deliberately pixelated to unrecognizability, with only the
  extracted skeletons drawn crisply.
- **Reproducibility & hygiene** — manifests are tiny and diff cleanly in git;
  multi-gigabyte video dumps do neither.

## Collection workflow

1. `ingest/collect.py` runs `yt-dlp` in **metadata-only** mode to search for
   varied pool footage and registers matches in the manifest (no download).
2. Condition metadata (`camera_view`, `setting`, `time_of_day`, `weather`) is
   auto-filled as a *search-intent guess* and must be verified in a manual
   review pass — it is not ground truth until a human confirms it.
3. `ingest/download.py` fetches the actual video into the gitignored
   `raw_local/` for local processing only.
4. `ingest/register_local.py` registers self-recorded / local clips the same way.

## Labeling

**Outcome-based retrospective labeling:** if the swimmer surfaces → negative; if
guards intervene or someone is pulled out → positive. The footage answers the
question after the fact, avoiding ambiguous mid-event judgment calls. Anomaly
events are marked as typed time windows (`distress` / `submerged` / `face_down`)
in each clip's `events` field; frames outside those windows on an anomaly clip
are still reusable as normal training data.

## Positive (anomaly) examples — in priority order

1. Consented research datasets (e.g. the Figshare underwater drowning dataset)
2. Self-recorded simulations with consenting participants
3. Proxy footage (breath-hold freediving, lifeguard training drills)

See [PLAN.md](../PLAN.md) for how this data feeds the model build progression.
