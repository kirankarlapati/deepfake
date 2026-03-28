import argparse
import os
import random
import re
import subprocess
import sys
from pathlib import Path

try:
    import gdown
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: gdown. Install with 'pip install gdown'."
    ) from exc

DEFAULT_REPO_URL = "https://github.com/DASH-Lab/FakeAVCeleb"
DEFAULT_REPO_DIR = "FakeAVCeleb"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv")

FAKE_KEYWORDS = (
    "fake",
    "deepfake",
    "faceswap",
    "face_swap",
    "facereenact",
    "fsgan",
    "neural",
)
REAL_KEYWORDS = (
    "real",
    "original",
    "authentic",
)

DRIVE_URL_RE = re.compile(
    r"https?://(?:drive|docs)\.google\.com/[^\s\)\]]+",
    re.IGNORECASE,
)
FILENAME_RE = re.compile(r"[\w\-.]+\.(?:mp4|avi|mov|mkv|webm|flv|wmv)", re.IGNORECASE)


def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Command failed")
    return result.stdout.strip()


def ensure_repo(repo_url: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    run_cmd(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])


def extract_drive_id(url: str) -> str | None:
    # Common patterns: /file/d/<id>/, id=<id>
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


def normalize_drive_url(url: str) -> str | None:
    if "folders" in url:
        return None
    file_id = extract_drive_id(url)
    if not file_id:
        return None
    return f"https://drive.google.com/uc?id={file_id}"


def guess_label(text: str) -> str:
    lowered = text.lower()
    if any(key in lowered for key in FAKE_KEYWORDS):
        return "fake"
    if any(key in lowered for key in REAL_KEYWORDS):
        return "real"
    return "unknown"


def guess_filename(line: str, url: str) -> str:
    match = FILENAME_RE.search(line)
    if match:
        return match.group(0)
    file_id = extract_drive_id(url) or "video"
    return f"{file_id}.mp4"


def collect_links(repo_dir: Path):
    candidates = {}
    for path in repo_dir.rglob("*"):
        if path.is_dir():
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        if path.suffix.lower() not in {".md", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".lst"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in content.splitlines():
            for url in DRIVE_URL_RE.findall(line):
                normalized = normalize_drive_url(url)
                if not normalized:
                    continue
                label = guess_label(f"{line} {path}")
                filename = guess_filename(line, url)
                key = normalized
                if key not in candidates:
                    candidates[key] = {
                        "url": normalized,
                        "label": label,
                        "filename": filename,
                        "source": str(path),
                    }
    return list(candidates.values())


def select_balanced(candidates, count: int, seed: int | None):
    rng = random.Random(seed)
    real = [c for c in candidates if c["label"] == "real"]
    fake = [c for c in candidates if c["label"] == "fake"]
    unknown = [c for c in candidates if c["label"] == "unknown"]

    rng.shuffle(real)
    rng.shuffle(fake)
    rng.shuffle(unknown)

    target_real = min(len(real), count // 2)
    target_fake = min(len(fake), count // 2)

    selected = real[:target_real] + fake[:target_fake]

    remaining = count - len(selected)
    if remaining > 0:
        pool = unknown + real[target_real:] + fake[target_fake:]
        rng.shuffle(pool)
        selected.extend(pool[:remaining])

    rng.shuffle(selected)
    return selected


def download_videos(items, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(items)
    for idx, item in enumerate(items, start=1):
        filename = item["filename"].replace("/", "_").replace("\\", "_")
        target_path = output_dir / filename
        if target_path.exists():
            print(f"[{idx}/{total}] Skipping existing: {target_path.name}")
            continue
        print(f"[{idx}/{total}] Downloading: {target_path.name}")
        try:
            result = gdown.download(
                item["url"],
                output=str(target_path),
                quiet=False,
                fuzzy=True,
                resume=True,
            )
        except Exception as exc:  # gdown can raise on quota or network issues
            message = str(exc).lower()
            if "quota" in message or "exceed" in message:
                print("  Quota error detected. Skipping this file for now.")
                continue
            print(f"  Download failed: {exc}")
            continue

        if not result:
            print("  Download failed or blocked (possible quota issue).")
        else:
            print("  Done")


def main():
    parser = argparse.ArgumentParser(
        description="Download a small subset of FakeAVCeleb videos using gdown.",
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="dataset_videos")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    ensure_repo(args.repo_url, repo_dir)

    candidates = collect_links(repo_dir)
    if not candidates:
        print("No Google Drive video links found in the repository.")
        sys.exit(1)

    selected = select_balanced(candidates, args.count, args.seed)
    print(f"Found {len(candidates)} links. Selected {len(selected)} for download.")

    download_videos(selected, Path(args.output))


if __name__ == "__main__":
    main()
