#!/usr/bin/env python3
"""Blender rendering helper with caching and progress reporting.

This wraps Blender invocation so that:
  * renders are cached under a hashed key (including dynamic timestamps);
  * progress is periodically printed by counting rendered frames;
  * Blender stdout/stderr remain suppressed to keep SLURM logs readable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"}
METADATA_FILENAME = "_metadata.json"


def canonicalize_blender_args(raw_args: Iterable[str]) -> List[str]:
    """Normalize Blender CLI arguments so cache keys ignore output-specific paths."""
    raw_list = list(raw_args)
    canonical: List[str] = []
    i = 0
    while i < len(raw_list):
        token = raw_list[i]
        next_token = raw_list[i + 1] if i + 1 < len(raw_list) else None

        def append_placeholder(flag: str, placeholder: str) -> None:
            canonical.append(flag)
            canonical.append(placeholder)

        if token in {"--results", "--output"}:
            append_placeholder(token, "__OUTPUT_DIR__")
            if next_token is not None and not next_token.startswith("--"):
                i += 2
            else:
                i += 1
            continue

        if token == "--transforms-json":
            append_placeholder(token, "__TRANSFORMS__")
            if next_token is not None and not next_token.startswith("--"):
                i += 2
            else:
                i += 1
            continue

        if token.startswith("--results="):
            canonical.append("--results=__OUTPUT_DIR__")
            i += 1
            continue

        if token.startswith("--transforms-json="):
            canonical.append("--transforms-json=__TRANSFORMS__")
            i += 1
            continue

        canonical.append(token)
        i += 1

    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blender render wrapper with caching")
    parser.add_argument("--blend-file", required=True, help="Path to the .blend file")
    parser.add_argument(
        "--render-script",
        required=True,
        help="Path to the Blender Python script (render_blender.py)",
    )
    parser.add_argument(
        "--render-dir",
        required=True,
        help="Output directory where Blender writes rendered frames",
    )
    parser.add_argument(
        "--transforms-json",
        help="Transforms JSON used for hashing dynamic trajectories (optional)",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--n-train-views", type=int, required=True)
    parser.add_argument("--animation-views", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--blender-arg",
        action="append",
        default=[],
        help="Argument to pass through to Blender after the script '--' delimiter. "
        "Specify multiple times to add multiple arguments.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=20,
        help="Seconds between progress checks while Blender runs",
    )
    return parser.parse_args()


def compute_cache_key(args: argparse.Namespace, canonical_blender_args: List[str]) -> str:
    payload = {
        "blender_args": canonical_blender_args,
        "frames": [],
    }

    transforms_path = args.transforms_json
    if transforms_path:
        transforms_file = Path(transforms_path)
        if transforms_file.exists():
            try:
                transforms = json.loads(transforms_file.read_text())
            except json.JSONDecodeError:
                transforms = {}
            frames = transforms.get("frames", [])
            for frame in frames:
                payload["frames"].append(
                    {
                        "time": frame.get("time"),
                        "file_path": frame.get("file_path"),
                        "camera_name": frame.get("camera_name"),
                    }
                )
        else:
            payload["frames"].append({"time": None, "missing": True})
    else:
        payload["frames"].append({"time": None, "transforms": False})

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
    return digest[:16]


def count_rendered_frames(render_dir: Path) -> int:
    if not render_dir.exists():
        return 0
    count = 0
    for path in render_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def copy_tree(src: Path, dst: Path, *, ignore_metadata: bool = False) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = None
    if ignore_metadata:
        ignore = shutil.ignore_patterns(METADATA_FILENAME)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def ensure_cache_hit(cache_dir: Path, render_dir: Path, expected_frames: int) -> bool:
    if not cache_dir.exists():
        return False
    cached_frames = count_rendered_frames(cache_dir)
    if cached_frames < expected_frames:
        return False
    print(f">>> Using cached Blender renders from {cache_dir}")
    copy_tree(cache_dir, render_dir, ignore_metadata=True)
    percent = min(100, int(round(100 * expected_frames / max(1, expected_frames))))
    print(
        f"    Blender progress: {expected_frames}/{expected_frames} frames ({percent}%) [cache]",
        flush=True,
    )
    return True


def run_blender_with_progress(
    args: argparse.Namespace,
    blend_file: Path,
    render_script: Path,
    render_dir: Path,
    expected_frames: int,
) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "blender",
        "-b",
        str(blend_file),
        "-P",
        str(render_script),
        "--",
    ] + args.blender_arg

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    last_reported = -1
    poll_interval = max(1, args.progress_interval)

    try:
        while True:
            retcode = process.poll()
            frame_count = count_rendered_frames(render_dir)
            if frame_count != last_reported:
                capped = min(frame_count, expected_frames)
                percent = int(round(100 * capped / max(1, expected_frames)))
                print(
                    f"    Blender progress: {capped}/{expected_frames} frames ({percent}%)",
                    flush=True,
                )
                last_reported = frame_count
            if retcode is not None:
                break
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        process.terminate()
        process.wait(timeout=10)
        raise

    if process.returncode != 0:
        raise RuntimeError(
            f"Blender render failed with exit code {process.returncode} "
            f"while running: {' '.join(cmd)}"
        )

    # Final progress emit (in case Blender finished faster than the loop noticed)
    final_count = count_rendered_frames(render_dir)
    capped = min(final_count, expected_frames)
    percent = int(round(100 * capped / max(1, expected_frames)))
    print(
        f"    Blender progress: {capped}/{expected_frames} frames ({percent}%)",
        flush=True,
    )


def populate_cache(cache_dir: Path, render_dir: Path, metadata: dict) -> None:
    tmp_dir = cache_dir.parent / f".tmp_cache_{cache_dir.name}_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(render_dir, tmp_dir, dirs_exist_ok=True)
    metadata_path = tmp_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)
    print(f">>> Cached Blender renders stored at {cache_dir}")


def main() -> int:
    args = parse_args()

    blend_file = Path(args.blend_file).resolve()
    render_script = Path(args.render_script).resolve()
    render_dir = Path(args.render_dir).resolve()
    cache_root = Path(args.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    if not blend_file.exists():
        raise FileNotFoundError(f"Blend file not found: {blend_file}")
    if not render_script.exists():
        raise FileNotFoundError(f"Render script not found: {render_script}")

    canonical_blender_args = canonicalize_blender_args(args.blender_arg)
    cache_key = compute_cache_key(args, canonical_blender_args)
    cache_dir = cache_root / cache_key

    if ensure_cache_hit(cache_dir, render_dir, args.expected_frames):
        return 0

    run_blender_with_progress(
        args,
        blend_file=blend_file,
        render_script=render_script,
        render_dir=render_dir,
        expected_frames=args.expected_frames,
    )

    metadata = {
        "cache_key": cache_key,
        "dataset": args.dataset_name,
        "blend_file": str(blend_file),
        "render_script": str(render_script),
        "transforms_json": str(args.transforms_json) if args.transforms_json else None,
        "expected_frames": args.expected_frames,
        "original_blender_args": args.blender_arg,
        "canonical_blender_args": canonical_blender_args,
        "generated_at": time.time(),
    }

    populate_cache(cache_dir, render_dir, metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
