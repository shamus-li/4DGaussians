#!/usr/bin/env python3
"""
Generate LaTeX tables summarising dual-camera training metrics.

The script scans `output/multipleview/<scene>/<capture>/<mode>/results.json`
and produces:
  * dual_summary_table.tex  – aggregated metrics per capture type.
  * dual_per_scene_tables.tex – per-scene PSNR/SSIM/LPIPS tables.

Usage (from repo root):
    python scripts/generate_dual_tex_tables.py

You can override the base directory, scene list, or output paths via CLI flags.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_SCENES = ["ball", "coffee", "orange", "roll", "spinner", "sugar"]

# Preserve row order in the output tables.
METHODS: "OrderedDict[str, Tuple[str, str]]" = OrderedDict(
    [
        ("Monocular (iPhone)", ("iphone", "filtered")),
        ("iPhone", ("iphone", "combined")),
        ("Monocular (Stereo)", ("stereo", "filtered")),
        ("Stereo", ("stereo", "combined")),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate LaTeX performance tables for dual-camera experiments."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/multipleview"),
        help="Root directory containing per-scene experiment folders.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Scene folder names under --base-dir to include in the tables.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("dual_summary_table.tex"),
        help="Path to write the summary LaTeX table.",
    )
    parser.add_argument(
        "--per-scene-output",
        type=Path,
        default=Path("dual_per_scene_tables.tex"),
        help="Path to write the per-scene LaTeX tables.",
    )
    return parser.parse_args()


def scene_label(scene: str) -> str:
    """Convert a scene folder name to a display label."""
    return scene.replace("_", " ").title().replace(" ", "")


def load_metrics(path: Path) -> Dict[str, float]:
    """Read the best metrics entry from a results.json file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not data:
        raise ValueError(f"No entries found in {path}")

    def key_score(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    best_key = max(data.keys(), key=key_score)
    return data[best_key]


def gather_metrics(
    base_dir: Path, scenes: Iterable[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collect metrics per method and scene."""
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        method: {} for method in METHODS
    }
    for scene in scenes:
        scene_dir = base_dir / scene
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

        for method, (capture, mode) in METHODS.items():
            results_path = scene_dir / capture / mode / "results.json"
            metrics[method][scene] = load_metrics(results_path)
    return metrics


def mean_and_std(values: List[float]) -> Tuple[float, float]:
    avg = sum(values) / len(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return avg, std


def format_value(metric: str, value: float) -> str:
    if metric == "PSNR":
        return f"{value:.2f}"
    return f"{value:.3f}"


def build_per_scene_table(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    scenes: List[str],
    per_scene_output: Path,
) -> None:
    scene_headers = [scene_label(scene) for scene in scenes]
    columns = "l" + "c" * (len(scene_headers) + 2)
    sections = [
        ("PSNR$\\uparrow$", "PSNR"),
        ("SSIM$\\uparrow$", "SSIM"),
        ("LPIPS$\\downarrow$", "LPIPS-alex"),
    ]

    lines: List[str] = [r"\begin{table*}[ht]", r"\centering", ""]
    for idx, (title, metric_key) in enumerate(sections):
        lines.append(rf"\textbf{{{title}}}")
        lines.append(rf"\begin{{tabular}}{{{columns}}}")
        lines.append(r"\hline")
        header = "Method & " + " & ".join(scene_headers) + " & Avg & Std \\\\"
        lines.append(header)
        lines.append(r"\hline")

        for method in METHODS:
            values = [
                metrics[method][scene][metric_key] for scene in scenes
            ]
            avg, std = mean_and_std(values)
            row = [method]
            row.extend(format_value(metric_key.split("-")[0], v) for v in values)
            row.append(format_value(metric_key.split("-")[0], avg))
            row.append(format_value(metric_key.split("-")[0], std))
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        if idx < len(sections) - 1:
            lines.extend(["", r"\vspace{1em}", ""])

    lines.append(
        r"\caption{Per-scene metrics for dual-training evaluation on real dynamic scenes.}"
    )
    lines.append(r"\label{tab:dynamic-real-per-scene}")
    lines.append(r"\end{table*}")
    per_scene_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_table(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    scenes: List[str],
    summary_output: Path,
) -> None:
    averages: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        averages[method] = {}
        for metric_key in ("PSNR", "SSIM", "LPIPS-alex"):
            values = [metrics[method][scene][metric_key] for scene in scenes]
            avg, _ = mean_and_std(values)
            averages[method][metric_key] = avg

    sections = [
        ("iPhone Capture", [("Monocular", "Monocular (iPhone)"), ("iPhone", "iPhone")]),
        ("Stereo Capture", [("Monocular", "Monocular (Stereo)"), ("Stereo", "Stereo")]),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\",
        r"\midrule",
    ]

    for section_idx, (section_label, rows) in enumerate(sections):
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{section_label}}}}} \\")
        for label, method_name in rows:
            psnr = format_value("PSNR", averages[method_name]["PSNR"])
            ssim = format_value("SSIM", averages[method_name]["SSIM"])
            lpips = format_value("LPIPS", averages[method_name]["LPIPS-alex"])
            lines.append(f"{label} & {psnr} & {ssim} & {lpips} \\\\")
        if section_idx == 0:
            lines.append(r"\midrule")
            lines.append(r"\addlinespace[4pt]")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"\caption{\textbf{Real, dynamic results.} We compare iPhone and stereo cameras to their monocular baselines on dynamic 4DGS reconstruction.}",
            r"\label{tab:dynamic-real-summary}",
            r"\end{table}",
        ]
    )
    summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scenes = list(args.scenes)
    metrics = gather_metrics(args.base_dir, scenes)
    build_per_scene_table(metrics, scenes, args.per_scene_output)
    build_summary_table(metrics, scenes, args.summary_output)
    print(f"Wrote {args.per_scene_output}")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
