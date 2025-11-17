#!/usr/bin/env python3
"""
Generate dual_summary_table.tex by parsing results.json files from dual training outputs.

This script scans output/multipleview/<scene>/<camera_type>/ directories and aggregates
metrics from combined (multi-camera) and filtered (monocular) training runs.

Usage:
    python scripts/generate_dual_summary_table.py [--output dual_summary_table.tex]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics


def find_latest_method(results: Dict) -> Optional[str]:
    """Find the latest method key (e.g., 'ours_24000') in results.json."""
    method_keys = [k for k in results.keys() if k.startswith("ours_")]
    if not method_keys:
        return None
    
    # Extract iteration numbers and return the highest
    def get_iteration(key: str) -> int:
        try:
            return int(key.split("_")[-1])
        except ValueError:
            return 0
    
    return max(method_keys, key=get_iteration)


def load_metrics(results_path: Path) -> Optional[Dict[str, float]]:
    """Load metrics from a results.json file."""
    if not results_path.exists():
        return None
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        method = find_latest_method(results)
        if method is None:
            return None
        
        return results[method]
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Warning: Failed to load {results_path}: {e}")
        return None


def scan_scenes(base_dir: Path = Path("output/multipleview")) -> Dict[str, Dict[str, Dict[str, Dict]]]:
    """
    Scan output directory structure and collect metrics.
    
    Returns:
        Dict mapping camera_type -> scene -> run_type -> metrics
        camera_type: 'iphone' or 'stereo'
        scene: scene name (e.g., 'ball', 'coffee')
        run_type: 'combined' or 'filtered'
    """
    data: Dict[str, Dict[str, Dict[str, Dict]]] = {
        'iphone': {},
        'stereo': {}
    }
    
    if not base_dir.exists():
        print(f"Warning: Base directory {base_dir} does not exist")
        return data
    
    # Scan for scene directories
    for scene_dir in base_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        
        # Check for iphone and stereo subdirectories
        for camera_type in ['iphone', 'stereo']:
            camera_dir = scene_dir / camera_type
            if not camera_dir.exists():
                continue
            
            data[camera_type][scene_name] = {}
            
            # Check for combined and filtered runs
            for run_type in ['combined', 'filtered']:
                run_dir = camera_dir / run_type
                if not run_dir.exists():
                    continue
                
                results_path = run_dir / "results.json"
                metrics = load_metrics(results_path)
                
                if metrics is not None:
                    data[camera_type][scene_name][run_type] = metrics
    
    return data


def compute_averages(data: Dict[str, Dict[str, Dict[str, Dict]]], 
                     require_both: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute average metrics across scenes for each camera type and run type.
    
    Args:
        require_both: If True, only include scenes that have both combined and filtered runs.
    
    Returns:
        Dict mapping camera_type -> run_type -> metric_name -> average_value
    """
    averages: Dict[str, Dict[str, Dict[str, float]]] = {
        'iphone': {'combined': {}, 'filtered': {}},
        'stereo': {'combined': {}, 'filtered': {}}
    }
    
    for camera_type in ['iphone', 'stereo']:
        for run_type in ['combined', 'filtered']:
            # Collect all values for each metric across scenes
            metric_values: Dict[str, List[float]] = {}
            
            for scene_name, runs in data[camera_type].items():
                # Skip if scene doesn't have the required run type
                if run_type not in runs:
                    continue
                
                # If require_both, skip scenes that don't have both run types
                if require_both and ('combined' not in runs or 'filtered' not in runs):
                    continue
                
                metrics = runs[run_type]
                for metric_name, value in metrics.items():
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(value)
            
            # Compute averages
            for metric_name, values in metric_values.items():
                if values:
                    averages[camera_type][run_type][metric_name] = statistics.mean(values)
    
    return averages


def generate_latex_table(averages: Dict[str, Dict[str, Dict[str, float]]], 
                         output_path: Path) -> None:
    """Generate LaTeX table in the format of dual_summary_table.tex."""
    
    # Extract metrics
    iphone_mono = averages['iphone']['filtered']
    iphone_multi = averages['iphone']['combined']
    stereo_mono = averages['stereo']['filtered']
    stereo_multi = averages['stereo']['combined']
    
    # Format values (use LPIPS-vgg for LPIPS)
    def fmt(val: float, decimals: int = 2) -> str:
        return f"{val:.{decimals}f}"
    
    iphone_mono_psnr = fmt(iphone_mono.get('PSNR', 0.0), 2)
    iphone_mono_ssim = fmt(iphone_mono.get('SSIM', 0.0), 3)
    iphone_mono_lpips = fmt(iphone_mono.get('LPIPS-vgg', 0.0), 3)
    
    iphone_multi_psnr = fmt(iphone_multi.get('PSNR', 0.0), 2)
    iphone_multi_ssim = fmt(iphone_multi.get('SSIM', 0.0), 3)
    iphone_multi_lpips = fmt(iphone_multi.get('LPIPS-vgg', 0.0), 3)
    
    stereo_mono_psnr = fmt(stereo_mono.get('PSNR', 0.0), 2)
    stereo_mono_ssim = fmt(stereo_mono.get('SSIM', 0.0), 3)
    stereo_mono_lpips = fmt(stereo_mono.get('LPIPS-vgg', 0.0), 3)
    
    stereo_multi_psnr = fmt(stereo_multi.get('PSNR', 0.0), 2)
    stereo_multi_ssim = fmt(stereo_multi.get('SSIM', 0.0), 3)
    stereo_multi_lpips = fmt(stereo_multi.get('LPIPS-vgg', 0.0), 3)
    
    latex_content = f"""\\begin{{table}}[ht]
\\centering
\\begin{{tabular}}{{lrrr}}
\\toprule
Method & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ \\\\
\\midrule
\\multicolumn{{4}}{{l}}{{\\textbf{{iPhone Capture}}}} \\\\
Monocular & {iphone_mono_psnr} & {iphone_mono_ssim} & {iphone_mono_lpips} \\\\
iPhone & {iphone_multi_psnr} & {iphone_multi_ssim} & {iphone_multi_lpips} \\\\
\\midrule
\\addlinespace[4pt]
\\multicolumn{{4}}{{l}}{{\\textbf{{Stereo Capture}}}} \\\\
Monocular & {stereo_mono_psnr} & {stereo_mono_ssim} & {stereo_mono_lpips} \\\\
Stereo & {stereo_multi_psnr} & {stereo_multi_ssim} & {stereo_multi_lpips} \\\\
\\bottomrule
\\end{{tabular}}%
\\caption{{\\textbf{{Real, dynamic results.}} We compare iPhone and stereo cameras to their monocular baselines on dynamic 4DGS reconstruction.}}
\\label{{tab:dynamic-real-summary}}
\\end{{table}}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"Generated {output_path}")
    print(f"\nSummary:")
    print(f"  iPhone Capture:")
    print(f"    Monocular: PSNR={iphone_mono_psnr}, SSIM={iphone_mono_ssim}, LPIPS={iphone_mono_lpips}")
    print(f"    iPhone:    PSNR={iphone_multi_psnr}, SSIM={iphone_multi_ssim}, LPIPS={iphone_multi_lpips}")
    print(f"  Stereo Capture:")
    print(f"    Monocular: PSNR={stereo_mono_psnr}, SSIM={stereo_mono_ssim}, LPIPS={stereo_mono_lpips}")
    print(f"    Stereo:    PSNR={stereo_multi_psnr}, SSIM={stereo_multi_ssim}, LPIPS={stereo_multi_lpips}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dual_summary_table.tex from dual training results"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('dual_summary_table.tex'),
        help='Output LaTeX file path (default: dual_summary_table.tex)'
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('output/multipleview'),
        help='Base directory to scan for results (default: output/multipleview)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose information about scanned scenes'
    )
    parser.add_argument(
        '--allow-incomplete',
        action='store_true',
        help='Include scenes that only have combined or filtered (not both)'
    )
    
    args = parser.parse_args()
    
    print(f"Scanning {args.base_dir} for dual training results...")
    data = scan_scenes(args.base_dir)
    
    if args.verbose:
        print("\nScanned data:")
        for camera_type in ['iphone', 'stereo']:
            print(f"\n  {camera_type}:")
            for scene_name, runs in data[camera_type].items():
                print(f"    {scene_name}: {list(runs.keys())}")
    
    # Check if we have any data
    total_scenes = sum(len(scenes) for scenes in data.values())
    if total_scenes == 0:
        print(f"\nError: No results found in {args.base_dir}")
        print("Expected structure: output/multipleview/<scene>/<camera_type>/<combined|filtered>/results.json")
        return 1
    
    print(f"\nFound results for {total_scenes} scene(s)")
    
    # Count scenes with both run types
    if not args.allow_incomplete:
        complete_scenes = {}
        for camera_type in ['iphone', 'stereo']:
            complete_scenes[camera_type] = sum(
                1 for runs in data[camera_type].values()
                if 'combined' in runs and 'filtered' in runs
            )
        print(f"Scenes with both combined and filtered:")
        print(f"  iPhone: {complete_scenes['iphone']}")
        print(f"  Stereo: {complete_scenes['stereo']}")
    
    averages = compute_averages(data, require_both=not args.allow_incomplete)
    
    generate_latex_table(averages, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())

