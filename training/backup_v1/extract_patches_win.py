"""
extract_patches_win.py - Extract cross-point patches from Gomrade dataset
================================================================
Uses built-in perspective matrices and grid coordinates from the Gomrade 
dataset to automatically extract patches for each cross-point, 
classifying them into B/W/E (Black/White/Empty).

No manual labeling required! All coordinates and labels are provided by the dataset.

Usage:
  python extract_patches_win.py                       # Use default config
  python extract_patches_win.py --config config_windows.yaml  # Specify config
  python extract_patches_win.py --gomrade D:/path/to/Gomrade-dataset1
  python extract_patches_win.py --debug-session session_name  # Visual debug
"""

import cv2
import numpy as np
import os
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm


# --------------------------------------------------------------
# Configuration Loading
# --------------------------------------------------------------
def load_config(config_path=None):
    if config_path is None:
        actual_path = Path("config_windows.yaml")
    else:
        actual_path = Path(config_path)
    
    if not actual_path.exists():
        return {}
    # Use UTF-8 encoding for Windows compatibility
    with open(actual_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_param(config, *keys, default=None):
    val = config
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val if val is not None else default


# --------------------------------------------------------------
# Gomrade YML Parsing
# --------------------------------------------------------------
def parse_board_txt(txt_path):
    """Parse 19x19 board state file. Returns list[list[str]] or None."""
    state = []
    # Use UTF-8 encoding for Windows compatibility
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            row = [c if c in ('B', 'W') else 'E' for c in line.strip().split()]
            if len(row) == 19:
                state.append(row)
    return state if len(state) == 19 else None


def parse_classifier_state_yml(yml_path):
    """
    Parse board_state_classifier_state.yml.
    Returns x_grid (len=19), y_grid (len=19).
    """
    # Use UTF-8 encoding for Windows compatibility
    with open(yml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    x_grid = data.get("x_grid", [])
    y_grid = data.get("y_grid", [])
    return x_grid, y_grid


def compute_perspective_matrix(yml_path, warp_w, warp_h):
    """
    Get perspective transformation matrix M from board_extractor_state.yml.
    """
    # Use UTF-8 encoding for Windows compatibility
    with open(yml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "M" in data:
        return np.array(data["M"], dtype=np.float64)

    if "pts_clicks" in data:
        pts = np.array(data["pts_clicks"], dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError(f"pts_clicks format error: shape={pts.shape}")
        # pts_clicks order: top-left -> top-right -> bottom-right -> bottom-left (clockwise)
        dst = np.float32([
            [0,         0         ],
            [warp_w - 1, 0        ],
            [warp_w - 1, warp_h -1],
            [0,         warp_h - 1],
        ])
        return cv2.getPerspectiveTransform(pts, dst)

    raise KeyError("board_extractor_state.yml contains neither M nor pts_clicks")


# --------------------------------------------------------------
# Patch Extraction
# --------------------------------------------------------------
def extract_patch(img, x, y, radius):
    """Crop 2*radius x 2*radius patch at (x,y), with boundary safety."""
    h, w = img.shape[:2]
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius), min(h, y + radius)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]


def process_session(session_dir, output_dir, patch_size, stats, debug=False):
    """
    Process one game session folder in Gomrade dataset.
    Returns number of patches extracted, -1 if skipped.
    """
    session_dir = Path(session_dir)
    ext_yml = session_dir / "board_extractor_state.yml"
    cls_yml = session_dir / "board_state_classifier_state.yml"

    if not ext_yml.exists() or not cls_yml.exists():
        return -1  # Silently skip if missing essential files

    try:
        # -- Step 1: Read grid coordinates --
        x_grid, y_grid = parse_classifier_state_yml(cls_yml)

        if len(x_grid) != 19 or len(y_grid) != 19:
            return -1

        # Infer image dimensions from grid coordinates
        # x_grid[0] is often > 0 if there's a margin
        warp_w = x_grid[-1] + x_grid[0] + 1
        warp_h = y_grid[-1] + y_grid[0] + 1

        # -- Step 2: Get perspective matrix --
        M = compute_perspective_matrix(ext_yml, warp_w, warp_h)

    except Exception as e:
        tqdm.write(f"  [Skip] {session_dir.name}: Parse failed ({e})")
        return 0

    # Stone radius = ~42% of grid spacing
    step_x = (x_grid[-1] - x_grid[0]) / 18
    step_y = (y_grid[-1] - y_grid[0]) / 18
    radius = max(8, int(min(step_x, step_y) * 0.42))

    # Find all .txt annotation files (one per frame)
    txt_files = sorted(session_dir.glob("*.txt"))
    count = 0

    for txt_path in txt_files:
        # Find corresponding image (.png preferred, then .jpg)
        img_path = txt_path.with_suffix(".png")
        if not img_path.exists():
            img_path = txt_path.with_suffix(".jpg")
        if not img_path.exists():
            continue

        board_state = parse_board_txt(txt_path)
        if board_state is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Perspective warp
        try:
            warped = cv2.warpPerspective(img, M, (warp_w, warp_h))
        except Exception:
            continue

        # -- Debug mode: save image with grid visualization --
        if debug:
            _save_debug_visualization(warped, x_grid, y_grid, board_state,
                                      session_dir.name, txt_path.stem)
            debug = False  # Only save first frame per session

        # Extract patch for each cross-point
        for row in range(19):
            for col in range(19):
                label = board_state[row][col]  # 'B', 'W', or 'E'
                x = int(x_grid[col])
                y = int(y_grid[row])

                patch = extract_patch(warped, x, y, radius)
                if patch is None:
                    continue

                patch = cv2.resize(patch, (patch_size, patch_size))

                cls_dir = output_dir / label
                cls_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{session_dir.name}_{txt_path.stem}_r{row:02d}_c{col:02d}.jpg"
                cv2.imwrite(str(cls_dir / fname), patch,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])

                stats[label] += 1
                count += 1

    return count


def _save_debug_visualization(warped, x_grid, y_grid, board_state, session_name, frame_stem):
    """Overlay grid and labels on warped image, save to debug/ folder."""
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)

    vis = warped.copy()
    for row in range(19):
        for col in range(19):
            x, y = int(x_grid[col]), int(y_grid[row])
            label = board_state[row][col]
            color = (0, 255, 0) if label == 'B' else \
                    (255, 255, 255) if label == 'W' else (0, 0, 255)
            cv2.circle(vis, (x, y), 8, color, 2)
            if label != 'E':
                cv2.putText(vis, label, (x-5, y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw grid lines
    for col in range(19):
        cv2.line(vis, (int(x_grid[col]), int(y_grid[0])),
                      (int(x_grid[col]), int(y_grid[-1])), (100, 100, 100), 1)
    for row in range(19):
        cv2.line(vis, (int(x_grid[0]), int(y_grid[row])),
                      (int(x_grid[-1]), int(y_grid[row])), (100, 100, 100), 1)

    out_path = debug_dir / f"{session_name}_{frame_stem}_grid.jpg"
    # Resize to reasonable dimension
    scale = min(1.0, 800 / max(vis.shape[:2]))
    if scale < 1.0:
        vis = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tqdm.write(f"  [Debug] Grid visualization -> {out_path}")


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract Weiqi cross-point patches from Gomrade dataset (Windows Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python extract_patches_win.py
  python extract_patches_win.py --config config_windows.yaml
  python extract_patches_win.py --gomrade D:/path/to/Gomrade-dataset1 --output patches/
  python extract_patches_win.py --debug-session session_name
  python extract_patches_win.py --limit 3   # Test first 3 folders
        """
    )
    parser.add_argument("--config",  default="config_windows.yaml",   help="Path to YAML config")
    parser.add_argument("--gomrade", default=None,             help="Override: Gomrade dataset root")
    parser.add_argument("--output",  default=None,             help="Override: patch output dir")
    parser.add_argument("--size",    type=int, default=None,   help="Override: patch size (px)")
    parser.add_argument("--limit",   type=int, default=None,   help="Debug: process max N folders")
    parser.add_argument("--debug-session", default=None,
                        help="Output grid debug images for specified session")
    args = parser.parse_args()

    config = load_config(args.config)

    gomrade_dir = Path(args.gomrade or get_param(config, "data", "gomrade_dir", default=""))
    output_dir  = Path(args.output  or get_param(config, "data", "patches_dir",  default="patches"))
    patch_size  = args.size         or get_param(config, "model", "patch_size",   default=48)

    if not gomrade_dir.exists():
        print(f"Error: Gomrade dataset directory not found: {gomrade_dir}")
        print("Please fix data.gomrade_dir in your config file.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================")
    print("   Gomrade Patch Extractor (Windows)           ")
    print("================================================")
    print(f"  Dataset: {gomrade_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Patch Size: {patch_size}x{patch_size}")
    print()

    sessions = sorted([d for d in gomrade_dir.iterdir() if d.is_dir()])
    if args.limit:
        sessions = sessions[:args.limit]

    print(f"  Found {len(sessions)} session folders")

    stats = {"B": 0, "W": 0, "E": 0, "processed": 0, "skipped": 0, "no_file": 0}
    total = 0

    for session in tqdm(sessions, desc="Processing sessions", unit="session"):
        debug_this = (args.debug_session is not None and
                      args.debug_session in session.name)
        n = process_session(session, output_dir, patch_size, stats, debug=debug_this)
        if n == -1:
            stats["no_file"] += 1   # Missing files
        elif n == 0:
            stats["skipped"] += 1   # Parse failed
        else:
            stats["processed"] += 1
        total += max(0, n)

    # Save summary
    summary = {
        "total_patches":     total,
        "black_patches":     stats["B"],
        "white_patches":     stats["W"],
        "empty_patches":     stats["E"],
        "sessions_processed":stats["processed"],
        "sessions_skipped":  stats["skipped"],
        "sessions_no_file":  stats["no_file"],
        "patch_size":        patch_size,
        "gomrade_dir":       str(gomrade_dir),
        "output_dir":        str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nExtraction complete!")
    print(f"   Total: {total:,} patches")
    print(f"   Black (B): {stats['B']:,}")
    print(f"   White (W): {stats['W']:,}")
    print(f"   Empty (E): {stats['E']:,}")
    print(f"   Success: {stats['processed']}  Failed: {stats['skipped']}  "
          f"Missing files: {stats['no_file']}")
    print(f"\n   Output structure:")
    print(f"   {output_dir}/")
    print(f"     B/  - {stats['B']:,} black patches")
    print(f"     W/  - {stats['W']:,} white patches")
    print(f"     E/  - {stats['E']:,} empty patches")
    print(f"     summary.json")
    print(f"\n   Next step: python train-win.py --config config_windows.yaml")


if __name__ == "__main__":
    main()
