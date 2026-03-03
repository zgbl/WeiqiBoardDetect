"""
extract_patches.py — 从 Gomrade 数据集提取交叉点 Patch
================================================================
利用 Gomrade 数据集内置的透视矩阵和网格坐标，自动提取每个
交叉点的 patch，按 B/W/E 分类存储。

不需要任何人工标注工作！所有坐标和标注由数据集提供。

使用方法:
  python3 extract_patches.py                       # 使用默认配置
  python3 extract_patches.py --config config.yaml  # 指定配置文件
  python3 extract_patches.py --gomrade /path/to/Gomrade-dataset1
  python3 extract_patches.py --debug-session DeLazzari-Greenberg  # 可视化调试
"""

import cv2
import numpy as np
import os
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────
def load_config(config_path="config.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def get_param(config, *keys, default=None):
    val = config
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val if val is not None else default


# ──────────────────────────────────────────────────────────────
# Gomrade YML 解析
# ──────────────────────────────────────────────────────────────
def parse_board_txt(txt_path):
    """解析 19×19 棋盘状态文件。返回 list[list[str]] 或 None。"""
    state = []
    with open(txt_path) as f:
        for line in f:
            row = [c if c in ('B', 'W') else 'E' for c in line.strip().split()]
            if len(row) == 19:
                state.append(row)
    return state if len(state) == 19 else None


def parse_classifier_state_yml(yml_path):
    """
    解析 board_state_classifier_state.yml。
    返回 x_grid (len=19), y_grid (len=19)。
    注意：不同会话的 width/height 字段不可靠，只用 x_grid/y_grid。
    """
    with open(yml_path) as f:
        data = yaml.safe_load(f)
    x_grid = data.get("x_grid", [])
    y_grid = data.get("y_grid", [])
    return x_grid, y_grid


def compute_perspective_matrix(yml_path, warp_w, warp_h):
    """
    从 board_extractor_state.yml 获取透视变换矩阵 M。

    优先使用预计算的 M 矩阵（格式A）。
    若没有 M，从 pts_clicks 四角点反算（格式B）。

    warp_w, warp_h 是目标变换图像的尺寸（从 x_grid/y_grid 推导而来）。
    """
    with open(yml_path) as f:
        data = yaml.safe_load(f)

    if "M" in data:
        return np.array(data["M"], dtype=np.float64)

    if "pts_clicks" in data:
        pts = np.array(data["pts_clicks"], dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError(f"pts_clicks 格式错误: shape={pts.shape}")
        # pts_clicks 顺序：左上 → 右上 → 右下 → 左下（顺时针）
        dst = np.float32([
            [0,         0         ],
            [warp_w - 1, 0        ],
            [warp_w - 1, warp_h -1],
            [0,         warp_h - 1],
        ])
        return cv2.getPerspectiveTransform(pts, dst)

    raise KeyError("board_extractor_state.yml 既没有 M 也没有 pts_clicks")


# ──────────────────────────────────────────────────────────────
# Patch 提取
# ──────────────────────────────────────────────────────────────
def extract_patch(img, x, y, radius):
    """在 (x,y) 处裁剪 2*radius × 2*radius patch，边界安全。"""
    h, w = img.shape[:2]
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius), min(h, y + radius)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]


def process_session(session_dir, output_dir, patch_size, stats, debug=False):
    """
    处理 Gomrade 数据集中一个对局文件夹。
    返回本次提取的 patch 数量，-1 表示跳过。
    """
    session_dir = Path(session_dir)
    ext_yml = session_dir / "board_extractor_state.yml"
    cls_yml = session_dir / "board_state_classifier_state.yml"

    if not ext_yml.exists() or not cls_yml.exists():
        return -1  # 静默跳过（缺少必要文件）

    try:
        # ── Step 1: 读取网格坐标 ──
        # 关键：warp 尺寸必须从 x_grid/y_grid 推导，不能依赖 YML 里的 width/height
        # （各会话的 width/height 字段不一致，有的在 extractor，有的在 classifier，有的没有）
        x_grid, y_grid = parse_classifier_state_yml(cls_yml)

        if len(x_grid) != 19 or len(y_grid) != 19:
            return -1

        # 从网格坐标反推图像尺寸
        # x_grid[i] 是第 i 列在变换后图像中的 x 像素坐标
        # x_grid[0]=0（最左列贴边），x_grid[-1]=width-1（最右列贴边）
        # 所以 warp_w = x_grid[-1] + 1, warp_h = y_grid[-1] + 1
        # 若有 margin（x_grid[0]>0），则右侧也加同样 padding：
        warp_w = x_grid[-1] + x_grid[0] + 1
        warp_h = y_grid[-1] + y_grid[0] + 1

        # ── Step 2: 获取透视变换矩阵 ──
        M = compute_perspective_matrix(ext_yml, warp_w, warp_h)

    except Exception as e:
        tqdm.write(f"  [跳过] {session_dir.name}: 解析失败 ({e})")
        return 0

    # 棋子半径 = 格子间距的 42%（留出一点空间让相邻棋子轮廓可见）
    step_x = (x_grid[-1] - x_grid[0]) / 18
    step_y = (y_grid[-1] - y_grid[0]) / 18
    radius = max(8, int(min(step_x, step_y) * 0.42))

    # 找该文件夹下所有 .txt 标注文件（对应每帧图像）
    txt_files = sorted(session_dir.glob("*.txt"))
    count = 0

    for txt_path in txt_files:
        # 找对应图像（.png 优先，.jpg 备用）
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

        # 透视变换
        try:
            warped = cv2.warpPerspective(img, M, (warp_w, warp_h))
        except Exception:
            continue

        # ── 调试模式：保存带网格可视化的图像 ──
        if debug:
            _save_debug_visualization(warped, x_grid, y_grid, board_state,
                                      session_dir.name, txt_path.stem)
            debug = False  # 每个 session 只保存第一帧

        # 提取每个交叉点的 patch
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
    """
    在变换后的图像上叠加网格和标注，保存到 debug/ 目录。
    绿色圆圈 = 标注为有子的交叉点，红色 = 空。
    """
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

    # 画网格线
    for col in range(19):
        cv2.line(vis, (int(x_grid[col]), int(y_grid[0])),
                      (int(x_grid[col]), int(y_grid[-1])), (100, 100, 100), 1)
    for row in range(19):
        cv2.line(vis, (int(x_grid[0]), int(y_grid[row])),
                      (int(x_grid[-1]), int(y_grid[row])), (100, 100, 100), 1)

    out_path = debug_dir / f"{session_name}_{frame_stem}_grid.jpg"
    # 缩放到合理尺寸
    scale = min(1.0, 800 / max(vis.shape[:2]))
    if scale < 1.0:
        vis = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tqdm.write(f"  [调试] 网格可视化 → {out_path}")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="从 Gomrade 数据集提取围棋交叉点 Patch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 extract_patches.py
  python3 extract_patches.py --config /path/to/config.yaml
  python3 extract_patches.py --gomrade /path/to/Gomrade-dataset1 --output patches/
  python3 extract_patches.py --debug-session DeLazzari-Greenberg  # 可视化网格对齐
  python3 extract_patches.py --limit 3   # 快速测试前3个文件夹
        """
    )
    parser.add_argument("--config",  default="config.yaml",   help="YAML 配置文件路径")
    parser.add_argument("--gomrade", default=None,             help="覆盖配置: Gomrade 数据集根目录")
    parser.add_argument("--output",  default=None,             help="覆盖配置: patch 输出目录")
    parser.add_argument("--size",    type=int, default=None,   help="覆盖配置: patch 尺寸（像素）")
    parser.add_argument("--limit",   type=int, default=None,   help="调试用：最多处理 N 个文件夹")
    parser.add_argument("--debug-session", default=None,
                        help="对指定 session 输出带网格的调试图像（检查坐标是否准确）")
    args = parser.parse_args()

    config = load_config(args.config)

    gomrade_dir = Path(args.gomrade or get_param(config, "data", "gomrade_dir", default=""))
    output_dir  = Path(args.output  or get_param(config, "data", "patches_dir",  default="patches"))
    patch_size  = args.size         or get_param(config, "model", "patch_size",   default=48)

    if not gomrade_dir.exists():
        print(f"❌ 找不到 Gomrade 数据集目录: {gomrade_dir}")
        print("   请修改 config.yaml 中的 data.gomrade_dir")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════╗")
    print("║   Gomrade Patch 提取器                       ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  数据集: {gomrade_dir}")
    print(f"  输出:   {output_dir}")
    print(f"  Patch 尺寸: {patch_size}×{patch_size}")
    print()

    sessions = sorted([d for d in gomrade_dir.iterdir() if d.is_dir()])
    if args.limit:
        sessions = sessions[:args.limit]

    print(f"  找到 {len(sessions)} 个对局文件夹")

    stats = {"B": 0, "W": 0, "E": 0, "processed": 0, "skipped": 0, "no_file": 0}
    total = 0

    for session in tqdm(sessions, desc="处理对局", unit="session"):
        debug_this = (args.debug_session is not None and
                      args.debug_session in session.name)
        n = process_session(session, output_dir, patch_size, stats, debug=debug_this)
        if n == -1:
            stats["no_file"] += 1   # 缺少必要文件
        elif n == 0:
            stats["skipped"] += 1   # 有文件但解析失败
        else:
            stats["processed"] += 1
        total += max(0, n)

    # 保存汇总
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
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 提取完成！")
    print(f"   总计: {total:,} 个 patch")
    print(f"   黑子 (B): {stats['B']:,}")
    print(f"   白子 (W): {stats['W']:,}")
    print(f"   空格 (E): {stats['E']:,}")
    print(f"   处理成功: {stats['processed']}  解析失败: {stats['skipped']}  "
          f"缺少文件: {stats['no_file']}")
    print(f"\n   输出目录结构:")
    print(f"   {output_dir}/")
    print(f"     B/  ← {stats['B']:,} 个黑子 patch")
    print(f"     W/  ← {stats['W']:,} 个白子 patch")
    print(f"     E/  ← {stats['E']:,} 个空格 patch")
    print(f"     summary.json")
    print(f"\n   下一步: python3 train.py")

    if stats["no_file"] > 0:
        print(f"\n   提示: {stats['no_file']} 个文件夹缺少 board_state_classifier_state.yml")
        print(f"   （这些通常是只有单张图片的小会话，属于正常情况）")


if __name__ == "__main__":
    main()
