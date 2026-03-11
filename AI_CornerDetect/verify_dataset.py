import cv2
import yaml
import numpy as np
import os
from pathlib import Path

"""
Task: Validation (第一阶段：验证数据集)
Description: 
    此脚本用于验证 Gomrade 数据集中的人工标注点 (pts_clicks) 是否与图像坐标系一致。
    它会读取 board_extractor_state.yml 中的四角坐标，并在原始图像上绘制出来。
Usage:
    python verify_gomrade_data.py --session 6
    python verify_gomrade_data.py --session Metta-Sgaravatti2
"""

def verify_session(session_path, output_root):
    session_path = Path(session_path)
    if not session_path.exists():
        print(f"❌ 找不到会话目录: {session_path}")
        return

    ext_yml = session_path / "board_extractor_state.yml"
    if not ext_yml.exists():
        print(f"⚠️ 缺少标注文件: {ext_yml}")
        return

    # 1. 加载标注点 (Ground Truth Corners)
    try:
        with open(ext_yml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 解析 YML 失败: {e}")
        return

    if "pts_clicks" not in data:
        print(f"⚠️ {ext_yml.name} 中没有找到 pts_clicks 字段")
        return
    
    # 标注点顺序通常是: 左上, 右上, 右下, 左下
    pts = np.array(data["pts_clicks"], dtype=np.int32)
    
    # 2. 查找图像文件
    # 优先查找 .png，然后是 .jpg
    img_files = list(session_path.glob("*.png")) + list(session_path.glob("*.jpg"))
    if not img_files:
        print(f"⚠️ 在 {session_path} 中没找到任何图片文件")
        return
    
    # 取第一张进行验证
    img_path = img_files[0]
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ 无法读取图像: {img_path}")
        return
    
    print(f"🔍 正在验证 {session_path.name}，图像尺寸: {img.shape[1]}x{img.shape[0]}")

    # 3. 绘制标注
    vis = img.copy()
    
    # 绘制连接线 (闭合多边形)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=8)
    
    # 绘制角点
    for i, pt in enumerate(pts):
        # 绘制实心圆
        cv2.circle(vis, tuple(pt), radius=15, color=(0, 0, 255), thickness=-1)
        # 标注索引编号
        cv2.putText(vis, str(i), (pt[0]+15, pt[1]+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)

    # 4. 保存验证结果
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    out_name = f"verify_{session_path.name}_{img_path.name}"
    out_path = output_root / out_name
    
    cv2.imwrite(str(out_path), vis)
    print(f"✅ 验证图像已保存至: {out_path}")
    print(f"   请检查图片中的红色圆点是否精确对齐棋盘的四个物理角。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gomrade 数据集坐标验证工具")
    parser.add_argument("--gomrade_root", default="/Users/tuxy/Codes/AI/Data/Gomrade-dataset1", help="数据集根目录")
    parser.add_argument("--session", default="6", help="要验证的子文件夹名称")
    parser.add_argument("--output", default="./data_check", help="结果保存目录")
    
    args = parser.parse_args()
    
    target_path = Path(args.gomrade_root) / args.session
    verify_session(target_path, args.output)
