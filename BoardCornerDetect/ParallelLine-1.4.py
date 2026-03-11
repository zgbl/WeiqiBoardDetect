"""
围棋盘角点网格检测 v1.4 (Debug版)
==========================================
增加了分步展示功能，帮助定位为何忽略了正经角点。
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from scipy.spatial import KDTree

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Harris Corner 提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_harris_corners(img_bgr, block_size=3, ksize=3, k=0.04, thresh_ratio=0.003, nms_radius=6):
    # 1. 基础灰度化
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 增强对比度 (CLAHE) - 调强 clipLimit 以抠出远端细线
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    
    # 3. 轻微模糊
    gray_blur = cv2.GaussianBlur(gray_eq, (3,3), 0)
    
    # --- Step 1.a: 展示处理后的灰度图 ---
    dw = 1000
    gray_disp = cv2.resize(gray_blur, (dw, int(gray_blur.shape[0]*dw/gray_blur.shape[1])))
    cv2.imshow("Step 1.a: Super-Enhanced Grayscale", gray_disp)
    print("[*] 1.a 请确认远端(右上角)的线是否变明显了？(按键继续)")
    cv2.waitKey(0)

    gray_f = np.float32(gray_blur)
    
    # Harris 检测
    dst = cv2.cornerHarris(gray_f, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    
    # 显著降低阈值比例 (0.015)，宁滥勿缺
    threshold = thresh_ratio * dst.max()
    ys, xs = np.where(dst > threshold)
    scores = dst[ys, xs]
    
    if len(xs) == 0: return np.array([])

    # 按分数排序
    order = np.argsort(-scores)
    xs, ys, scores = xs[order], ys[order], scores[order]
    
    # NMS (Non-Maximum Suppression)
    kept = []
    pts_arr = np.stack([xs, ys], axis=1).astype(np.float32)
    tree = KDTree(pts_arr)
    removed = np.zeros(len(xs), dtype=bool)
    
    for i in range(len(xs)):
        if removed[i]: continue
        kept.append((xs[i], ys[i]))
        # 压制半径内的其他点
        indices = tree.query_ball_point([xs[i], ys[i]], nms_radius)
        for idx in indices: removed[idx] = True
            
    pts = np.array(kept, dtype=np.float32)
    
    # 亚像素细化 (统一使用增强后的灰度图以保持一致性)
    if len(pts) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        pts = cv2.cornerSubPix(gray_blur, pts, (5, 5), (-1, -1), criteria)
        
    print(f"[*] 1.b Harris NMS 后提取角点数: {len(pts)}")
    
    # --- Step 1.b: 展示所有角点叠加图 ---
    # 背景使用增强后的灰度图，转为 BGR 只是为了能画红点
    debug_img = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)
    for pt in pts:
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    cv2.imshow("Step 1.b: Found Candidate Corners (on Enhanced Gray)", cv2.resize(debug_img, (dw, int(dw*debug_img.shape[0]/debug_img.shape[1]))))
    print("      -> 请基于增强灰度底图，观察交点红点分布。(按键继续)")
    cv2.waitKey(0)
    
    return pts, gray_blur

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 主方向估计
# ─────────────────────────────────────────────────────────────────────────────

def estimate_main_directions(img_bgr, img_gray):
    # 调低 Canny 阈值 (30, 80)，让远端弱线有机会生成边缘
    edges = cv2.Canny(img_gray, 30, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    
    # 修正：可视化背景必须使用增强灰度图！
    debug_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for rho, theta in lines[:20, 0]: # 画前20条
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            p1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            p2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv2.line(debug_img, p1, p2, (255, 0, 0), 1)

    angles = lines[:, 0, 1] if lines is not None else [0, np.pi/2]
    angles = angles % np.pi
    
    # 计算直方图找到最强的两个相互垂直的方向
    hist, bin_edges = np.histogram(angles, bins=180, range=(0, np.pi))
    
    # 找第一个峰值
    a1_idx = np.argmax(hist)
    a1 = (bin_edges[a1_idx] + bin_edges[a1_idx+1]) / 2
    a2 = (a1 + np.pi/2) % np.pi
    
    cv2.imshow("Step 2: Main Directions (Blue Lines)", cv2.resize(debug_img, (1000, int(1000*debug_img.shape[0]/debug_img.shape[1]))))
    print(f"[*] 2. 估计主方向: {np.degrees(a1):.1f}°, {np.degrees(a2):.1f}° (按任意键继续)")
    cv2.waitKey(0)
    
    return a1, a2

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 1D RANSAC 拟合
# ─────────────────────────────────────────────────────────────────────────────

def find_equidistant_inliers_1d(pos, n_iter=1000, tol_ratio=0.15, axis_name="Row"):
    n = len(pos)
    if n < 5: return np.ones(n, dtype=bool), 0, 0
    
    pos = np.sort(pos)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    best_d = 0
    best_orig = 0
    
    rng = np.random.default_rng()
    for _ in range(n_iter):
        idx1, idx2 = rng.choice(n, 2, replace=False)
        gap = abs(pos[idx1] - pos[idx2])
        if gap < 5: continue
        
        for k in range(1, 19): 
            d = gap / k
            if d < 15 or d > 150: continue # 限制合理间距
            
            orig = pos[idx1] % d
            shifted = (pos - orig) % d
            dist = np.minimum(shifted, d - shifted)
            mask = dist < (d * tol_ratio)
            count = mask.sum()
            
            if count > best_count:
                best_count, best_mask, best_d, best_orig = count, mask, d, orig
                
    print(f"    - {axis_name} 轴: 找到最佳间距 d={best_d:.2f}, 内点数={best_count}")
    return best_mask, best_d, best_orig

def grid_ransac(pts, a1, a2, tol_ratio=0.15):
    best_combined_mask = np.zeros(len(pts), dtype=bool)
    best_score = 0
    best_params = None
    
    for row_angle, col_angle in [(a1, a2), (a2, a1)]:
        # 投影轴
        sa, ca = np.sin(row_angle), np.cos(row_angle)
        proj_row = pts[:, 0] * (-sa) + pts[:, 1] * ca
        
        sa2, ca2 = np.sin(col_angle), np.cos(col_angle)
        proj_col = pts[:, 0] * (-sa2) + pts[:, 1] * ca2
        
        row_mask, rd, r_orig = find_equidistant_inliers_1d(proj_row, tol_ratio=tol_ratio, axis_name="Horizontal-ish")
        col_mask, cd, c_orig = find_equidistant_inliers_1d(proj_col, tol_ratio=tol_ratio, axis_name="Vertical-ish")
        
        combined = row_mask & col_mask
        score = combined.sum()
        
        if score > best_score:
            best_score, best_combined_mask = score, combined
            best_params = {'row_angle': row_angle, 'col_angle': col_angle, 'row_d': rd, 'row_orig': r_orig, 'col_d': cd, 'col_orig': c_orig}
            
    return best_combined_mask, best_params

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize_result(img_bgr, all_pts, inlier_mask, params):
    out = img_bgr.copy()
    inliers = all_pts[inlier_mask]
    
    # 画所有灰点
    for pt in all_pts: cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (120, 120, 120), -1)
    # 画绿色内点
    for pt in inliers: cv2.circle(out, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
        
    if params and len(inliers) >= 4:
        sa, ca = np.sin(params['row_angle']), np.cos(params['row_angle'])
        proj_row = inliers[:, 0] * (-sa) + inliers[:, 1] * ca
        sa2, ca2 = np.sin(params['col_angle']), np.cos(params['col_angle'])
        proj_col = inliers[:, 0] * (-sa2) + inliers[:, 1] * ca2
        
        r_idx = np.round((proj_row - params['row_orig']) / params['row_d']).astype(int)
        c_idx = np.round((proj_col - params['col_orig']) / params['col_d']).astype(int)
        
        pts_dict = {(r_idx[i], c_idx[i]): inliers[i] for i in range(len(inliers))}
        r_min, r_max, c_min, c_max = r_idx.min(), r_idx.max(), c_idx.min(), c_idx.max()
        
        corner_indices = [(r_min, c_min), (r_min, c_max), (r_max, c_max), (r_max, c_min)]
        corners = []
        for ri, ci in corner_indices:
            if (ri, ci) in pts_dict: corners.append(pts_dict[(ri, ci)])
            else:
                dists = (r_idx - ri)**2 + (c_idx - ci)**2
                corners.append(inliers[np.argmin(dists)])
        
        for i, pt in enumerate(corners):
            cv2.circle(out, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), 3)
            cv2.putText(out, f"C{i}", (int(pt[0])+15, int(pt[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        for i in range(4):
            cv2.line(out, tuple(corners[i].astype(int)), tuple(corners[(i+1)%4].astype(int)), (0, 0, 255), 2)
            
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    args = parser.parse_args()
    
    img = cv2.imread(args.img)
    if img is None: return
    
    # 流程开始
    print("\n--- 开始调试流程 ---")
    pts, enhanced_gray = extract_harris_corners(img)
    a1, a2 = estimate_main_directions(img, enhanced_gray)
    
    print("[*] 3. 执行两轴 RANSAC 拟合网格...")
    mask, params = grid_ransac(pts, a1, a2)
    
    result = visualize_result(img, pts, mask, params)
    
    dw = 1200
    disp = cv2.resize(result, (dw, int(result.shape[0]*dw/result.shape[1])))
    cv2.imshow("Final Result (Press any key to exit)", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
