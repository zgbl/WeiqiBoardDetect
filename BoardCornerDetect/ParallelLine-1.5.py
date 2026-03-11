"""
围棋盘角点网格检测 v1.5 — Shi-Tomasi + 稳健网格拟合
==================================================
改进：
  1. 使用 Shi-Tomasi 角点检测 (cv2.goodFeaturesToTrack) 取代 Harris。
  2. 保持并强化了灰度预处理 (CLAHE 4.0)。
  3. 增加更多调试显示窗口，全过程灰度底图展示。
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from scipy.spatial import KDTree

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Shi-Tomasi 角点检测
# ─────────────────────────────────────────────────────────────────────────────

def extract_shi_tomasi_corners(img_bgr, max_corners=2000, quality_level=0.005, min_distance=15):
    # 1. 基础灰度化
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 增强对比度 (CLAHE) - 深度抠出微弱线
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    
    # 3. 轻微模糊
    gray_blur = cv2.GaussianBlur(gray_eq, (3,3), 0)
    
    # --- Step 1.a: 调试展示增强后的灰度图 ---
    dw = 1000
    gray_disp = cv2.resize(gray_blur, (dw, int(gray_blur.shape[0]*dw/gray_blur.shape[1])))
    cv2.imshow("Step 1.a: Super-Enhanced Grayscale (V1.5)", gray_disp)
    print("[*] 1.a 观察增强灰度图。右上角交点是否清晰可见？(按键继续)")
    cv2.waitKey(0)

    # 4. Shi-Tomasi 检测
    # qualityLevel 越低，捕获的弱点越多
    corners = cv2.goodFeaturesToTrack(gray_blur, 
                                      maxCorners=max_corners, 
                                      qualityLevel=quality_level, 
                                      minDistance=min_distance)
    
    if corners is None:
        return np.array([]), gray_blur

    pts = corners.reshape(-1, 2)
    
    # 5. 亚像素细化 (还是在增强灰度图上做更好)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    pts = cv2.cornerSubPix(gray_blur, pts, (5, 5), (-1, -1), criteria)
    
    print(f"[*] 1.b Shi-Tomasi 发现候选角点数: {len(pts)}")
    
    # --- Step 1.b: 展示所有角点叠加图 ---
    debug_img = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)
    for pt in pts:
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    cv2.imshow("Step 1.b: Shi-Tomasi Candidates", cv2.resize(debug_img, (dw, int(dw*debug_img.shape[0]/debug_img.shape[1]))))
    print("      -> 观察右上角是否有红点覆盖？(按键继续)")
    cv2.waitKey(0)
    
    return pts, gray_blur

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 主方向估计
# ─────────────────────────────────────────────────────────────────────────────

def estimate_main_directions(img_gray):
    # 使用增强灰度图做边缘提取
    edges = cv2.Canny(img_gray, 30, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/360, 150)
    
    # 准备可视化底图 (灰度转BGR)
    debug_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for rho, theta in lines[:25, 0]: # 画前25条最强的线
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            p1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            p2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv2.line(debug_img, p1, p2, (255, 0, 0), 1)

    angles = lines[:, 0, 1] if lines is not None else [0, np.pi/2]
    angles = angles % np.pi
    
    hist, bin_edges = np.histogram(angles, bins=360, range=(0, np.pi))
    a1_idx = np.argmax(hist)
    a1 = (bin_edges[a1_idx] + bin_edges[a1_idx+1]) / 2
    
    # 此处假设第二个方向在第一个方向 90 度附近
    a2 = (a1 + np.pi/2) % np.pi
    
    cv2.imshow("Step 2: Hough Lines & Directions", cv2.resize(debug_img, (1000, 1000*debug_img.shape[0]//debug_img.shape[1])))
    print(f"[*] 2. 估计主方向: {np.degrees(a1):.1f}°, {np.degrees(a2):.1f}° (按键继续)")
    cv2.waitKey(0)
    
    return a1, a2

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 网格拟合与 RANSAC
# ─────────────────────────────────────────────────────────────────────────────

def project_points(pts, angle):
    ca, sa = np.cos(angle), np.sin(angle)
    proj_parallel = pts[:,0]*ca + pts[:,1]*sa
    proj_normal   = pts[:,0]*(-sa) + pts[:,1]*ca
    return proj_parallel, proj_normal

def find_equidistant_inliers_1d(pos, n_iter=1200, tol_ratio=0.15, debug_label=""):
    n = len(pos)
    if n < 5: return np.ones(n, dtype=bool), 0, 0
    
    pos = np.sort(pos)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    best_d = 0
    best_orig = 0
    
    rng = np.random.default_rng()
    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        gap = abs(pos[i] - pos[j])
        if gap < 5: continue
        
        for k in range(1, 19): 
            d = gap / k
            if d < 12 or d > 120: continue 
            
            orig = pos[i] % d
            shifted = (pos - orig) % d
            dist = np.minimum(shifted, d - shifted)
            mask = dist < (d * tol_ratio)
            count = mask.sum()
            
            if count > best_count:
                best_count, best_mask, best_d, best_orig = count, mask, d, orig
                
    print(f"    - [{debug_label}] 最佳间距 d={best_d:.2f}, 匹配点数={best_count}")
    return best_mask, best_d, best_orig

def grid_ransac(pts, a1, a2, tol_ratio=0.18):
    best_score = 0
    best_mask = np.zeros(len(pts), dtype=bool)
    best_params = None
    
    # a1, a2 是线的方向，我们需要向其垂线方向(投影轴)做等间距搜索
    for row_angle, col_angle in [(a1, a2), (a2, a1)]:
        # 投影轴
        _, row_proj = project_points(pts, row_angle + np.pi/2)
        _, col_proj = project_points(pts, col_angle + np.pi/2)
        
        row_mask, rd, r_orig = find_equidistant_inliers_1d(row_proj, debug_label="轴A", tol_ratio=tol_ratio)
        col_mask, cd, c_orig = find_equidistant_inliers_1d(col_proj, debug_label="轴B", tol_ratio=tol_ratio)
        
        combined = row_mask & col_mask
        score = combined.sum()
        
        if score > best_score:
            best_score, best_combined_mask = score, combined
            best_params = {'row_angle': row_angle, 'col_angle': col_angle, 'row_d': rd, 'row_orig': r_orig, 'col_d': cd, 'col_orig': c_orig}
            
    return best_combined_mask, best_params

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize_final(img_gray, all_pts, inlier_mask, params):
    # 背景恒定使用增强灰度图
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    inliers = all_pts[inlier_mask]
    
    # 灰点背景
    for pt in all_pts: cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (100, 100, 100), -1)
    # 绿点内点
    for pt in inliers: cv2.circle(out, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
    
    if params and len(inliers) >= 4:
        # 重新分配格坐标找边界
        _, rp = project_points(inliers, params['row_angle'] + np.pi/2)
        _, cp = project_points(inliers, params['col_angle'] + np.pi/2)
        
        r_idx = np.round((rp - params['row_orig']) / params['row_d']).astype(int)
        c_idx = np.round((cp - params['col_orig']) / params['col_d']).astype(int)
        
        pts_dict = {(r_idx[i], c_idx[i]): inliers[i] for i in range(len(inliers))}
        rmin, rmax, cmin, cmax = r_idx.min(), r_idx.max(), c_idx.min(), c_idx.max()
        
        # 标记顶点
        corner_coords = [(rmin, cmin), (rmin, cmax), (rmax, cmax), (rmax, cmin)]
        box = []
        for rc in corner_coords:
            if rc in pts_dict: box.append(pts_dict[rc])
            else:
                dists = (r_idx - rc[0])**2 + (c_idx - rc[1])**2
                box.append(inliers[np.argmin(dists)])
        
        for i, pt in enumerate(box):
            cv2.circle(out, (int(pt[0]), int(pt[1])), 18, (0, 0, 255), 3)
            cv2.putText(out, f"V{i}", (int(pt[0])+15, int(pt[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        for i in range(4):
            cv2.line(out, tuple(box[i].astype(int)), tuple(box[(i+1)%4].astype(int)), (0, 0, 255), 2)

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    args = parser.parse_args()
    
    img = cv2.imread(args.img)
    if img is None: return

    print("\n--- [V1.5 Shi-Tomasi 版本] 调试开始 ---")
    
    # 1. 采用 Shi-Tomasi 找点
    pts, enhanced_gray = extract_shi_tomasi_corners(img, 
                                                   max_corners=2000, 
                                                   quality_level=0.005) # 极低质量门槛以捕捉远端
    
    # 2. 估计方向 (传增强灰度图)
    a1, a2 = estimate_main_directions(enhanced_gray)
    
    # 3. 网格拟合
    print("[*] 3. 执行两轴 RANSAC 拟合网格...")
    mask, params = grid_ransac(pts, a1, a2)
    
    # 4. 可视化
    res = visualize_final(enhanced_gray, pts, mask, params)
    
    dw = 1200
    display = cv2.resize(res, (dw, int(res.shape[0]*dw/res.shape[1])))
    cv2.imshow("Final Result V1.5 (Shi-Tomasi on Gray)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
