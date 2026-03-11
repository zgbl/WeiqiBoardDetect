"""
围棋盘角点网格检测 v1.6 — 深度图像预处理版
==================================================
预处理流程：
  1. 双边滤波 (Bilateral Filter) — 去噪且保留边缘
  2. 非锐化掩模 (Unsharp Mask) — 提升线条锐度
  3. CLAHE — 局部对比度增强
检测：Shi-Tomasi (Good Features to Track)
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from scipy.spatial import KDTree

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 深度预处理 + 角点提取
# ─────────────────────────────────────────────────────────────────────────────

def extract_corners_v16(img_bgr, max_corners=2500, quality_level=0.005, min_dist=8):
    # 1. 基础灰度化
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Step 1.a: Grayscale vs Enhanced (V1.6)", gray)
    cv2.waitKey(0)
    
    # 2. 去噪 (Smoothing)
    # 双边滤波能在去噪的同时保持边缘锐利，非常适合棋盘线
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imshow("Step 1.a: Grayscale vs Enhanced (V1.6)", denoised)
    cv2.waitKey(0)
    
    # 3. 锐化 (Sharpening)
    # 使用 Unsharp Mask: 增强高频分量
    gaussian_blur = cv2.GaussianBlur(denoised, (5, 5), 0)
    sharpened = cv2.addWeighted(denoised, 1.2, gaussian_blur, -0.8, 0)
    
    # 4. 边缘增强 (Edge Enhancement via CLAHE)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    
    # --- 调试展示预处理过程 ---
    dw = 1000
    pre_show = np.hstack([cv2.resize(gray, (dw//2, int(dw/2*gray.shape[0]/gray.shape[1]))), 
                          cv2.resize(enhanced, (dw//2, int(dw/2*enhanced.shape[0]/enhanced.shape[1])))])
    cv2.imshow("Step 1.a: Grayscale vs Enhanced (V1.6)", pre_show)
    print("[*] 1.a 观察增强效果。左侧为原生灰度，右侧为 [去噪+锐化+增强] 后的效果。(按键继续)")
    cv2.waitKey(0)

    # 5. 角点提取 (Shi-Tomasi)
    # 在这个高度清晰的图上找点
    corners = cv2.goodFeaturesToTrack(enhanced, 
                                      maxCorners=max_corners, 
                                      qualityLevel=quality_level, 
                                      minDistance=min_dist,
                                      blockSize=3)
    
    if corners is None: return np.array([]), enhanced

    pts = corners.reshape(-1, 2)
    
    # 6. 亚像素细化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    pts = cv2.cornerSubPix(enhanced, pts, (5, 5), (-1, -1), criteria)
    
    print(f"[*] 1.b 候选角点数: {len(pts)}")
    
    # --- Step 1.b: 结果叠加展示 ---
    debug_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for pt in pts:
        cv2.circle(debug_vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    cv2.imshow("Step 1.b: Candidate Corners (V1.6)", cv2.resize(debug_vis, (dw, int(dw*debug_vis.shape[0]/debug_vis.shape[1]))))
    print("      -> 确认右上角漏检是否改善？(按键继续)")
    cv2.waitKey(0)
    
    return pts, enhanced

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 方向估计
# ─────────────────────────────────────────────────────────────────────────────

def estimate_main_directions(img_gray):
    # 使用深度增强图做边缘提取
    edges = cv2.Canny(img_gray, 40, 100)
    lines = cv2.HoughLines(edges, 1, np.pi/360, 160)
    
    debug_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for rho, theta in lines[:25, 0]:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            p1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            p2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv2.line(debug_vis, p1, p2, (255, 0, 0), 1)

    angles = lines[:, 0, 1] if lines is not None else [0, np.pi/2]
    angles = angles % np.pi
    
    hist, bin_edges = np.histogram(angles, bins=360, range=(0, np.pi))
    a1 = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist)+1]) / 2
    a2 = (a1 + np.pi/2) % np.pi
    
    cv2.imshow("Step 2: Hough Lines Direction (V1.6)", cv2.resize(debug_vis, (1000, 1000*debug_vis.shape[0]//debug_vis.shape[1])))
    print(f"[*] 2. 估计主方向: {np.degrees(a1):.1f}°, {np.degrees(a2):.1f}° (按键继续)")
    cv2.waitKey(0)
    
    return a1, a2

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 & 4: RANSAC 与可视化 (继承之前的稳健逻辑)
# ─────────────────────────────────────────────────────────────────────────────

def project_points(pts, angle):
    ca, sa = np.cos(angle), np.sin(angle)
    p_norm = pts[:,0]*(-sa) + pts[:,1]*ca
    return p_norm

def find_equidistant_inliers_1d(pos, n_iter=1500, tol_ratio=0.18):
    n = len(pos)
    if n < 5: return np.ones(n, dtype=bool), 0, 0
    pos = np.sort(pos)
    best_mask, best_count, best_d, best_orig = None, 0, 0, 0
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
            if mask.sum() > best_count:
                best_count, best_mask, best_d, best_orig = mask.sum(), mask, d, orig
    return best_mask, best_d, best_orig

def grid_ransac(pts, a1, a2):
    best_score, best_mask, best_params = 0, np.zeros(len(pts), dtype=bool), None
    for r_ang, c_ang in [(a1, a2), (a2, a1)]:
        row_proj = project_points(pts, r_ang + np.pi/2)
        col_proj = project_points(pts, c_ang + np.pi/2)
        rm, rd, ro = find_equidistant_inliers_1d(row_proj)
        cm, cd, co = find_equidistant_inliers_1d(col_proj)
        combined = rm & cm
        if combined.sum() > best_score:
            best_score, best_mask = combined.sum(), combined
            best_params = {'row_angle': r_ang, 'col_angle': c_ang, 'row_d': rd, 'row_orig': ro, 'col_d': cd, 'col_orig': co}
    return best_mask, best_params

def visualize_final(img_gray, pts, mask, params):
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    inliers = pts[mask]
    for p in pts: cv2.circle(out, (int(p[0]), int(p[1])), 2, (100,100,100), -1)
    for p in inliers: cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
    
    if params and len(inliers) >= 4:
        rp = project_points(inliers, params['row_angle'] + np.pi/2)
        cp = project_points(inliers, params['col_angle'] + np.pi/2)
        r_idx = np.round((rp - params['row_orig']) / params['row_d']).astype(int)
        c_idx = np.round((cp - params['col_orig']) / params['col_d']).astype(int)
        pts_dict = {(r_idx[i], c_idx[i]): inliers[i] for i in range(len(inliers))}
        rmin, rmax, cmin, cmax = r_idx.min(), r_idx.max(), c_idx.min(), c_idx.max()
        corner_indices = [(rmin, cmin), (rmin, cmax), (rmax, cmax), (rmax, cmin)]
        box = []
        for rc in corner_indices:
            if rc in pts_dict: box.append(pts_dict[rc])
            else:
                dist = (r_idx - rc[0])**2 + (c_idx - rc[1])**2
                box.append(inliers[np.argmin(dist)])
        for i, p in enumerate(box):
            cv2.circle(out, (int(p[0]), int(p[1])), 20, (0, 0, 255), 3)
            cv2.putText(out, f"P{i}", (int(p[0])+20, int(p[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.line(out, tuple(box[i].astype(int)), tuple(box[(i+1)%4].astype(int)), (0, 0, 255), 2)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    args = parser.parse_args()
    img = cv2.imread(args.img)
    if img is None: return
    print("\n--- [V1.6 深度处理版] 开始 ---")
    
    # 按照用户要求：去噪 -> 锐化 -> 增强
    pts, enhanced_gray = extract_corners_v16(img)
    a1, a2 = estimate_main_directions(enhanced_gray)
    mask, params = grid_ransac(pts, a1, a2)
    res = visualize_final(enhanced_gray, pts, mask, params)
    
    dw = 1200
    disp = cv2.resize(res, (dw, int(res.shape[0]*dw/res.shape[1])))
    cv2.imshow("Final Result V1.6 (Deep Preprocessed Gray)", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
