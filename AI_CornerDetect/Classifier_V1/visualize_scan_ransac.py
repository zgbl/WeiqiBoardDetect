import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import PatchClassifier
from sklearn.cluster import DBSCAN

def filter_by_geometry_ransac(all_preds, img_h, img_w):
    """
    全图几何过滤核心逻辑：
    1. 用 DBSCAN 聚类分出可能的 Corner 和 Edge 区域，剔除孤立噪点。
    2. 如果有足够多的 C 点，尝试寻找四边形。
    3. 如果 Edge 点很多，尝试用 RANSAC 拟合四条边。
    """
    corners = [p for p in all_preds if p['cat'] == 0]
    edges = [p for p in all_preds if p['cat'] == 1]
    
    if len(corners) < 4:
        return corners, edges # 点不够，无法过滤

    # --- A. 过滤孤立 Corner 点 (DBSCAN 聚类) ---
    c_coords = np.array([[p['x'], p['y']] for p in corners])
    clustering = DBSCAN(eps=100, min_samples=2).fit(c_coords) # 100 像素内必须有伴，否则是噪点
    valid_corners = [corners[i] for i in range(len(corners)) if clustering.labels_[i] != -1]

    # --- B. 寻找最外围四分位点 (寻找 4 个极致点作为候选角点) ---
    if len(valid_corners) >= 4:
        # 使用凸包获取候选角点
        coords = np.array([[p['x'], p['y']] for p in valid_corners])
        hull = cv2.convexHull(coords.astype(np.int32))
        # 简化凸包到 4 个顶点
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx_quad = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx_quad) == 4:
            quad = approx_quad.reshape(4, 2)
            # --- C. 过滤 Edge 点 ---
            # 只有在四边形边界 50 像素范围内的 Edge 才保留
            filtered_edges = []
            for e in edges:
                p = np.array([e['x'], e['y']])
                dists = []
                for i in range(4):
                    l1, l2 = quad[i], quad[(i+1)%4]
                    dist = dist_to_segment(p, l1, l2)
                    dists.append(dist)
                if min(dists) < 50: # 距离棋盘边界太远的点全部干掉
                    filtered_edges.append(e)
            
            return [p for p in valid_corners], filtered_edges

    return valid_corners, edges

def dist_to_segment(p, a, b):
    pa = p - a
    ba = b - a
    t = np.dot(pa, ba) / (np.dot(ba, ba) + 1e-6)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(pa - t * ba)

def visualize_ransac_scan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 路径
    INPUT_DIR = Path(r"E:\Data\WeiqiPics\Test1")
    OUTPUT_DIR = Path(r"E:\Data\WeiqiPics\Test1\output_geometric")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 模型加载
    model = PatchClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load("weights/best_classifier.pth", map_location=device))
    model.eval()

    patch_size = 128
    stride = 64
    half = patch_size // 2

    # 归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    img_files = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))

    for img_path in tqdm(img_files):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        h, w = img_bgr.shape[:2]
        all_preds = []

        # 1. 第一遍扫描：收集所有原始预测
        for y in range(half, h - half, stride):
            for x in range(half, w - half, stride):
                patch = img_bgr[y-half:y+half, x-half:x+half]
                p_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(p_rgb).permute(2, 0, 1).float() / 255.0
                t = (t.to(device) - mean) / std
                
                with torch.no_grad():
                    out = model(t.unsqueeze(0))
                    conf, pred_id = torch.max(torch.softmax(out, dim=1), 1)
                
                if conf.item() > 0.8:
                    all_preds.append({'x': x, 'y': y, 'cat': pred_id.item(), 'conf': conf.item()})

        # 2. 第二遍：应用几何约束过滤 (RANSAC/凸包过滤)
        f_corners, f_edges = filter_by_geometry_ransac(all_preds, h, w)

        # 3. 绘图
        display_img = img_bgr.copy()
        overlay = img_bgr.copy()

        for c in f_corners:
            cv2.circle(display_img, (c['x'], c['y']), 10, (0, 0, 255), -1) # 红色实心圆点
            cv2.putText(display_img, "C", (c['x']-15, c['y']-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            
        for e in f_edges:
            cv2.circle(display_img, (e['x'], e['y']), 5, (0, 255, 0), -1) # 绿色小点表示边
            
        # 保存结果
        cv2.imwrite(str(OUTPUT_DIR / f"geom_{img_path.name}"), display_img)

    print(f"Geometric filtering complete! Check {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_ransac_scan()
