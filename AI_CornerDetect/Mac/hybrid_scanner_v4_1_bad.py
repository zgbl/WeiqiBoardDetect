import cv2
import numpy as np
import sys
from pathlib import Path

# =====================================================================
# 底层几何工具
# =====================================================================
def segment_to_rho_theta(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx*dx + dy*dy)
    if length < 1e-8: return None
    nx, ny = -dy/length, dx/length
    rho = nx*x1 + ny*y1
    if rho < 0: rho, nx, ny = -rho, -nx, -ny
    theta = np.arctan2(ny, nx)
    if theta < 0: theta += 2*np.pi
    return rho, theta

def normalize_line(rho, theta):
    if rho < 0: rho, theta = -rho, theta + np.pi
    return rho, theta % (2*np.pi)

def circular_angle_diff(a1, a2, period=np.pi):
    d = abs(a1 - a2) % period
    return min(d, period - d)

def intersect_lines(rho1, theta1, rho2, theta2):
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    det = a1*b2 - a2*b1
    if abs(det) < 1e-8: return None
    return ((b2*rho1 - b1*rho2)/det, (a1*rho2 - a2*rho1)/det)


# =====================================================================
# 核心算法：角度分组
# =====================================================================
def separate_by_angle(lines, v_tol=18, h_tol=10):
    if len(lines) == 0:
        return [], [], [], 0, 0

    normalized = []
    for rho, theta in lines:
        r, t = normalize_line(rho, theta)
        normalized.append((r, t))

    mapped_thetas = np.array([t % np.pi for _, t in normalized])

    n_bins = 180
    bin_size = np.pi / n_bins 
    hist = np.zeros(n_bins)
    for theta in mapped_thetas:
        bin_idx = int(theta / bin_size) % n_bins
        hist[bin_idx] += 1

    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    padded = np.concatenate([hist[-kernel_size:], hist, hist[:kernel_size]])
    smoothed = np.convolve(padded, kernel, mode='same')[kernel_size:-kernel_size]

    best_h_score = -1
    peak1_bin = -1
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, np.pi/2) < np.radians(5):
            if smoothed[i] > best_h_score:
                best_h_score = smoothed[i]
                peak1_bin = i

    if peak1_bin == -1 or best_h_score < 1e-3:
        peak1_bin = int(np.argmax(smoothed))

    peak1_angle = (peak1_bin + 0.5) * bin_size 

    best_v_score = -1
    peak2_bin = -1
    target_peak2 = (peak1_angle + np.pi/2) % np.pi
    
    for i in range(n_bins):
        bin_angle = (i + 0.5) * bin_size
        if circular_angle_diff(bin_angle, target_peak2) < np.radians(15):
            if smoothed[i] > best_v_score:
                best_v_score = smoothed[i]
                peak2_bin = i

    if peak2_bin == -1 or best_v_score < 0.05 * smoothed[peak1_bin]:
        return [], [], [], peak1_angle, (peak1_angle + np.pi/2) % np.pi

    peak2_angle = (peak2_bin + 0.5) * bin_size

    def dynamic_perspective_filter(group, peak):
        if len(group) < 3: return group
        group = sorted(group, key=lambda x: x[0])
        mid_idx = len(group) // 2
        best_center_idx = mid_idx
        for offset in range(len(group)):
            idx = mid_idx + offset
            if idx < len(group) and circular_angle_diff(group[idx][1], peak) < np.radians(10):
                best_center_idx = idx
                break
            idx = mid_idx - offset
            if idx >= 0 and circular_angle_diff(group[idx][1], peak) < np.radians(10):
                best_center_idx = idx
                break
        valid_lines = [group[best_center_idx]]
        curr_angle = group[best_center_idx][1]
        for i in range(best_center_idx + 1, len(group)):
            r, t = group[i]
            if circular_angle_diff(t, curr_angle) < np.radians(12):
                valid_lines.append((r, t))
                curr_angle = t
        curr_angle = group[best_center_idx][1]
        for i in range(best_center_idx - 1, -1, -1):
            r, t = group[i]
            if circular_angle_diff(t, curr_angle) < np.radians(12):
                valid_lines.append((r, t))
                curr_angle = t
        return sorted(valid_lines, key=lambda x: x[0])

    group1, group2, ignored = [], [], []
    for i, (rho, theta) in enumerate(normalized):
        mt = mapped_thetas[i]
        d1 = circular_angle_diff(mt, peak1_angle)
        d2 = circular_angle_diff(mt, peak2_angle)
        if d1 < d2:
            if d1 < np.radians(45): group1.append((rho, theta))
            else: ignored.append((rho, theta))
        else:
            if d2 < np.radians(45): group2.append((rho, theta))
            else: ignored.append((rho, theta))

    group1 = dynamic_perspective_filter(group1, peak1_angle)
    group2 = dynamic_perspective_filter(group2, peak2_angle)
    return group1, group2, ignored, peak1_angle, peak2_angle


# =====================================================================
# 核心算法：聚类
# =====================================================================
def cluster_lines(lines, rho_threshold=6, theta_threshold_deg=2):
    if not lines: return []
    lines = sorted(lines, key=lambda l: l[0])
    clusters = []
    for rho, theta in lines:
        found_cluster = False
        for cluster in clusters:
            avg_rho = np.mean([item[0] for item in cluster])
            sum_sin = np.sum([np.sin(item[1]) for item in cluster])
            sum_cos = np.sum([np.cos(item[1]) for item in cluster])
            avg_theta = np.arctan2(sum_sin, sum_cos)
            d_rho = abs(rho - avg_rho)
            d_theta = circular_angle_diff(theta, avg_theta)
            if d_rho < rho_threshold and d_theta < np.radians(theta_threshold_deg):
                cluster.append((rho, theta))
                found_cluster = True
                break
        if not found_cluster: clusters.append([(rho, theta)])
    result = []
    for cluster in clusters:
        final_rho = np.mean([l[0] for l in cluster])
        sum_sin = np.sum([np.sin(l[1]) for l in cluster])
        sum_cos = np.sum([np.cos(l[1]) for l in cluster])
        final_theta = np.arctan2(sum_sin, sum_cos)
        if final_theta < 0: final_theta += 2 * np.pi
        result.append((final_rho, final_theta))
    return sorted(result, key=lambda x: x[0])


# =====================================================================
# 核心算法：DP 筛选
# =====================================================================
def select_n_evenly_spaced(lines, n=19, group_peak_angle=0, external_expected_gap=None):
    if len(lines) <= n: return lines
    lines = sorted(lines, key=lambda l: l[0])
    rhos = np.array([l[0] for l in lines])
    if external_expected_gap and external_expected_gap > 10:
        expected_gap = external_expected_gap
    else:
        span = rhos[-1] - rhos[0]
        expected_gap = span / (n - 1)
    
    cleaned_lines = []
    if len(lines) > 0:
        cleaned_lines.append(lines[0])
        for i in range(1, len(lines)):
            if (lines[i][0] - cleaned_lines[-1][0]) >= 10:
                cleaned_lines.append(lines[i])
    lines = cleaned_lines
    if len(lines) <= n: return lines

    rhos = np.array([l[0] for l in lines])
    thetas = [l[1] for l in lines]
    N = len(rhos)
    typical_gap = expected_gap

    dp = np.full((n, N, N), np.inf)
    parent = np.full((n, N, N), -1, dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            gap = rhos[j] - rhos[i]
            if gap < 0.2 * typical_gap or gap > 3.5 * expected_gap: continue
            ang_diff = circular_angle_diff(thetas[j], thetas[i])
            dp[1][j][i] = abs(gap - typical_gap) * 1.5 + (np.degrees(ang_diff) * 10.0)**2

    for k in range(2, n):
        for j in range(k, N):
            for i in range(k - 1, j):
                if dp[k-1][i].min() == np.inf: continue
                gap_ij = rhos[j] - rhos[i]
                if gap_ij < 0.2 * typical_gap or gap_ij > 3.5 * expected_gap: continue
                ang_diff = circular_angle_diff(thetas[j], thetas[i])
                best_cost, best_p = np.inf, -1
                for p in range(k - 2, i):
                    if dp[k-1][i][p] == np.inf: continue
                    gap_pi = rhos[i] - rhos[p]
                    skip_penalty = abs(gap_ij - expected_gap) * 2.0 if gap_ij > 1.5 * expected_gap else 0
                    cost = dp[k-1][i][p] + abs(gap_ij - gap_pi) + abs(gap_ij - typical_gap) * 0.5 + (np.degrees(ang_diff) * 10.0)**2 + skip_penalty
                    if cost < best_cost: best_cost, best_p = cost, p
                if best_p != -1: dp[k][j][i], parent[k][j][i] = best_cost, best_p

    final_best_cost, best_j, best_i = np.inf, -1, -1
    for i in range(n - 2, N):
        for j in range(i + 1, N):
            if dp[n-1][j][i] != np.inf:
                cost = dp[n-1][j][i] + abs((rhos[j]-rhos[0]) - expected_gap*(n-1)) * 2.0
                if cost < final_best_cost: final_best_cost, best_j, best_i = cost, j, i
    if best_i == -1: return [lines[idx] for idx in np.linspace(0, N-1, n, dtype=int)]
    path = [best_j, best_i]
    curr_k, curr_j, curr_i = n-1, best_j, best_i
    while curr_k > 1:
        p = parent[curr_k][curr_j][curr_i]
        path.append(p)
        curr_j, curr_i, curr_k = curr_i, p, curr_k - 1
    path.reverse()
    return [lines[idx] for idx in path]


# =====================================================================
# 阶段 1: OpenCV 粗检测
# =====================================================================
def find_seed_and_directions(img, debug_show=False):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    edges = cv2.Canny(enhanced, 50, 200)
    raw_segs = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    if raw_segs is None or len(raw_segs) < 10: return None, None, None, 0, 0, 0
    lines_rt = []
    for seg in raw_segs:
        rt = segment_to_rho_theta(*seg[0])
        if rt: lines_rt.append(rt)
    group1, group2, ignored, angle1, angle2 = separate_by_angle(lines_rt, v_tol=35, h_tol=15)
    if not group1 or not group2: return None, None, None, 0, 0, 0
    dir1 = np.array([-np.sin(angle1), np.cos(angle1)])
    dir2 = np.array([-np.sin(angle2), np.cos(angle2)])
    seed = (w // 2, h // 2)
    def median_gap(rhos):
        if len(rhos) < 3: return 60
        diffs = np.diff(sorted(rhos))
        valid = diffs[diffs > 5]
        return np.median(valid) if len(valid) > 3 else 60
    est_gap = (median_gap([r for r,t in group1]) + median_gap([r for r,t in group2])) / 2
    return seed, dir1, dir2, angle1, angle2, est_gap


# =====================================================================
# 阶段 5: 精确拟合
# =====================================================================
def refine_grid_with_known_bounds(img, corners, angle1, angle2, dist_dir1, dist_dir2, debug_show=False):
    h, w = img.shape[:2]
    if len(corners) < 4: return [], []
    tl, tr, br, bl = [np.array(c) for c in corners[:4]]
    board_poly = np.array([tl, tr, br, bl], dtype=np.int32)
    board_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(board_mask, board_poly, 255)
    board_mask = cv2.dilate(board_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)), iterations=1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0).apply(cv2.GaussianBlur(gray, (5,5), 0))
    edges_masked = cv2.bitwise_and(cv2.Canny(enhanced, 50, 200), board_mask)
    raw_segs = cv2.HoughLinesP(edges_masked, 1, np.pi/180, 80, minLineLength=40, maxLineGap=10)
    if raw_segs is None: return [], []
    lines_rt = [segment_to_rho_theta(*seg[0]) for seg in raw_segs]
    lines_rt = [rt for rt in lines_rt if rt]
    g1, g2, _, p1, p2 = separate_by_angle(lines_rt, v_tol=35, h_tol=15)
    c1, c2 = cluster_lines(g1), cluster_lines(g2)
    
    if debug_show:
        debug_img = img.copy()
        def draw_inf_line(canvas, rho, theta, color, thickness):
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
            pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
            cv2.line(canvas, pt1, pt2, color, thickness)
        for r, t in c1: draw_inf_line(debug_img, r, t, (0, 255, 0), 1)
        for r, t in c2: draw_inf_line(debug_img, r, t, (255, 200, 0), 1)
        cv2.imshow("[Refine Grid] Clustered Lines", debug_img)

    sel1 = select_n_evenly_spaced(c1, 19, p1, dist_dir2/18.0)
    sel2 = select_n_evenly_spaced(c2, 19, p2, dist_dir1/18.0)
    
    if debug_show:
        debug_img2 = img.copy()
        for r, t in sel1: draw_inf_line(debug_img2, r, t, (0, 255, 0), 2)
        for r, t in sel2: draw_inf_line(debug_img2, r, t, (255, 200, 0), 2)
        cv2.imshow("[Refine Grid] Final 19x19 Lines", debug_img2)

    return sel1, sel2


# =====================================================================
# Main Class V4_1
# =====================================================================
class HybridScannerV4_1:
    SCAN_RADIUS = 64
    CNN_INPUT_SIZE = 128

    def __init__(self, weights_path, debug_show=True):
        import torch
        from torchvision import transforms
        from PIL import Image
        self.debug_show = debug_show
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        MAC_DIR = Path(__file__).parent
        sys.path.insert(0, str(MAC_DIR.parent / "Classifier_V1"))
        from model import PatchClassifier
        self.model = PatchClassifier(num_classes=4).to(self.device)
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt.get("model") or ckpt.get("model_state_dict") or ckpt, strict=False)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['Corner', 'Inner', 'Edge', 'Outer']
        self._torch, self._Image = torch, Image
        self._patch_vis_boxes, self._patch_vis_canvas = None, None
        self._captured_patches = []

    def classify_patch(self, img, center_xy, scan_radius=None):
        scan_radius = scan_radius or self.SCAN_RADIUS
        h, w = img.shape[:2]
        cx, cy = int(center_xy[0]), int(center_xy[1])
        x1, y1 = max(0, cx-scan_radius), max(0, cy-scan_radius)
        x2, y2 = min(w, cx+scan_radius), min(h, cy+scan_radius)
        if x1 >= x2 or y1 >= y2: return 'Outer', 1.0
        patch_bgr = img[y1:y2, x1:x2]
        ph, pw = patch_bgr.shape[:2]
        target = scan_radius * 2
        if ph < target or pw < target:
            patch_bgr = cv2.copyMakeBorder(patch_bgr, 0, max(0, target-ph), 0, max(0, target-pw), cv2.BORDER_CONSTANT, value=(128,128,128))
        patch_resized = cv2.resize(patch_bgr, (self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE))
        input_tensor = self.transform(self._Image.fromarray(cv2.cvtColor(patch_resized, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(input_tensor), dim=1)[0]
            conf, pred = self._torch.max(probs, 0)
        label, confidence = self.classes[pred.item()], conf.item()
        self._captured_patches.append(patch_resized)
        return label, confidence

    def detect(self, img):
        vis = img.copy()
        seed, d1, d2, a1, a2, gap = find_seed_and_directions(img, self.debug_show)
        if d1 is None: return [], [], [], {}, 0.0, 0.0
        edges = self._cnn_search(img, seed, d1, d2, 30)
        def get_dist(p1, p2): return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 if p1 and p2 else 0.0
        dist1 = get_dist(edges['dir1_pos']['point'], edges['dir1_neg']['point'])
        dist2 = get_dist(edges['dir2_pos']['point'], edges['dir2_neg']['point'])
        corners = self._relocate_precise_corners(img, edges, a1, a2)
        if len(corners) < 4: return corners, [], [], edges, dist1, dist2
        
        debug_canvas = img.copy() if self.debug_show else None
        exact_corners = []
        for i, p in enumerate(corners):
            ec = self.exact_recognize_corner(img, p, ['TL','TR','BR','BL'][i], a1, a2, gap, debug_canvas)
            exact_corners.append(ec)
        
        if self.debug_show and debug_canvas is not None:
            cv2.imshow("[Exact Corner] Tracking Trajectory", debug_canvas)

        sel_h, sel_v = refine_grid_with_known_bounds(img, exact_corners, a1, a2, dist1, dist2, self.debug_show)
        
        # Drawing final result on vis
        for r, t in sel_h:
            p1 = intersect_lines(r, t, sel_v[0][0], sel_v[0][1])
            p2 = intersect_lines(r, t, sel_v[-1][0], sel_v[-1][1])
            if p1 and p2: cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 1)
        for r, t in sel_v:
            p1 = intersect_lines(r, t, sel_h[0][0], sel_h[0][1])
            p2 = intersect_lines(r, t, sel_h[-1][0], sel_h[-1][1])
            if p1 and p2: cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 200, 0), 1)
            
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        labels = ["TL", "TR", "BR", "BL"]
        for i, (cx, cy) in enumerate(exact_corners):
            cv2.circle(vis, (int(cx), int(cy)), 15, colors[i], -1)
            cv2.circle(vis, (int(cx), int(cy)), 17, (255, 255, 255), 2)
            cv2.putText(vis, labels[i], (int(cx)+20, int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if self.debug_show:
            cv2.imshow("Result", vis)
            cv2.waitKey(0)
        return exact_corners, sel_h, sel_v, edges, dist1, dist2

    def _cnn_search(self, img, seed, d1, d2, step):
        edges = {}
        for n, d in [('dir1_pos', d1), ('dir1_neg', -d1), ('dir2_pos', d2), ('dir2_neg', -d2)]:
            p, t, path = self._search_one_direction(img, seed, d, step)
            edges[n] = {'point': p, 'type': t, 'path': path}
        return edges

    def _search_one_direction(self, img, start, direction, step, max_steps=25):
        h, w = img.shape[:2]
        x, y = float(start[0]), float(start[1])
        dx, dy = direction[0] * step, direction[1] * step
        path, last_inner = [], None
        for _ in range(max_steps):
            xi, yi = int(round(x)), int(round(y))
            if xi < 64 or xi >= w-64 or yi < 64 or yi >= h-64: return (last_inner, 'Edge', path) if last_inner else (None, None, path)
            label, conf = self.classify_patch(img, (xi, yi))
            path.append({'x': xi, 'y': yi, 'label': label, 'conf': conf})
            if label == 'Inner': last_inner = (xi, yi)
            elif label in ('Edge', 'Corner'): return (xi, yi), label, path
            elif label == 'Outer': return (self._binary_search(img, last_inner, (xi, yi)), 'Edge', path) if last_inner else (None, 'Outer', path)
            x, y = x + dx, y + dy
        return (last_inner, 'Edge', path) if last_inner else (None, None, path)

    def _binary_search(self, img, inner, outer, max_iter=6):
        ix, iy, ox, oy = float(inner[0]), float(inner[1]), float(outer[0]), float(outer[1])
        for _ in range(max_iter):
            mx, my = (ix+ox)/2, (iy+oy)/2
            label, _ = self.classify_patch(img, (int(mx), int(my)))
            if label == 'Inner': ix, iy = mx, my
            else: ox, oy = mx, my
        return (int(round((ix+ox)/2)), int(round((iy+oy)/2)))

    def _relocate_precise_corners(self, img, edges, a1, a2):
        pts = {k: v['point'] for k, v in edges.items() if v['point']}
        if len(pts) < 4: return []
        def get_line(pt, ang): return (pt[0]*np.cos(ang) + pt[1]*np.sin(ang), ang)
        l1, l2, l3, l4 = get_line(pts['dir1_pos'], a2), get_line(pts['dir1_neg'], a2), get_line(pts['dir2_neg'], a1), get_line(pts['dir2_pos'], a1)
        c = [intersect_lines(l3[0], l3[1], l1[0], l1[1]), intersect_lines(l3[0], l3[1], l2[0], l2[1]), intersect_lines(l4[0], l4[1], l2[0], l2[1]), intersect_lines(l4[0], l4[1], l1[0], l1[1])]
        return self._sort_corners_geometrically([p for p in c if p])

    def _sort_corners_geometrically(self, pts):
        pts = np.array(pts)
        return [tuple(pts[np.argmin(pts[:,0]+pts[:,1])]), tuple(pts[np.argmin(pts[:,1]-pts[:,0])]), tuple(pts[np.argmax(pts[:,0]+pts[:,1])]), tuple(pts[np.argmax(pts[:,1]-pts[:,0])])]

    def _validate_corner_by_harris_neighbors(self, harris_pts, cand, c_type, gap, a1, a2, tol=0.55):
        if not harris_pts: return False, 0.0
        pts = np.array(harris_pts)
        d_along1, d_along2 = np.array([np.cos(a1), np.sin(a1)]), np.array([np.cos(a2), np.sin(a2)])
        dirs = {'TL': [d_along2, d_along1], 'TR': [-d_along2, d_along1], 'BR': [-d_along2, -d_along1], 'BL': [d_along2, -d_along1]}
        for d in dirs[c_type]:
            found = False
            for p in pts:
                dx, dy = p[0]-cand[0], p[1]-cand[1]
                dist = np.sqrt(dx*dx+dy*dy)
                if gap*(1-tol) < dist < gap*(1+tol) and (dx*d[0]+dy*d[1])/dist > 0.866:
                    found = True; break
            if not found: return False, 0.0
        return True, 1.0

    def _snap_to_best_harris_corner(self, roi, hough_pt, x1, y1, c_type, a1, a2, gap, r=80):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dst = cv2.dilate(cv2.cornerHarris(gray, 3, 3, 0.04), None)
        _, thresh = cv2.threshold(dst, 0.05*dst.max() if dst.max()>0 else 1.0, 255, cv2.THRESH_BINARY)
        _, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
        all_h = [(int(c[0])+x1, int(c[1])+y1) for c in centroids[1:]]
        candidates = [p for p in all_h if abs(p[0]-hough_pt[0]) < r and abs(p[1]-hough_pt[1]) < r]
        best_pt, best_s = hough_pt, -1.0
        for cand in candidates:
            v, s = self._validate_corner_by_harris_neighbors(all_h, cand, c_type, gap, a1, a2)
            if v and s > best_s: best_pt, best_s = cand, s
        return best_pt

    def exact_recognize_corner(self, img, approx, c_type, a1, a2, gap, debug=None):
        if c_type not in ['TL', 'TR', 'BR', 'BL']: return approx
        step = 30
        v = {'TL': ((-step, 0), (0, -step)), 'TR': ((step, 0), (0, -step)), 'BL': ((-step, 0), (0, step)), 'BR': ((step, 0), (0, step))}[c_type]
        cx, cy = float(approx[0]), float(approx[1])
        h, w = img.shape[:2]
        for k in range(10):
            label, _ = self.classify_patch(img, (int(cx), int(cy)))
            if label == 'Corner': break
            if label == 'Outer':
                dx, dy = w/2-cx, h/2-cy
                dist = (dx**2+dy**2)**0.5
                if dist>1e-3: cx, cy = cx+(dx/dist)*20, cy+(dy/dist)*20
            elif label == 'Inner': cx, cy = cx+v[0][0]+v[1][0], cy+v[0][1]+v[1][1]
            elif label == 'Edge':
                lx, _ = self.classify_patch(img, (int(cx+v[0][0]), int(cy)))
                ly, _ = self.classify_patch(img, (int(cx), int(cy+v[1][1])))
                if lx == 'Corner': cx += v[0][0]; continue
                if ly == 'Corner': cy += v[1][1]; continue
                if lx in ['Inner','Edge']: cx += v[0][0]
                elif ly in ['Inner','Edge']: cy += v[1][1]
                else: cx, cy = cx-v[0][0], cy-v[1][1]
        
        # Intermediate Visualization: Showing ROI for each corner during refinement
        final_pt = self._extract_precise_opencv_l_shape(img, (cx, cy), c_type, a1, a2, gap, debug)
        return final_pt

    def _extract_precise_opencv_l_shape(self, img, center, c_type, a1, a2, gap, debug=None, r=100):
        h, w = img.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        final_h_pt = center
        for attempt in range(3):
            curr_r = int(r * (1.5**attempt))
            x1, y1 = max(0, cx-curr_r), max(0, cy-curr_r)
            x2, y2 = min(w-1, cx+curr_r), min(h-1, cy+curr_r)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0: break
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.createCLAHE(clipLimit=2.0).apply(cv2.GaussianBlur(gray, (5,5), 0)), 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, 20, 10)
            h_r, v_r = [], []
            if lines is not None:
                for s in lines:
                    rt = segment_to_rho_theta(s[0][0]+x1, s[0][1]+y1, s[0][2]+x1, s[0][3]+y1)
                    if rt and circular_angle_diff(rt[1], a1) < 0.5: h_r.append(rt)
                    elif rt and circular_angle_diff(rt[1], a2) < 0.5: v_r.append(rt)
            h_pt = center
            if h_r and v_r:
                h_sorted = sorted(h_r, key=lambda x: x[0], reverse='B' in c_type)
                v_sorted = sorted(v_r, key=lambda x: x[0], reverse='R' in c_type)
                h_line = h_sorted[1] if len(h_sorted)>1 else h_sorted[0]
                v_line = v_sorted[1] if len(v_sorted)>1 else v_sorted[0]
                res = intersect_lines(h_line[0], h_line[1], v_line[0], v_line[1])
                if res: h_pt = res
            best = self._snap_to_best_harris_corner(roi, h_pt, x1, y1, c_type, a1, a2, gap)
            
            if debug is not None:
                roi_vis = roi.copy()
                lx, ly = int(best[0]-x1), int(best[1]-y1)
                cv2.circle(roi_vis, (lx, ly), 5, (0,0,255), -1)
                cv2.imshow(f"ROI_{c_type}_Attempt_{attempt}", roi_vis)

            if best != h_pt: return best
            final_h_pt = h_pt
        return final_h_pt
