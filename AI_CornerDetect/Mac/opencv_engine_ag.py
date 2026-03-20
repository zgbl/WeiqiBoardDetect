"""
opencv_engine_ag.py — AG V3: 修复聚类+预过滤导致线不够的问题
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


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

def draw_line_rt(canvas, rho, theta, color, thickness=1):
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho, b*rho
    cv2.line(canvas, (int(x0-4000*b), int(y0+4000*a)),
             (int(x0+4000*b), int(y0-4000*a)), color, thickness)


# ==================== 阶段1: HoughLinesP ====================

def detect_line_segments(gray, debug_show=False):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 200)
    if debug_show: cv2.imshow("[AG] Edges", edges)

    segs = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    if segs is None:
        segs = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=30, maxLineGap=15)
    if segs is None: return []

    result = []
    for s in segs:
        r = segment_to_rho_theta(*s[0])
        if r: result.append(r)
    print(f"[AG] HoughLinesP: {len(segs)} segments -> {len(result)} lines")
    return result


# ==================== 阶段2: 分组 ====================

def separate_two_directions(lines, h_tol=10, v_tol=18, debug_show=False, img=None):
    if len(lines) < 2: return [], [], [], 0, 0

    normalized = [normalize_line(r, t) for r, t in lines]
    mapped = np.array([t % np.pi for _, t in normalized])

    n_bins, bin_sz = 180, np.pi/180
    hist = np.zeros(n_bins)
    for t in mapped: hist[int(t/bin_sz) % n_bins] += 1

    k = 7; ker = np.ones(k)/k
    pad = np.concatenate([hist[-k:], hist, hist[:k]])
    sm = np.convolve(pad, ker, mode='same')[k:-k]

    # 水平优先
    p1_bin = -1; best = -1
    for i in range(n_bins):
        a = (i+0.5)*bin_sz
        if circular_angle_diff(a, np.pi/2) < np.radians(10) and sm[i] > best:
            best = sm[i]; p1_bin = i
    if p1_bin == -1 or best < 1e-3: p1_bin = int(np.argmax(sm))
    p1_angle = (p1_bin+0.5)*bin_sz

    target = (p1_angle + np.pi/2) % np.pi
    p2_bin = -1; best2 = -1
    for i in range(n_bins):
        a = (i+0.5)*bin_sz
        if circular_angle_diff(a, target) < np.radians(15) and sm[i] > best2:
            best2 = sm[i]; p2_bin = i
    if p2_bin == -1 or best2 < 0.05*sm[p1_bin]:
        sup = sm.copy()
        for i in range(n_bins):
            if circular_angle_diff((i+0.5)*bin_sz, p1_angle) < np.radians(30): sup[i] = 0
        p2_bin = int(np.argmax(sup))
    p2_angle = (p2_bin+0.5)*bin_sz

    sep = np.degrees(circular_angle_diff(p1_angle, p2_angle))
    print(f"[AG] Peaks: {np.degrees(p1_angle):.1f}° / {np.degrees(p2_angle):.1f}° (sep {sep:.1f}°)")

    def get_tol(a):
        return (np.radians(h_tol), "H") if circular_angle_diff(a, np.pi/2) < np.radians(45) else (np.radians(v_tol), "V")

    tol1, t1 = get_tol(p1_angle); tol2, t2 = get_tol(p2_angle)
    g1, g2, ign = [], [], []
    for i, (r, t) in enumerate(normalized):
        mt = mapped[i]
        d1 = circular_angle_diff(mt, p1_angle)
        d2 = circular_angle_diff(mt, p2_angle)
        if d1 < tol1 and d1 <= d2: g1.append((r, t))
        elif d2 < tol2: g2.append((r, t))
        else: ign.append((r, t))

    print(f"[AG] Groups: {t1}={len(g1)}, {t2}={len(g2)}, Ign={len(ign)}")

    if debug_show and img is not None:
        vis = img.copy()
        for r,t in ign: draw_line_rt(vis, r, t, (0,0,200), 1)
        for r,t in g1: draw_line_rt(vis, r, t, (0,255,0), 1)
        for r,t in g2: draw_line_rt(vis, r, t, (255,200,0), 1)
        cv2.imshow("[AG] Line Groups", vis)

    return g1, g2, ign, p1_angle, p2_angle


# ==================== 阶段3: 自适应聚类 ====================

def cluster_lines(lines, rho_thresh=6, theta_thresh_deg=2):
    if not lines: return []
    lines = sorted(lines, key=lambda l: l[0])
    t_thr = np.radians(theta_thresh_deg)
    clusters = []
    for rho, theta in lines:
        merged = False
        for cl in clusters:
            ar = np.mean([c[0] for c in cl])
            at = np.arctan2(sum(np.sin(c[1]) for c in cl), sum(np.cos(c[1]) for c in cl))
            if abs(rho - ar) < rho_thresh and circular_angle_diff(theta, at) < t_thr:
                cl.append((rho, theta)); merged = True; break
        if not merged: clusters.append([(rho, theta)])

    result = []
    for cl in clusters:
        ar = np.mean([c[0] for c in cl])
        at = np.arctan2(sum(np.sin(c[1]) for c in cl), sum(np.cos(c[1]) for c in cl))
        if at < 0: at += 2*np.pi
        result.append((ar, at, len(cl)))  # 含簇大小
    return sorted(result, key=lambda x: x[0])


def adaptive_cluster(lines, n=19):
    """自适应聚类：迭代增大rho阈值直到簇数在合理范围"""
    if not lines: return []
    rhos = sorted([l[0] for l in lines])
    span = rhos[-1] - rhos[0]
    approx_gap = span / max(n-1, 1)
    
    # 从 0.15*gap 开始，逐步增大
    for factor in [0.15, 0.20, 0.25, 0.30, 0.35]:
        rt = max(8, factor * approx_gap)
        clustered = cluster_lines(lines, rho_thresh=rt, theta_thresh_deg=2)
        if len(clustered) <= 50:
            print(f"[AG] Cluster: rho_thresh={rt:.1f}, {len(lines)}->{len(clustered)}")
            return clustered
    
    # fallback
    clustered = cluster_lines(lines, rho_thresh=0.4*approx_gap)
    print(f"[AG] Cluster fallback: {len(lines)}->{len(clustered)}")
    return clustered


# ==================== 阶段4: 选19条线 (3D DP) ====================

def select_n_lines(clustered, n=19, group_peak_angle=0):
    # clustered 元素是 (rho, theta, cluster_size)
    lines_rt = [(r, t) for r, t, _ in clustered]
    sizes = [s for _, _, s in clustered]
    
    if len(lines_rt) <= n:
        return lines_rt

    lines_rt = sorted(lines_rt, key=lambda l: l[0])
    rhos = np.array([l[0] for l in lines_rt])
    N = len(rhos)
    span = rhos[-1] - rhos[0]
    if span < 1: return lines_rt[:n]
    expected_gap = span / (n-1)

    # 预过滤: 0.35 * expected_gap (比之前的0.7宽松很多)
    dedup_thr = 0.35 * expected_gap
    cleaned = [0]  # indices
    for i in range(1, N):
        if rhos[i] - rhos[cleaned[-1]] < dedup_thr:
            # 保留簇更大的那条
            if sizes[i] > sizes[cleaned[-1]]:
                cleaned[-1] = i
        else:
            cleaned.append(i)

    # 安全: 若过滤后不够, 跳过预过滤
    if len(cleaned) < n + 3:
        print(f"[AG] Pre-filter too aggressive ({len(cleaned)}<{n+3}), keeping all {N}")
        cleaned = list(range(N))
    else:
        print(f"[AG] Pre-filter: {N} -> {len(cleaned)} (thr={dedup_thr:.1f})")

    # 重建
    filt_lines = [lines_rt[i] for i in cleaned]
    filt_rhos = np.array([l[0] for l in filt_lines])
    M = len(filt_rhos)
    
    if M <= n: return filt_lines

    # 重新计算 typical_gap
    diffs = np.diff(filt_rhos)
    eg = (filt_rhos[-1] - filt_rhos[0]) / (n-1)
    valid = diffs[diffs > 0.5 * eg]
    tg = np.median(valid) if len(valid) >= 5 else eg
    tg = max(tg, 0.7 * eg)
    print(f"[AG] DP input: {M} lines, typical_gap={tg:.1f}, expected={eg:.1f}")

    # === DP ===
    if M <= 50:
        # 3D DP: dp[k][j][i], 惩罚相邻间距突变
        INF = 1e18
        dp = np.full((n, M, M), INF)
        par = np.full((n, M, M), -1, dtype=int)
        for i in range(M):
            for j in range(i+1, M):
                g = filt_rhos[j] - filt_rhos[i]
                if g >= 0.5 * tg:
                    dp[1][j][i] = abs(g - tg) * 1.5
        for k in range(2, n):
            has_fin = np.any(dp[k-1] < INF, axis=1)
            for j in range(k, M):
                for i in range(k-1, j):
                    if not has_fin[i]: continue
                    gij = filt_rhos[j] - filt_rhos[i]
                    if gij < 0.5*tg or gij > 3.0*tg: continue
                    for p in range(max(0, k-2), i):
                        if dp[k-1][i][p] >= INF: continue
                        gpi = filt_rhos[i] - filt_rhos[p]
                        cost = dp[k-1][i][p] + abs(gij-gpi) + abs(gij-tg)*0.5
                        if cost < dp[k][j][i]:
                            dp[k][j][i] = cost; par[k][j][i] = p
        # 回溯
        best_c = INF; bj, bi = -1, -1
        for i in range(n-2, M):
            for j in range(i+1, M):
                if dp[n-1][j][i] < best_c:
                    best_c = dp[n-1][j][i]; bj, bi = j, i
        if bi == -1:
            print("[AG] 3D-DP failed, uniform sampling")
            return [filt_lines[i] for i in np.linspace(0, M-1, n, dtype=int)]
        path = [bj, bi]
        ck, cj, ci = n-1, bj, bi
        while ck > 1:
            p = par[ck][cj][ci]; path.append(p); cj, ci = ci, p; ck -= 1
        path.reverse()
    else:
        # 2D DP fallback for large N
        INF = 1e18
        dp2 = np.full((n, M), INF)
        par2 = np.full((n, M), -1, dtype=int)
        for j in range(M): dp2[0][j] = 0.0
        for k in range(1, n):
            for j in range(k, M):
                for i in range(k-1, j):
                    if dp2[k-1][i] >= INF: continue
                    g = filt_rhos[j] - filt_rhos[i]
                    if g < 0.4*tg: continue
                    if g > 3.0*tg: break
                    cost = dp2[k-1][i] + (g - tg)**2
                    if cost < dp2[k][j]:
                        dp2[k][j] = cost; par2[k][j] = i
        bj = int(np.argmin(dp2[n-1]))
        if dp2[n-1][bj] >= INF:
            return [filt_lines[i] for i in np.linspace(0, M-1, n, dtype=int)]
        path = [bj]
        for k in range(n-1, 0, -1): path.append(par2[k][path[-1]])
        path.reverse()

    sel = [filt_lines[i] for i in path]
    gaps = np.diff([l[0] for l in sel])
    print(f"[AG] Selected {n}: gaps [{gaps.min():.1f}, {gaps.max():.1f}], "
          f"mean={gaps.mean():.1f}, std={gaps.std():.1f}")
    return sel


# ==================== 阶段5: 角点 ====================

def find_board_corners(intersections, img_shape, shrink=0.05):
    if len(intersections) < 4: return intersections
    h, w = img_shape[:2]
    pts = np.array(intersections, dtype=np.float32)
    sums = pts[:,0] + pts[:,1]
    diffs = pts[:,0] - pts[:,1]
    tl = pts[np.argmin(sums)]; br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]; bl = pts[np.argmin(diffs)]
    corners = [tuple(tl), tuple(tr), tuple(br), tuple(bl)]
    cx = sum(c[0] for c in corners)/4
    cy = sum(c[1] for c in corners)/4
    return [(max(10,min(w-10, x+(cx-x)*shrink)), max(10,min(h-10, y+(cy-y)*shrink)))
            for x,y in corners]


# ==================== 透视矫正 ====================

def detect_board_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for canny_lo, canny_hi in [(50,150), (30,100), (80,200)]:
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, canny_lo, canny_hi)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        proc = cv2.dilate(edges, kern, iterations=3)
        proc = cv2.erode(proc, kern, iterations=2)
        contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < 50000: break
            for eps_r in [0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(cnt, eps_r * cv2.arcLength(cnt, True), True)
                if len(approx) == 4: return approx, area
            # 尝试 minAreaRect 降级
            if len(approx) >= 4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(np.float32)
                if cv2.contourArea(box.astype(np.int32)) > 50000:
                    return box.reshape(-1,1,2).astype(np.int32), area
    return None, 0

def order_quad(pts):
    pts = pts.reshape(-1,2).astype(np.float32)
    s = pts[:,0]+pts[:,1]; d = pts[:,0]-pts[:,1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)

def warp_perspective(img, contour):
    if contour is None: return img, None, None
    src = order_quad(contour)
    w = int(max(np.linalg.norm(src[1]-src[0]), np.linalg.norm(src[2]-src[3])))
    h = int(max(np.linalg.norm(src[3]-src[0]), np.linalg.norm(src[2]-src[1])))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w,h)), M, src


# ==================== 主引擎 ====================

class OpenCVDetector:
    def __init__(self, debug_show=True):
        self.debug_show = debug_show

    def _pipeline(self, img, tag="[AG]"):
        """完整管线: 检测->分组->聚类->选线->交点"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = detect_line_segments(gray, debug_show=self.debug_show)
        if len(lines) < 10:
            print(f"{tag} Too few lines"); return [],[],[]

        g1, g2, ign, p1, p2 = separate_two_directions(
            lines, h_tol=10, v_tol=18, debug_show=self.debug_show, img=img)
        if len(g1)<5 or len(g2)<5:
            print(f"{tag} Not enough groups"); return [],[],[]

        c1 = adaptive_cluster(g1, n=19)
        c2 = adaptive_cluster(g2, n=19)
        print(f"{tag} Clustered: G1={len(c1)}, G2={len(c2)}")

        print(f"{tag} Select G1:")
        s1 = select_n_lines(c1, 19, p1)
        print(f"{tag} Select G2:")
        s2 = select_n_lines(c2, 19, p2)
        print(f"{tag} Final: G1={len(s1)}, G2={len(s2)}")

        inters = []
        for r1,t1 in s1:
            for r2,t2 in s2:
                pt = intersect_lines(r1,t1,r2,t2)
                if pt:
                    x,y = pt
                    if -50<=x<=w+50 and -50<=y<=h+50:
                        inters.append((x,y))
        print(f"{tag} Intersections: {len(inters)}")
        return s1, s2, inters

    def find_corners(self, img):
        h, w = img.shape[:2]
        print("\n=== OpenCV Module (AG Engine V3) ===")
        print(f"[AG] Image size: {w}x{h}")

        # 尝试透视矫正
        contour, area = detect_board_contour(img)
        use_warp = False

        if contour is not None:
            print(f"[AG] Contour found, area={area}")
            warped, M, src = warp_perspective(img, contour)
            if self.debug_show:
                ci = img.copy()
                cv2.drawContours(ci, [contour], -1, (0,255,0), 3)
                cv2.imshow("[AG] Contour", ci)
                cv2.imshow("[AG] Warped", warped)
            s1, s2, inters = self._pipeline(warped, "[AG-W]")
            if len(inters) >= 100:
                corners_w = find_board_corners(inters, warped.shape, 0.05)
                if len(corners_w) >= 4 and M is not None:
                    Mi = cv2.invert(M)[1]
                    corners = []
                    for x,y in corners_w:
                        p = cv2.perspectiveTransform(
                            np.array([[[x,y]]], dtype=np.float32), Mi)[0][0]
                        corners.append((max(10,min(w-10,p[0])),
                                       max(10,min(h-10,p[1]))))
                    use_warp = True
                    final = corners
                    print("[AG] Using warped result")
        else:
            print("[AG] No contour, direct detection")

        if not use_warp:
            s1, s2, inters = self._pipeline(img, "[AG]")
            if len(s1)<2 or len(s2)<2:
                print("[AG] Insufficient lines!"); return []
            final = find_board_corners(inters, img.shape, 0.05)

        # 可视化
        if self.debug_show:
            self._vis(img, s1, s2, final, use_warp)

        print(f"[AG] Corners: {len(final)}")
        for i,(x,y) in enumerate(final):
            lb = ["TL","TR","BR","BL"][i] if i<4 else f"C{i+1}"
            print(f"  {lb}: ({int(x)}, {int(y)})")
        print("=" * 45 + "\n")
        return final

    def _vis(self, img, s1, s2, corners, warped=False):
        h, w = img.shape[:2]
        if len(s1)>=2 and len(s2)>=2:
            li = img.copy()
            for r,t in s1:
                a = intersect_lines(r,t,s2[0][0],s2[0][1])
                b = intersect_lines(r,t,s2[-1][0],s2[-1][1])
                if a and b: cv2.line(li,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),(0,255,0),1)
            for r,t in s2:
                a = intersect_lines(r,t,s1[0][0],s1[0][1])
                b = intersect_lines(r,t,s1[-1][0],s1[-1][1])
                if a and b: cv2.line(li,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),(255,200,0),1)
            cv2.imshow("[AG] Selected Grid Lines", li)

        ci = img.copy()
        for r1,t1 in s1:
            for r2,t2 in s2:
                pt = intersect_lines(r1,t1,r2,t2)
                if pt:
                    x,y = int(pt[0]),int(pt[1])
                    if -50<=x<=w+50 and -50<=y<=h+50:
                        cv2.circle(ci,(x,y),3,(0,200,0),-1)
        lbs = ["TL","TR","BR","BL"]
        cols = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        for i,(x,y) in enumerate(corners):
            c = cols[i] if i<4 else (255,255,255)
            cv2.circle(ci,(int(x),int(y)),15,c,-1)
            cv2.circle(ci,(int(x),int(y)),17,(255,255,255),2)
            lb = lbs[i] if i<4 else f"C{i}"
            cv2.putText(ci,lb,(int(x)+20,int(y)-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.imshow("[AG] Corners" + (" (warp)" if warped else ""), ci)
