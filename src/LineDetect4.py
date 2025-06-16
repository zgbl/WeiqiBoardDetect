import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

def is_similar(line1, line2, dist_thresh=3, angle_thresh=5):
    x1, y1, x2, y2 = line1
    a1 = np.arctan2(y2 - y1, x2 - x1)
    x3, y3, x4, y4 = line2
    a2 = np.arctan2(y4 - y3, x4 - x3)
    angle_diff = abs(a1 - a2) * 180 / np.pi
    angle_diff = min(angle_diff, 180 - angle_diff)

    # 判断角度是否接近
    if angle_diff > angle_thresh:
        return False

    # 判断起点和终点是否接近（任选一个点即可）
    dist = np.hypot(x1 - x3, y1 - y3)
    return dist < dist_thresh


def group_lines_by_orientation(lines, angle_threshold=10):
    horizontal = []
    vertical = []

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_threshold or abs(angle + 90) < angle_threshold:
            vertical.append((x1, y1, x2, y2))
    
    return horizontal, vertical

def fit_lines(line_group, axis='h'):
    # axis='h': 横线，按 y 聚合；axis='v': 纵线，按 x 聚合
    bins = defaultdict(list)
    for x1, y1, x2, y2 in line_group:
        key = y1 if axis == 'h' else x1
        bins[key // 5 * 5].append((x1, y1, x2, y2))  # 用 5 像素为窗口聚合

    fitted_lines = []
    for group in bins.values():
        xs, ys = [], []
        for x1, y1, x2, y2 in group:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        if axis == 'h':
            y_avg = int(np.mean(ys))
            x_min, x_max = min(xs), max(xs)
            fitted_lines.append((x_min, y_avg, x_max, y_avg))
        else:
            x_avg = int(np.mean(xs))
            y_min, y_max = min(ys), max(ys)
            fitted_lines.append((x_avg, y_min, x_avg, y_max))
    return fitted_lines

def classify_lines_by_angle(lines, angle_thresh=10):
    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < angle_thresh or abs(angle - 180) < angle_thresh:
            horizontals.append((x1, y1, x2, y2))
        elif abs(angle - 90) < angle_thresh or abs(angle + 90) < angle_thresh:
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals

def cluster_lines(line_group, n_clusters, axis='h'):
    # 取横向：按 y 聚类，取竖向：按 x 聚类
    coords = []
    for x1, y1, x2, y2 in line_group:
        coord = (y1 + y2) / 2 if axis == 'h' else (x1 + x2) / 2
        coords.append([coord])

    if len(coords) < n_clusters:
        return []  # 不足以聚类

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(coords)
    centers = sorted([int(c[0]) for c in kmeans.cluster_centers_])

    # 对每个中心，选取该簇中所有点，拟合出一条代表线
    final_lines = []
    for i, center in enumerate(centers):
        group_lines = [line for idx, line in enumerate(line_group) if labels[idx] == i]
        xs, ys = [], []
        for x1, y1, x2, y2 in group_lines:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        if axis == 'h':
            x_min, x_max = min(xs), max(xs)
            final_lines.append((x_min, center, x_max, center))
        else:
            y_min, y_max = min(ys), max(ys)
            final_lines.append((center, y_min, center, y_max))
    return final_lines

# 1. 读图 & 转灰度
#img = cv2.imread("../data/raw/BasicStraightLine2.jpg")
img = cv2.imread("../data/raw/Pic1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#先做高斯模糊 + 提升对比度
blur = cv2.GaussianBlur(gray, (3, 3), 0)
equalized = cv2.equalizeHist(blur)

#显示原图
cv2.imshow("Original Pic", img) 
cv2.moveWindow("Original Pic", 0, 0)

# 2. 边缘检测
edges = cv2.Canny(gray, 40, 160, apertureSize=3)
#edges = cv2.Canny(equalized, 50, 150)   # 提升了对比度后的Canny

print("Equalized shape:", equalized.shape, "dtype:", equalized.dtype)
print("Equalized min/max:", np.min(equalized), np.max(equalized))
cv2.imshow("Equalizeed", equalized)
cv2.moveWindow("Equalizeed", 600, 0)

# 3. 使用 HoughLinesP 检测“线段”（不是无限延长线）
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=0, 
                        minLineLength=20, 
                        maxLineGap=5)


# 2. 去重处理
filtered_lines = []


# 4. 画线（设置为和原图一致的线宽：1）
lineCount = 0
if lines is not None:

    # 拆分横线和竖线
    horizontals, verticals = group_lines_by_orientation(lines)

    # 分组并拟合
    merged_h = fit_lines(horizontals, axis='h')
    merged_v = fit_lines(verticals, axis='v')

    # 合并后画线
    for x1, y1, x2, y2 in merged_h + merged_v:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# 5. 显示
cv2.imshow("Detected Line", img)
cv2.moveWindow("Detected Line", 0, 300)
cv2.waitKey(0)
cv2.destroyAllWindows()