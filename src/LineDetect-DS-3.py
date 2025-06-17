# 这是Deepseek 给的第3版，他看过 Claude的 5-2 版本
# not working, - Can't parse 'pt1'. Sequence item with index 0 has a wrong type

import cv2
import numpy as np
from sklearn.cluster import KMeans

def preprocess_image(img):
    """增强图像预处理"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def detect_lines(image):
    """使用优化的Hough变换检测线条"""
    edges = cv2.Canny(image, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=50,
        maxLineGap=20
    )
    return lines

def classify_and_filter_lines(lines, img_shape):
    """分类并过滤线条"""
    if lines is None:
        return [], []
    
    horizontals = []
    verticals = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        
        if (angle < 15) or (angle > 165):  # 放宽水平线角度阈值
            y_avg = (y1 + y2) / 2
            if 0.05 * img_shape[0] < y_avg < 0.95 * img_shape[0]:  # 放宽边缘限制
                if abs(x2 - x1) > 0.3 * img_shape[1]:  # 降低长度要求
                    horizontals.append((x1, y1, x2, y2))
        elif (75 < angle < 105):  # 放宽垂直线角度阈值
            x_avg = (x1 + x2) / 2
            if 0.05 * img_shape[1] < x_avg < 0.95 * img_shape[1]:
                if abs(y2 - y1) > 0.3 * img_shape[0]:
                    verticals.append((x1, y1, x2, y2))
    
    return horizontals, verticals

def cluster_lines(lines, n_clusters, axis='y'):
    """使用K-means聚类线条"""
    if not lines or n_clusters <= 0:
        return []
    
    if len(lines) < n_clusters:
        print(f"警告: 检测到的{axis}方向线条数量({len(lines)})少于要求的聚类数({n_clusters})")
        return lines
    
    coords = []
    for x1, y1, x2, y2 in lines:
        if axis == 'y':
            coords.append([(y1 + y2) / 2])
        else:
            coords.append([(x1 + x2) / 2])
    
    kmeans = KMeans(n_clusters=min(n_clusters, len(lines)), n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    
    clustered_lines = []
    for i in range(min(n_clusters, len(lines))):
        cluster_points = [lines[j] for j in range(len(lines)) if labels[j] == i]
        if cluster_points:
            xs = []
            ys = []
            for x1, y1, x2, y2 in cluster_points:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            
            if axis == 'y':
                y_avg = int(np.mean(ys))
                x_min = min(xs)
                x_max = max(xs)
                clustered_lines.append((x_min, y_avg, x_max, y_avg))
            else:
                x_avg = int(np.mean(xs))
                y_min = min(ys)
                y_max = max(ys)
                clustered_lines.append((x_avg, y_min, x_avg, y_max))
    
    return clustered_lines

def regularize_grid(h_lines, v_lines, n=9):
    """规整化棋盘网格"""
    if not h_lines or not v_lines:
        return [], []
    
    h_positions = sorted([(y1 + y2) / 2 for x1, y1, x2, y2 in h_lines])
    v_positions = sorted([(x1 + x2) / 2 for x1, y1, x2, y2 in v_lines])
    
    # 如果线条不足，均匀补全
    if len(h_positions) < n:
        h_spacing = (h_positions[-1] - h_positions[0]) / (len(h_positions) - 1) if len(h_positions) > 1 else 20
        h_positions = [h_positions[0] + i * h_spacing for i in range(n)]
    
    if len(v_positions) < n:
        v_spacing = (v_positions[-1] - v_positions[0]) / (len(v_positions) - 1) if len(v_positions) > 1 else 20
        v_positions = [v_positions[0] + i * v_spacing for i in range(n)]
    
    regular_h = []
    regular_v = []
    
    x_min = min(v_positions)
    x_max = max(v_positions)
    for y in [int(pos) for pos in h_positions]:
        regular_h.append((x_min, y, x_max, y))
    
    y_min = min(h_positions)
    y_max = max(h_positions)
    for x in [int(pos) for pos in v_positions]:
        regular_v.append((x, y_min, x, y_max))
    
    return regular_h, regular_v

def main():
    img = cv2.imread("../data/raw/Pic1.jpg")
    if img is None:
        print("无法读取图像文件")
        return
    
    cv2.imshow("Original Image", img)
    processed = preprocess_image(img)
    cv2.imshow("Processed Image", processed)
    
    lines = detect_lines(processed)
    
    if lines is None:
        print("未检测到任何线条")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    h_lines, v_lines = classify_and_filter_lines(lines, img.shape)
    
    print(f"检测到水平线: {len(h_lines)}条, 垂直线: {len(v_lines)}条")
    
    n_lines = 9
    clustered_h = cluster_lines(h_lines, n_lines, 'y')
    clustered_v = cluster_lines(v_lines, n_lines, 'x')
    
    regular_h, regular_v = regularize_grid(clustered_h, clustered_v, n_lines)
    
    result = img.copy()
    for line in h_lines + v_lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    for line in regular_h + regular_v:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow("Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()