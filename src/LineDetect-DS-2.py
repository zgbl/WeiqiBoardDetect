# 这是Deepseek 给的第2版，他看过 Claude的 5-2 版本
import cv2
import numpy as np
from sklearn.cluster import KMeans

def preprocess_image(img):
    """增强图像预处理"""
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # 二值化
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def detect_lines(image):
    """使用优化的Hough变换检测线条"""
    # 边缘检测
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # 形态学操作增强线条
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Hough变换参数优化
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,  # 降低阈值以检测更多线条
        minLineLength=100,  # 增加最小长度要求
        maxLineGap=10  # 允许适当的间隙
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
        
        # 分类为水平或垂直线
        if (angle < 10) or (angle > 170):  # 水平线
            # 过滤边缘线条
            y_avg = (y1 + y2) / 2
            if 0.1 * img_shape[0] < y_avg < 0.9 * img_shape[0]:
                if abs(x2 - x1) > 0.5 * img_shape[1]:  # 长度要求
                    horizontals.append((x1, y1, x2, y2))
        elif (80 < angle < 100):  # 垂直线
            # 过滤边缘线条
            x_avg = (x1 + x2) / 2
            if 0.1 * img_shape[1] < x_avg < 0.9 * img_shape[1]:
                if abs(y2 - y1) > 0.5 * img_shape[0]:  # 长度要求
                    verticals.append((x1, y1, x2, y2))
    
    return horizontals, verticals

def cluster_lines(lines, n_clusters, axis='y'):
    """使用K-means聚类线条"""
    if not lines or n_clusters <= 0:
        return []
    
    # 提取坐标特征
    coords = []
    for x1, y1, x2, y2 in lines:
        if axis == 'y':  # 水平线聚类
            coords.append([(y1 + y2) / 2])
        else:  # 垂直线聚类
            coords.append([(x1 + x2) / 2])
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(coords)
    
    # 生成聚类后的线条
    clustered_lines = []
    for i in range(n_clusters):
        cluster_points = [lines[j] for j in range(len(lines)) if labels[j] == i]
        if cluster_points:
            xs = []
            ys = []
            for x1, y1, x2, y2 in cluster_points:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            
            if axis == 'y':  # 水平线
                y_avg = int(np.mean(ys))
                x_min = min(xs)
                x_max = max(xs)
                clustered_lines.append((x_min, y_avg, x_max, y_avg))
            else:  # 垂直线
                x_avg = int(np.mean(xs))
                y_min = min(ys)
                y_max = max(ys)
                clustered_lines.append((x_avg, y_min, x_avg, y_max))
    
    return clustered_lines

def regularize_grid(h_lines, v_lines, n=19):
    """规整化棋盘网格"""
    if not h_lines or not v_lines:
        return [], []
    
    # 提取位置信息
    h_positions = sorted([(y1 + y2) / 2 for x1, y1, x2, y2 in h_lines])
    v_positions = sorted([(x1 + x2) / 2 for x1, y1, x2, y2 in v_lines])
    
    # 计算理想间距
    h_spacing = (h_positions[-1] - h_positions[0]) / (n - 1)
    v_spacing = (v_positions[-1] - v_positions[0]) / (n - 1)
    
    # 生成规整线条
    regular_h = []
    regular_v = []
    
    # 横线
    x_min = min(v_positions)
    x_max = max(v_positions)
    for i in range(n):
        y = int(h_positions[0] + i * h_spacing)
        regular_h.append((x_min, y, x_max, y))
    
    # 竖线
    y_min = min(h_positions)
    y_max = max(h_positions)
    for i in range(n):
        x = int(v_positions[0] + i * v_spacing)
        regular_v.append((x, y_min, x, y_max))
    
    return regular_h, regular_v

def main():
    # 读取图像
    img = cv2.imread("../data/raw/bd317d54.webp")
    if img is None:
        print("无法读取图像文件")
        return
    
    # 显示原始图像
    cv2.imshow("Original Image", img)
    
    # 图像预处理
    processed = preprocess_image(img)
    cv2.imshow("Processed Image", processed)
    
    # 检测线条
    lines = detect_lines(processed)
    
    if lines is None:
        print("未检测到任何线条")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    # 分类和过滤线条
    h_lines, v_lines = classify_and_filter_lines(lines, img.shape)
    
    # 聚类线条
    n_lines = 19  # 19x19围棋棋盘
    clustered_h = cluster_lines(h_lines, n_lines, 'y')
    clustered_v = cluster_lines(v_lines, n_lines, 'x')
    
    # 规整化网格
    regular_h, regular_v = regularize_grid(clustered_h, clustered_v, n_lines)
    
    # 绘制结果
    result = img.copy()
    
    # 绘制原始检测线条(绿色)
    for line in h_lines + v_lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # 绘制规整化线条(红色)
    for line in regular_h + regular_v:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 显示结果
    cv2.imshow("Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()