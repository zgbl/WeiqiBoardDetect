import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tkinter import Tk, filedialog

class ImprovedGoBoardAnalyzer:
    def __init__(self):
        self.board_size = 19
        self.corners = []
        self.grid_points = []
        self.stones = []
        
    def upload_and_load_image(self):
        """上传并加载图像"""
        print("请选择围棋棋盘图像...")
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="选择围棋棋盘图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            raise ValueError("未选择文件")
        
        image = Image.open(file_path)
        self.original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.image = self.original_image.copy()
        
        print(f"图像加载成功: {os.path.basename(file_path)}")
        print(f"图像尺寸: {self.image.shape}")
        return self.image
    
    def merge_similar_lines(self, lines, rho_threshold=10, theta_threshold=0.1):
        """合并相似的直线"""
        if lines is None or len(lines) == 0:
            return []
        
        merged_lines = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
                
            rho1, theta1 = line1[0]
            similar_lines = [(rho1, theta1)]
            used[i] = True
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if used[j]:
                    continue
                    
                rho2, theta2 = line2[0]
                
                # 修复：显式转换为标量并比较
                rho_diff = abs(float(rho1) - float(rho2))
                theta_diff = abs(float(theta1) - float(theta2))
                
                # 检查是否相似
                if rho_diff < rho_threshold and theta_diff < theta_threshold:
                    similar_lines.append((rho2, theta2))
                    used[j] = True
            
            # 计算平均值作为合并后的线
            avg_rho = np.mean([rho for rho, theta in similar_lines])
            avg_theta = np.mean([theta for rho, theta in similar_lines])
            merged_lines.append([[avg_rho, avg_theta]])
        
        return merged_lines
    
    def detect_board_corners_improved(self):
        """改进的棋盘角点检测"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # 边缘检测 - 使用更合适的阈值
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # 形态学操作增强线条
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 霍夫直线检测 - 使用更高的阈值减少噪声
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        self.detected_lines = []
        
        if lines is None:
            print("未检测到直线，尝试更低的阈值...")
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
        
        if lines is not None:
            print(f"原始检测到 {len(lines)} 条直线")
            
            # 合并相似直线
            merged_lines = self.merge_similar_lines(lines, rho_threshold=15, theta_threshold=0.15)
            print(f"合并后剩余 {len(merged_lines)} 条直线")
            
            # 过滤和分类直线
            horizontal_lines = []
            vertical_lines = []
            
            for line in merged_lines:
                rho, theta = line[0]
                angle_deg = np.degrees(theta)

                # 计算直线的起点和终点
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                # 存储直线信息
                self.detected_lines.append({
                    'rho': rho,
                    'theta': theta,
                    'angle_deg': angle_deg,
                    'pt1': pt1,
                    'pt2': pt2
                })
                
                # 修正的线条分类逻辑
                # 水平线: 角度接近0°或180°
                if (angle_deg < 15 or angle_deg > 165):
                    horizontal_lines.append((rho, theta))
                    print(f"水平线: rho={rho:.1f}, theta={theta:.3f}, angle={angle_deg:.1f}°")
                    cv2.line(self.image, pt1, pt2, (0, 255, 0), 2)  # 绿色水平线
                
                # 垂直线: 角度接近90°
                elif 75 < angle_deg < 105:
                    vertical_lines.append((rho, theta))
                    print(f"垂直线: rho={rho:.1f}, theta={theta:.3f}, angle={angle_deg:.1f}°")
                    cv2.line(self.image, pt1, pt2, (255, 0, 0), 2)  # 蓝色垂直线
                
                else:
                    # 其他角度的线用红色显示（通常是噪声）
                    cv2.line(self.image, pt1, pt2, (0, 0, 255), 1)
            
            print(f"分类结果 - 水平线: {len(horizontal_lines)}, 垂直线: {len(vertical_lines)}")
            
            # 如果检测效果不好，使用轮廓检测作为备选方案
            if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
                print("直线检测数量不足，尝试轮廓检测...")
                return self.detect_corners_by_contours()
            
            # 计算边界线 - 改进角点计算
            self.corners = self.calculate_corners_from_lines(horizontal_lines, vertical_lines)
        
        # 验证角点是否合理
        if len(self.corners) == 4:
            # 检查是否形成合理的矩形
            width = abs(self.corners[1][0] - self.corners[0][0])
            height = abs(self.corners[2][1] - self.corners[0][1])
            
            if width > 50 and height > 50:  # 最小尺寸检查
                print(f"检测到棋盘角点: {self.corners}")
                return self.corners
        
        # 如果霍夫变换失败，尝试轮廓检测
        print("霍夫变换检测失败，尝试轮廓检测...")
        return self.detect_corners_by_contours()
    
    def calculate_corners_from_lines(self, horizontal_lines, vertical_lines):
        """从水平线和垂直线计算角点"""
        if not horizontal_lines or not vertical_lines:
            return []
        
        # 找到最外围的线条
        h_rhos = [rho for rho, theta in horizontal_lines]
        v_rhos = [rho for rho, theta in vertical_lines]
        
        # 修复：处理负rho值的情况
        h_rhos = sorted(h_rhos)
        v_rhos = sorted(v_rhos)
        
        # 找到对应的线条
        top_line = None
        bottom_line = None
        left_line = None
        right_line = None
        
        for rho, theta in horizontal_lines:
            if abs(rho - h_rhos[0]) < 1e-6:
                top_line = (rho, theta)
            elif abs(rho - h_rhos[-1]) < 1e-6:
                bottom_line = (rho, theta)
        
        for rho, theta in vertical_lines:
            if abs(rho - v_rhos[0]) < 1e-6:
                left_line = (rho, theta)
            elif abs(rho - v_rhos[-1]) < 1e-6:
                right_line = (rho, theta)
        
        # 检查是否找到了所有边界线
        if not all([top_line, bottom_line, left_line, right_line]):
            print("无法找到所有边界线")
            return []
        
        # 计算交点
        corners = []
        line_pairs = [
            (top_line, left_line),     # 左上
            (top_line, right_line),    # 右上
            (bottom_line, left_line),  # 左下
            (bottom_line, right_line)  # 右下
        ]
        
        for line1, line2 in line_pairs:
            intersection = self.line_intersection(line1, line2)
            if intersection:
                corners.append(intersection)
        
        return corners
    
    def line_intersection(self, line1, line2):
        """计算两条直线的交点"""
        rho1, theta1 = line1
        rho2, theta2 = line2
        
        # 转换为直线方程 ax + by = c
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        c1, c2 = rho1, rho2
        
        # 解方程组
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:  # 平行线
            return None
        
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        
        return (int(x), int(y))
    
    def detect_corners_by_contours(self):
        """使用轮廓检测棋盘角点"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的矩形轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 轮廓近似
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) >= 4:
                # 取前4个点作为角点
                corners = approx[:4].reshape(-1, 2)
                
                # 排序角点：左上、右上、左下、右下
                corners = self.sort_corners(corners)
                self.corners = corners.tolist()
                
                print(f"通过轮廓检测到角点: {self.corners}")
                return self.corners
        
        # 最后的备选方案：使用图像边界
        print("使用图像边界作为默认角点...")
        h, w = self.image.shape[:2]
        margin = min(w, h) // 20  # 5%边距
        
        self.corners = [
            (margin, margin),           # 左上
            (w - margin, margin),       # 右上  
            (margin, h - margin),       # 左下
            (w - margin, h - margin)    # 右下
        ]
        
        print(f"使用默认角点: {self.corners}")
        return self.corners
    
    def sort_corners(self, corners):
        """排序角点为左上、右上、左下、右下的顺序"""
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 根据相对于中心的位置排序
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        # 按角度排序
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # 重新排列为 [左上, 右上, 左下, 右下]
        # 找到最左上的点作为起始点
        top_left_idx = np.argmin([p[0] + p[1] for p in sorted_corners])
        
        # 重新排列
        reordered = []
        reordered.append(sorted_corners[top_left_idx])  # 左上
        
        remaining = [p for i, p in enumerate(sorted_corners) if i != top_left_idx]
        remaining_x = [p[0] for p in remaining]
        remaining_y = [p[1] for p in remaining]
        
        # 找右上 (x大, y小)
        right_top_idx = np.argmax(remaining_x)
        reordered.append(remaining[right_top_idx])
        
        # 剩下的两个点
        final_remaining = [p for i, p in enumerate(remaining) if i != right_top_idx]
        
        # 按y坐标排序剩下的点
        final_remaining.sort(key=lambda p: p[1])
        reordered.extend(final_remaining)
        
        return np.array(reordered)
    
    def detect_grid_lines(self):
        """检测网格交叉点"""
        if len(self.corners) < 4:
            print("需要先检测棋盘角点")
            return []
        
        # 使用透视变换矫正棋盘
        corners_array = np.array(self.corners, dtype=np.float32)
        
        # 目标矩形 (正方形)
        board_size = 400  # 标准化尺寸
        dst_corners = np.array([
            [0, 0],
            [board_size, 0], 
            [0, board_size],
            [board_size, board_size]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(corners_array, dst_corners)
        
        # 生成网格点 (在标准化空间中)
        self.grid_points = []
        self.grid_points_normalized = []
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 标准化坐标
                x_norm = j * board_size / (self.board_size - 1) 
                y_norm = i * board_size / (self.board_size - 1)
                self.grid_points_normalized.append((x_norm, y_norm))
                
                # 反向变换到原图坐标
                point_norm = np.array([[x_norm, y_norm]], dtype=np.float32)
                point_orig = cv2.perspectiveTransform(point_norm.reshape(1, 1, 2), 
                                                    np.linalg.inv(M))
                
                x_orig = int(point_orig[0, 0, 0])
                y_orig = int(point_orig[0, 0, 1])
                self.grid_points.append((x_orig, y_orig))
        
        print(f"生成了 {len(self.grid_points)} 个网格交叉点")
        return self.grid_points
    
    def detect_stones(self):
        """改进的棋子检测"""
        if not self.grid_points:
            print("需要先检测网格点")
            return []
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 多尺度检测棋子
        self.stones = []
        
        # 尝试不同的参数组合
        param_sets = [
            {'dp': 1.2, 'minDist': 20, 'param1': 50, 'param2': 30, 'minR': 8, 'maxR': 25},
            {'dp': 1.5, 'minDist': 15, 'param1': 40, 'param2': 25, 'minR': 6, 'maxR': 30},
            {'dp': 1.0, 'minDist': 25, 'param1': 60, 'param2': 35, 'minR': 10, 'maxR': 20}
        ]
        
        all_circles = []
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=params['minDist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=params['minR'],
                maxRadius=params['maxR']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                all_circles.extend(circles)
        
        # 去重和筛选
        if all_circles:
            # 简单去重 (距离阈值)
            unique_circles = []
            for circle in all_circles:
                is_unique = True
                for existing in unique_circles:
                    dist = np.sqrt((circle[0] - existing[0])**2 + (circle[1] - existing[1])**2)
                    if dist < 15:  # 距离阈值
                        is_unique = False
                        break
                if is_unique:
                    unique_circles.append(circle)
            
            # 处理检测到的圆
            for (x, y, r) in unique_circles:
                # 找到最近的网格点
                min_dist = float('inf')
                closest_grid_idx = -1
                
                for idx, (gx, gy) in enumerate(self.grid_points):
                    dist = np.sqrt((x - gx)**2 + (y - gy)**2)
                    if dist < min_dist and dist < 30:  # 增加距离阈值
                        min_dist = dist
                        closest_grid_idx = idx
                
                if closest_grid_idx != -1:
                    # 改进的黑白判断
                    stone_color = self.determine_stone_color(x, y, r, gray)
                    
                    # 转换为棋盘坐标
                    grid_row = closest_grid_idx // self.board_size
                    grid_col = closest_grid_idx % self.board_size
                    
                    self.stones.append({
                        'position': (grid_row, grid_col),
                        'pixel_pos': (x, y),
                        'color': stone_color,
                        'radius': r
                    })
        
        print(f"检测到 {len(self.stones)} 个棋子")
        return self.stones
    
    def determine_stone_color(self, x, y, r, gray):
        """改进的棋子颜色判断"""
        # 创建多个同心圆进行采样
        total_intensity = 0
        sample_count = 0
        
        for radius_ratio in [0.6, 0.8]:  # 采样不同半径
            sample_r = int(r * radius_ratio)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), sample_r, 255, -1)
            
            # 排除边缘像素
            inner_mask = np.zeros(gray.shape, dtype=np.uint8) 
            cv2.circle(inner_mask, (x, y), max(1, sample_r-2), 255, -1)
            
            final_mask = cv2.bitwise_and(mask, inner_mask)
            intensity = cv2.mean(gray, mask=final_mask)[0]
            
            total_intensity += intensity
            sample_count += 1
        
        avg_intensity = total_intensity / sample_count if sample_count > 0 else 127
        
        # 动态阈值 (考虑背景亮度)
        background_intensity = cv2.mean(gray)[0]
        threshold = (background_intensity + 127) / 2
        
        return 'white' if avg_intensity > threshold else 'black'
    
    def visualize_results(self):
        """可视化检测结果"""
        result_image = self.original_image.copy()
        
        # 绘制角点
        for i, corner in enumerate(self.corners):
            cv2.circle(result_image, tuple(corner), 10, (0, 255, 0), -1)
            cv2.putText(result_image, f'C{i+1}', 
                       (corner[0]+15, corner[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制网格点 (每5个点绘制一个，避免过于密集)
        for i, point in enumerate(self.grid_points):
            if i % 5 == 0:  # 每5个点显示一个
                cv2.circle(result_image, point, 2, (255, 0, 0), -1)
        
        # 绘制棋子
        for stone in self.stones:
            x, y = stone['pixel_pos']
            color = (0, 0, 255) if stone['color'] == 'black' else (255, 255, 255)
            border_color = (255, 255, 255) if stone['color'] == 'black' else (0, 0, 0)
            
            cv2.circle(result_image, (x, y), stone['radius'], color, -1)
            cv2.circle(result_image, (x, y), stone['radius'], border_color, 2)
            
            # 添加位置标注
            pos_text = f"{stone['position']}"
            cv2.putText(result_image, pos_text, (x-15, y-stone['radius']-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 显示结果
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 2, 2) 
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Detection Results', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_board_state(self):
        """分析棋盘状态"""
        print("\n" + "="*50)
        print("           Board Analysis Results")
        print("="*50)
        print(f"Detected corners: {len(self.corners)}")
        print(f"Grid intersection points: {len(self.grid_points)}")
        print(f"Total stones detected: {len(self.stones)}")
        
        # 统计黑白子
        black_stones = [s for s in self.stones if s['color'] == 'black']
        white_stones = [s for s in self.stones if s['color'] == 'white']
        
        print(f"Black stones: {len(black_stones)}")
        print(f"White stones: {len(white_stones)}")
        
        if self.stones:
            print(f"\nStone positions:")
            print("-" * 30)
            for stone in sorted(self.stones, key=lambda x: (x['position'][0], x['position'][1])):
                row, col = stone['position']
                color_text = "●" if stone['color'] == 'black' else "○"
                print(f"{color_text} Position({row:2d},{col:2d}): {stone['color']:5s} stone")
        
        print("="*50)
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("Starting Go board analysis...")
        print("-" * 40)
        
        try:
            # 1. 上传并加载图像
            self.upload_and_load_image()
            
            # 2. 检测棋盘角点 (使用改进方法)
            self.detect_board_corners_improved()
            
            # 3. 检测网格线
            self.detect_grid_lines()
            
            # 4. 检测棋子
            self.detect_stones()
            
            # 5. 可视化结果
            self.visualize_results()
            
            # 6. 分析棋盘状态
            self.analyze_board_state()
            
            return {
                'corners': self.corners,
                'grid_points': self.grid_points,
                'stones': self.stones
            }
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

# 在local运行
if __name__ == "__main__":
    analyzer = ImprovedGoBoardAnalyzer()
    results = analyzer.run_full_analysis()