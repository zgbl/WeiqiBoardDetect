# 灵活的棋子检测方法 - 不依赖严格的圆形检测
import cv2
import numpy as np
from scipy import ndimage

def detect_pieces_by_regions(corrected_board, grid_points, grid_spacing_x, grid_spacing_y):
    """
    基于区域的棋子检测，不要求严格的圆形
    """
    print("开始基于区域的棋子检测...")
    
    # 转换为灰度图
    gray = cv2.cvtColor(corrected_board, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # 估算棋子大小
    expected_piece_size = min(grid_spacing_x, grid_spacing_y) * 0.8
    min_area = int((expected_piece_size * 0.6) ** 2 * np.pi / 4)  # 最小面积
    max_area = int((expected_piece_size * 1.4) ** 2 * np.pi / 4)  # 最大面积
    
    print(f"预期棋子大小: {expected_piece_size:.1f}, 面积范围: {min_area}-{max_area}")
    
    detected_pieces = []
    
    # 方法1: 基于阈值的区域检测
    pieces_from_threshold = detect_by_threshold(gray, grid_points, grid_spacing_x, grid_spacing_y, min_area, max_area)
    detected_pieces.extend(pieces_from_threshold)
    print(f"阈值方法检测到: {len(pieces_from_threshold)} 个棋子")
    
    # 方法2: 基于轮廓的检测
    pieces_from_contours = detect_by_contours(gray, grid_points, grid_spacing_x, grid_spacing_y, min_area, max_area)
    detected_pieces.extend(pieces_from_contours)
    print(f"轮廓方法检测到: {len(pieces_from_contours)} 个棋子")
    
    # 方法3: 基于模板匹配的检测
    pieces_from_template = detect_by_template_matching(gray, grid_points, grid_spacing_x, grid_spacing_y)
    detected_pieces.extend(pieces_from_template)
    print(f"模板匹配检测到: {len(pieces_from_template)} 个棋子")
    
    # 去重和合并结果
    final_pieces = merge_detections(detected_pieces, grid_spacing_x * 0.5)
    print(f"去重后最终检测到: {len(final_pieces)} 个棋子")
    
    return final_pieces

def detect_by_threshold(gray, grid_points, grid_spacing_x, grid_spacing_y, min_area, max_area):
    """基于双阈值的区域检测"""
    pieces = []
    
    # 检测黑色区域（低阈值）
    _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    black_pieces = find_regions_near_grid(black_mask, grid_points, grid_spacing_x, grid_spacing_y, 
                                         min_area, max_area, 'black')
    pieces.extend(black_pieces)
    
    # 检测白色区域（高阈值）
    _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    white_pieces = find_regions_near_grid(white_mask, grid_points, grid_spacing_x, grid_spacing_y,
                                         min_area, max_area, 'white')
    pieces.extend(white_pieces)
    
    return pieces

def detect_by_contours(gray, grid_points, grid_spacing_x, grid_spacing_y, min_area, max_area):
    """基于轮廓的检测"""
    pieces = []
    
    # 边缘检测
    edges = cv2.Canny(gray, 30, 100)
    
    # 形态学操作，连接断开的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # 计算轮廓中心
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 检查是否在网格点附近
                #distances = [np.sqrt((cx - gx)**2 + (cy - gy)**2) for gx, gy in grid_points]
                distances = [np.sqrt((cx - p[0])**2 + (cy - p[1])**2) for p in grid_points]
                min_dist = min(distances)
                tolerance = max(grid_spacing_x, grid_spacing_y) * 0.6
                
                if min_dist <= tolerance:
                    # 判断颜色
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    mean_intensity = cv2.mean(gray, mask=mask)[0]
                    
                    if mean_intensity < 100:
                        color = 'black'
                    elif mean_intensity > 150:
                        color = 'white'
                    else:
                        continue  # 跳过不确定的
                    
                    closest_idx = distances.index(min_dist)
                    pieces.append({
                        'center': (cx, cy),
                        'color': color,
                        'grid_point': grid_points[closest_idx],
                        'grid_index': closest_idx,
                        'method': 'contour',
                        'intensity': mean_intensity,
                        'area': area
                    })
    
    return pieces

def detect_by_template_matching(gray, grid_points, grid_spacing_x, grid_spacing_y):
    """基于每个网格点周围区域的分析"""
    pieces = []
    
    # 在每个网格点周围检查是否有棋子
    search_radius = int(min(grid_spacing_x, grid_spacing_y) * 0.6)
    
    #for i, (gx, gy) in enumerate(grid_points):
    for i, p in enumerate(grid_points):   #7/20
        gx, gy = p[0], p[1]   # 7/20
        # 提取网格点周围的区域
        x1 = max(0, gx - search_radius)
        y1 = max(0, gy - search_radius) 
        x2 = min(gray.shape[1], gx + search_radius)
        y2 = min(gray.shape[0], gy + search_radius)
        
        if x2 - x1 < 10 or y2 - y1 < 10:  # 区域太小
            continue
            
        roi = gray[y1:y2, x1:x2]
        
        # 分析ROI的统计特性
        mean_intensity = np.mean(roi)
        std_intensity = np.std(roi)
        
        # 如果标准差很小且强度偏向黑色或白色，可能是棋子
        #if std_intensity < 30:  # 相对均匀的区域
        if std_intensity < 35:  # 稍微放宽 7/20
            #if mean_intensity < 80:  # 黑色棋子
            if mean_intensity < 85:  # 黑色棋子，稍微提高阈值，因为周围有白子可能拉高平均值 7/20
                # 进一步验证：寻找最暗的连通区域
                _, binary = cv2.threshold(roi, mean_intensity + 10, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 找最大的连通区域
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    min_expected_area = (search_radius * 0.5) ** 2 * np.pi
                    
                    if area > min_expected_area:
                        # 计算这个区域的中心（相对于ROI）
                        M = cv2.moments(largest_contour)
                        if M["m00"] > 0:
                            local_cx = int(M["m10"] / M["m00"])
                            local_cy = int(M["m01"] / M["m00"])
                            # 转换为全图坐标
                            global_cx = x1 + local_cx
                            global_cy = y1 + local_cy
                            
                            pieces.append({
                                'center': (global_cx, global_cy),
                                'color': 'black',
                                'grid_point': (gx, gy),
                                'grid_index': i,
                                'method': 'template',
                                'intensity': mean_intensity,
                                'area': area
                            })
            
            elif mean_intensity > 180:  # 白色棋子
                _, binary = cv2.threshold(roi, mean_intensity - 20, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    min_expected_area = (search_radius * 0.5) ** 2 * np.pi
                    
                    if area > min_expected_area:
                        M = cv2.moments(largest_contour)
                        if M["m00"] > 0:
                            local_cx = int(M["m10"] / M["m00"])
                            local_cy = int(M["m01"] / M["m00"])
                            global_cx = x1 + local_cx
                            global_cy = y1 + local_cy
                            
                            pieces.append({
                                'center': (global_cx, global_cy),
                                'color': 'white',
                                'grid_point': (gx, gy),
                                'grid_index': i,
                                'method': 'template',
                                'intensity': mean_intensity,
                                'area': area
                            })
    
    return pieces

def find_regions_near_grid(binary_mask, grid_points, grid_spacing_x, grid_spacing_y, min_area, max_area, color):
    """在二值图像中寻找网格点附近的区域"""
    pieces = []
    
    # 形态学操作，清理噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    for i in range(1, num_labels):  # 跳过背景(0)
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            cx, cy = centroids[i]
            cx, cy = int(cx), int(cy)
            
            # 检查是否在网格点附近
            #distances = [np.sqrt((cx - gx)**2 + (cy - gy)**2) for gx, gy in grid_points]
            distances = [np.sqrt((cx - p[0])**2 + (cy - p[1])**2) for p in grid_points]
            min_dist = min(distances)
            tolerance = max(grid_spacing_x, grid_spacing_y) * 0.6
            
            if min_dist <= tolerance:
                closest_idx = distances.index(min_dist)
                pieces.append({
                    'center': (cx, cy),
                    'color': color,
                    'grid_point': grid_points[closest_idx],
                    'grid_index': closest_idx,
                    'method': 'threshold',
                    'intensity': 50 if color == 'black' else 200,  # 估算值
                    'area': area
                })
    
    return pieces

def merge_detections(detected_pieces, merge_distance):
    """合并重复检测的结果"""
    if not detected_pieces:
        return []
    
    # 按网格索引分组
    grid_groups = {}
    for piece in detected_pieces:
        grid_idx = piece['grid_index']
        if grid_idx not in grid_groups:
            grid_groups[grid_idx] = []
        grid_groups[grid_idx].append(piece)
    
    merged_pieces = []
    for grid_idx, pieces in grid_groups.items():
        if len(pieces) == 1:
            merged_pieces.append(pieces[0])
        else:
            # 多个检测结果，选择最可信的
            # 优先级：template > contour > threshold
            method_priority = {'template': 3, 'contour': 2, 'threshold': 1}
            best_piece = max(pieces, key=lambda p: method_priority.get(p['method'], 0))
            merged_pieces.append(best_piece)
    
    return merged_pieces

def visualize_detections(corrected_board, detected_pieces, output_path='debug_region_detection.jpg'):
    """可视化检测结果"""
    debug_img = corrected_board.copy()
    
    for i, piece in enumerate(detected_pieces):
        cx, cy = piece['center']
        color = piece['color']
        method = piece['method']
        
        # 根据颜色选择标记颜色
        mark_color = (0, 0, 255) if color == 'black' else (255, 255, 255)
        
        # 画圆标记
        cv2.circle(debug_img, (cx, cy), 15, mark_color, 2)
        
        # 添加文字标签
        label = f"{i}:{color[0]}{method[0]}"
        cv2.putText(debug_img, label, (cx-20, cy-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, debug_img)
    print(f"检测结果可视化保存到: {output_path}")

# 使用示例：
def replace_circle_detection_in_your_code():
    """在你的主程序中替换圆检测部分"""
    
    # 原来的代码:
    # circles = cv2.HoughCircles(...)
    
    # 替换为:
    detected_pieces = detect_pieces_by_regions(corrected_board, grid_points, grid_spacing_x, grid_spacing_y)
    
    # 可视化结果
    visualize_detections(corrected_board, detected_pieces)
    
    # 转换为你原来的格式
    pieces_for_grid = []
    for piece in detected_pieces:
        cx, cy = piece['center']
        color = piece['color']
        grid_point = piece['grid_point']
        grid_index = piece['grid_index']
        intensity = piece['intensity']
        
        pieces_for_grid.append((cx, cy, intensity, color))
    
    print(f"最终检测到棋子: {len(pieces_for_grid)} 个")
    for i, (x, y, intensity, color) in enumerate(pieces_for_grid):
        print(f"圆心({x},{y}) 强度:{intensity:.1f} 判断为:{color}")
    
    return pieces_for_grid