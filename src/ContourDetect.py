# 解决方案1: 改进轮廓筛选逻辑
# 这个版本的CONTOUR解决了小图丢失棋盘角的问题
def find_board_contour_improved(contours, img_shape):
    """
    改进的棋盘轮廓查找，避免选择整个图片边界
    """
    height, width = img_shape[:2]
    img_area = height * width
    
    candidates = []
    
    for item in contours:
        hull = cv2.convexHull(item)
        epsilon = 0.02 * cv2.arcLength(hull, True)  # 减小epsilon，更精确
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            ps = np.reshape(approx, (4,2))
            area = cv2.contourArea(approx)
            
            # 排除过大的轮廓（可能是整个图片边界）
            if area > img_area * 0.8:
                print(f"排除过大轮廓，面积: {area:.0f} (图片面积的 {area/img_area:.1%})")
                continue
            
            # 排除过小的轮廓
            if area < img_area * 0.05:
                print(f"排除过小轮廓，面积: {area:.0f}")
                continue
            
            # 检查四个角点是否太接近图片边界
            margin = min(width, height) * 0.02  # 2%的边界余量
            too_close_to_edge = False
            
            for point in ps:
                x, y = point
                if (x < margin or x > width - margin or 
                    y < margin or y > height - margin):
                    too_close_to_edge = True
                    break
            
            if too_close_to_edge:
                print(f"排除太接近边界的轮廓")
                continue
            
            # 检查四边形的长宽比是否合理（棋盘应该接近正方形）
            x_coords = ps[:, 0]
            y_coords = ps[:, 1]
            width_approx = max(x_coords) - min(x_coords)
            height_approx = max(y_coords) - min(y_coords)
            aspect_ratio = max(width_approx, height_approx) / min(width_approx, height_approx)
            
            if aspect_ratio > 2.0:  # 长宽比不能太极端
                print(f"排除长宽比过大的轮廓: {aspect_ratio:.2f}")
                continue
            
            candidates.append((approx, area, aspect_ratio))
    
    if not candidates:
        return None
    
    # 选择面积最大但长宽比合理的候选
    candidates.sort(key=lambda x: (-x[1], x[2]))  # 按面积降序，长宽比升序
    best_candidate = candidates[0]
    
    print(f"选择棋盘轮廓: 面积={best_candidate[1]:.0f}, 长宽比={best_candidate[2]:.2f}")
    
    # 重新整理角点顺序
    ps = np.reshape(best_candidate[0], (4,2))
    ps = ps[np.lexsort((ps[:,0],))]
    lt, lb = ps[:2][np.lexsort((ps[:2,1],))]
    rt, rb = ps[2:][np.lexsort((ps[2:,1],))]
    
    return (lt, lb, rt, rb)

# 解决方案2: 边缘图像预处理改进
def improve_edge_detection(img):
    """
    改进的边缘检测，减少图片边界干扰
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # 创建掩码，排除图片边界区域
    mask = np.ones_like(gray) * 255
    border_size = max(10, min(width, height) // 50)  # 动态边界大小
    mask[:border_size, :] = 0
    mask[-border_size:, :] = 0
    mask[:, :border_size] = 0
    mask[:, -border_size:] = 0
    
    # 高斯滤波
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 100)
    
    # 应用掩码，去除边界区域
    edges = cv2.bitwise_and(edges, mask)
    
    # 轻微dilate（比之前更保守）
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges, mask

# 解决方案3: 多候选验证
def validate_board_corners(corners, img):
    """
    验证检测到的棋盘角点是否合理
    """
    if corners is None:
        return False
    
    lt, lb, rt, rb = corners
    
    # 检查角点是否形成合理的四边形
    # 计算对角线长度
    diag1 = np.sqrt((rt[0] - lb[0])**2 + (rt[1] - lb[1])**2)
    diag2 = np.sqrt((lt[0] - rb[0])**2 + (lt[1] - rb[1])**2)
    
    # 对角线长度应该相近
    diag_ratio = max(diag1, diag2) / min(diag1, diag2)
    if diag_ratio > 1.5:
        print(f"对角线长度比例不合理: {diag_ratio:.2f}")
        return False
    
    # 检查四条边的长度
    side_lengths = [
        np.sqrt((rt[0] - lt[0])**2 + (rt[1] - lt[1])**2),  # 上边
        np.sqrt((rb[0] - rt[0])**2 + (rb[1] - rt[1])**2),  # 右边
        np.sqrt((lb[0] - rb[0])**2 + (lb[1] - rb[1])**2),  # 下边
        np.sqrt((lt[0] - lb[0])**2 + (lt[1] - lb[1])**2)   # 左边
    ]
    
    # 四条边长度应该相对均匀
    min_side = min(side_lengths)
    max_side = max(side_lengths)
    side_ratio = max_side / min_side
    
    if side_ratio > 1.8:
        print(f"边长比例不合理: {side_ratio:.2f}")
        return False
    
    print(f"角点验证通过: 对角线比={diag_ratio:.2f}, 边长比={side_ratio:.2f}")
    return True

# 使用改进的方法替换原来的透视变换部分
print("开始改进的透视变换...")

# 使用改进的边缘检测
im_edge_improved, mask = improve_edge_detection(img)
cv2.imshow('Improved Edge Detection', im_edge_improved)

# 查找轮廓
contours, hierarchy = cv2.findContours(im_edge_improved, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 使用改进的轮廓查找
rect = find_board_contour_improved(contours, img.shape)

# 验证角点
if rect is not None and not validate_board_corners(rect, img):
    print("角点验证失败，尝试其他候选...")
    rect = None

# 如果还是没找到，尝试降低标准
if rect is None:
    print("尝试降低检测标准...")
    # 可以在这里添加备用检测方法
    # 比如使用原始的edge图像，或者调整参数