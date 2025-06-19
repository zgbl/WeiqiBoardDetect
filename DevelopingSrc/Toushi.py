import cv2
import numpy as np

def detect_and_transform_chessboard(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像文件")
        return
    
    # 保存原始图像尺寸
    original_img = img.copy()
    
    # 图像预处理
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 使用双边滤波减少噪声
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    found = False
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4 and cv2.contourArea(approx) > 1000:
                cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)
                corners_pts = approx.reshape(4, 2).astype(np.float32)
                corners_pts = order_points(corners_pts)
                found = True
                break
    
    if found and corners_pts is not None:
        target_size = (400, 400)
        target_pts = np.array([
            [0, 0],
            [target_size[0], 0],
            [target_size[0], target_size[1]],
            [0, target_size[1]]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(corners_pts, target_pts)
        
        warped = cv2.warpPerspective(img, matrix, target_size)
        
        cv2.namedWindow('yuantu', cv2.WINDOW_NORMAL)
        cv2.imshow('yuantu', img)
        cv2.namedWindow('zhentu', cv2.WINDOW_NORMAL)
        cv2.imshow('zhentu', warped)
        
        cv2.imwrite('detected_chessboard.png', img)
        cv2.imwrite('transformed_chessboard.png', warped)
    else:
        print("无法进行透视变换：未检测到有效的棋盘角点")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    return rect

if __name__ == "__main__":
    image_path = "1.jpg"
    detect_and_transform_chessboard(image_path)
