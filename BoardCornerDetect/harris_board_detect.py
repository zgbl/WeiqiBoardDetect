import cv2
import numpy as np
import os

def detect_harris_corners(image_path, output_path=None):
    """
    使用 Harris Corner Detection 检测棋盘角点
    Harris Corner Detection detects corners by looking for regions with high intensity changes in all directions.
    """
    
    # 1. 读取图像
    if not os.path.exists(image_path):
        print(f"Error: 找不到文件 {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 无法加载图像 {image_path}")
        return

    print(f"Processing image: {image_path}")
    
    # 复制一份原图用于显示结果
    result_img = img.copy()

    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)


    # 3. 转换为 float32 (Harris 算法要求输入为 float32)
    gray = np.float32(gray)

    # 4. Harris Corner Detection
    # blockSize: 角点检测中考虑的邻域大小
    # ksize: Sobel 算子的中孔大小 (用于计算梯度)
    # k: Harris 检测器方程中的自由参数 [0.04, 0.06]
    print("Applying Harris Corner Detection...")
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)

    # 5. 结果膨胀（以便更好地标记角点）
    dst = cv2.dilate(dst, None)

    # 6. 阈值化：标记检测到的角点
    # 我们将使用细化后的坐标来画更大的实心圆，方便用户观察位置
    threshold = 0.01 * dst.max()

    # 7. 可选：细化角点位置 (Sub-pixel accuracy)
    # Harris 检测出的角点通常是一个区域，我们可以用 cornerSubPix 进一步精确
    print("Refining corner locations...")
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(dst > threshold))
    
    # 定义停止准则
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    # gray 类型需要是 uint8 用于 cornerSubPix
    gray_uint8 = np.uint8(gray)
    corners = cv2.cornerSubPix(gray_uint8, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # 在图像上绘制红色的圆点表示角点位置 (画得稍大一些)
    print(f"Drawing {len(corners)-1} corners...")
    for i in range(1, len(corners)):
        # 圆心，半径=8，红色，厚度=2
        cv2.circle(result_img, (int(corners[i][0]), int(corners[i][1])), 8, (0, 0, 255), 2)
        # 内部再打个小核心点
        cv2.circle(result_img, (int(corners[i][0]), int(corners[i][1])), 2, (0, 0, 255), -1)

    # 8. 保存或显示结果
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Saved result to: {output_path}")

    # UI 显示 (注意: 在某些环境下可能无法打开窗口)
    try:
        cv2.imshow('Harris Corners (Red: Raw, Green: Refined)', result_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Warning: Could not show window: {e}")

if __name__ == "__main__":
    # 默认路径，你可以修改为你自己的图片路径
    input_img = "/Users/tuxy/Codes/AI/OpenCVTest1/data/raw/Board1.jpg"
    output_img = "/Users/tuxy/Codes/AI/WeiqiBoardDetect/BoardCornerDetect/harris_result.jpg"
    
    detect_harris_corners(input_img, output_img)
