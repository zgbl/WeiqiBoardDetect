import cv2
import numpy as np

# 读取图像
#image_path = "../data/raw/Toushi1-pre.png"  # 原始图（拍摄的）
image_path = "../data/raw/Homeboard4.jpg"
img = cv2.imread(image_path)
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 手动设定棋盘四个角点（左上、左下、右上、右下）——注意顺序
# 你可以用 matplotlib 手动点选，也可以直接用图像查看器估算
lt = [42, 46]    # 左上角
lb = [39, 678]   # 左下角
rt = [702, 40]   # 右上角
rb = [707, 677]  # 右下角

# 构造透视变换矩阵
pts2 = np.float32([lt, lb, rt, rb])
pts1 = np.float32([[0, 0], [0, 660], [660, 0], [660, 660]])
M = cv2.getPerspectiveTransform(pts2, pts1)

# 应用透视变换
board_gray = cv2.warpPerspective(im_gray, M, (660, 660))
board_bgr = cv2.warpPerspective(img, M, (660, 660))

# 显示结果
cv2.imshow("Original", img)
cv2.imshow("Warped Gray", board_gray)
cv2.imshow("Warped BGR", board_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
>>> lt, lb, rt, rb = rect
>>> pts1 = np.float32([(10,10), (10,650), (650,10), (650,650)]) # 预期的棋盘四个角的坐标
>>> pts2 = np.float32([lt, lb, rt, rb]) # 当前找到的棋盘四个角的坐标
>>> m = cv2.getPerspectiveTransform(pts2, pts1) # 生成透视矩阵
>>> board_gray = cv2.warpPerspective(im_gray, m, (660, 660)) # 对灰度图执行透视变换
>>> board_bgr = cv2.warpPerspective(im_bgr, m, (660, 660)) # 对彩色图执行透视变换
>>> cv2.imshow('go', board_gray)
"""