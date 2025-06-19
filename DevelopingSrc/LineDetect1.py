import cv2
import numpy as np

# 1. 读取图像
img = cv2.imread('../data/raw/BasicStraightLine.jpg')  # 替换为你的图片路径
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#显示原图
cv2.imshow("Original Pic", img) 

# 2. 边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 3. 霍夫直线变换
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# 4. 绘制检测到的线条
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # x1,y1 和 x2,y2 是一条线上两点
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 5. 显示结果

cv2.imshow("Detected Line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()