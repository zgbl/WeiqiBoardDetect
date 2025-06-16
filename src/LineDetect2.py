import cv2
import numpy as np

# 1. 读图 & 转灰度
img = cv2.imread("../data/raw/BasicStraightLine.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#显示原图
cv2.imshow("Original Pic", img) 

# 2. 边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 3. 使用 HoughLinesP 检测“线段”（不是无限延长线）
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=50, 
                        minLineLength=50, 
                        maxLineGap=5)

# 4. 画线（设置为和原图一致的线宽：1）
lineCount = 0
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        if lineCount < 1:
            print("Line is:", x1, y1, "To:", x2, y2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1 )  # 线宽1，颜色红
            lineCount += 1
# 5. 显示
cv2.imshow("Detected Line", img)
cv2.waitKey(0)
cv2.destroyAllWindows()