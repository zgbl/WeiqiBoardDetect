import cv2
import numpy as np

def is_similar(line1, line2, dist_thresh=3, angle_thresh=5):
    x1, y1, x2, y2 = line1
    a1 = np.arctan2(y2 - y1, x2 - x1)
    x3, y3, x4, y4 = line2
    a2 = np.arctan2(y4 - y3, x4 - x3)
    angle_diff = abs(a1 - a2) * 180 / np.pi
    angle_diff = min(angle_diff, 180 - angle_diff)

    # 判断角度是否接近
    if angle_diff > angle_thresh:
        return False

    # 判断起点和终点是否接近（任选一个点即可）
    dist = np.hypot(x1 - x3, y1 - y3)
    return dist < dist_thresh

# 1. 读图 & 转灰度
#img = cv2.imread("../data/raw/BasicStraightLine2.jpg")
img = cv2.imread("../data/raw/Pic1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#先做高斯模糊 + 提升对比度
blur = cv2.GaussianBlur(gray, (3, 3), 0)
equalized = cv2.equalizeHist(blur)

#显示原图
cv2.imshow("Original Pic", img) 
cv2.moveWindow("Original Pic", 0, 0)

# 2. 边缘检测
edges = cv2.Canny(gray, 40, 160, apertureSize=3)
#edges = cv2.Canny(equalized, 50, 150)   # 提升了对比度后的Canny

print("Equalized shape:", equalized.shape, "dtype:", equalized.dtype)
print("Equalized min/max:", np.min(equalized), np.max(equalized))
cv2.imshow("Equalizeed", equalized)
cv2.moveWindow("Equalizeed", 600, 0)

# 3. 使用 HoughLinesP 检测“线段”（不是无限延长线）
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=0, 
                        minLineLength=20, 
                        maxLineGap=5)


# 2. 去重处理
filtered_lines = []


# 4. 画线（设置为和原图一致的线宽：1）
lineCount = 0
if lines is not None:

    for candidate in lines[:, 0]:
        if all(not is_similar(candidate, kept) for kept in filtered_lines):
            filtered_lines.append(candidate)

    #x1, y1, x2, y2 = lines[0][0]
    #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in lines[:, 0]:
       
        #print("Line is:", x1, y1, "To:", x2, y2)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1 )  # 线宽1，颜色红
        lineCount += 1
    #print("Total line is:", lineCount)
# 5. 显示
cv2.imshow("Detected Line", img)
cv2.moveWindow("Detected Line", 0, 300)
cv2.waitKey(0)
cv2.destroyAllWindows()