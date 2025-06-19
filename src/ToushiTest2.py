# 这个版本在透视变换上，如果原图符合一定要求，无遮挡，图片大小在一定范围，棋盘边缘清楚，则表现很好。
import cv2
import numpy as np

# 读取图像
#image_path = "../data/raw/Toushi1-pre.png"  # 原始图（拍摄的）
#image_path = "../data/raw/OGS4.jpg"  # 原始图（拍摄的）
#image_path = "../data/raw/IMG20171015161921.jpg"
image_path = "../data/raw/IMG20160706171004.jpg"
img = cv2.imread(image_path)

if img is None:
    print("没找到照片", image_path)
print(image_path)
cv2.imshow("orginal picture", img)

im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


im_gray = cv2.GaussianBlur(im_gray, (3,3), 0) # 滤波降噪

im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测

cv2.imshow('Go', im_edge) # 显示边缘检测结果

contours, hierarchy = cv2.findContours(im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 提取轮廓
rect, area = None, 0 # 找到的最大四边形及其面积
for item in contours:
	hull = cv2.convexHull(item) # 寻找凸包
	epsilon = 0.1 * cv2.arcLength(hull, True) # 忽略弧长10%的点
	approx = cv2.approxPolyDP(hull, epsilon, True) # 将凸包拟合为多边形
	if len(approx) == 4 and cv2.isContourConvex(approx): # 如果是凸四边形
		ps = np.reshape(approx, (4,2))
		ps = ps[np.lexsort((ps[:,0],))]
		lt, lb = ps[:2][np.lexsort((ps[:2,1],))]
		rt, rb = ps[2:][np.lexsort((ps[2:,1],))]
		a = cv2.contourArea(approx) # 计算四边形面积
		if a > area:
			area = a
			rect = (lt, lb, rt, rb)

if rect is None:
	print('在图像文件中找不到棋盘！')
else:
	print('棋盘坐标：')
	print('\t左上角：(%d,%d)'%(rect[0][0],rect[0][1]))
	print('\t左下角：(%d,%d)'%(rect[1][0],rect[1][1]))
	print('\t右上角：(%d,%d)'%(rect[2][0],rect[2][1]))
	print('\t右下角：(%d,%d)'%(rect[3][0],rect[3][1]))

	
	"""
	for i, (corner, color, label) in enumerate(zip(final_corners, colors, labels)):
	x, y = int(corner[0]), int(corner[1])
	cv2.circle(result_img, (x, y), 15, color, -1)
	cv2.putText(result_img, label, (x+20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
	"""

"""棋盘坐标：
	左上角：(111,216)
	左下角：(47,859)
	右上角：(753,204)
	右下角：(823,859)
"""

im = np.copy(img)
for p in rect:
	#im = cv2.line(im, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,0,255), 1)
	#im = cv2.line(im, (p[0],p[1]-10), (p[0],p[1]+10), (0,0,255), 1)

	cv2.circle(im,(p[0],p[1]),15,(0,255,0),-1)
	
cv2.imshow('go', im)
cv2.imshow('With Cornor', im)


lt, lb, rt, rb = rect
pts1 = np.float32([(10,10), (10,650), (650,10), (650,650)]) # 预期的棋盘四个角的坐标
pts2 = np.float32([lt, lb, rt, rb]) # 当前找到的棋盘四个角的坐标
m = cv2.getPerspectiveTransform(pts2, pts1) # 生成透视矩阵
board_gray = cv2.warpPerspective(im_gray, m, (660, 660)) # 对灰度图执行透视变换
board_bgr = cv2.warpPerspective(img, m, (660, 660)) # 对彩色图执行透视变换
cv2.imshow('go', board_gray)

"""经过测试，用他的图片，还真的是可以透视变换。""" 




cv2.waitKey(0)
cv2.destroyAllWindows()