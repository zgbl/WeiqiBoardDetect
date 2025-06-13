# Step 0: 安装 & 导入依赖
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os
from PIL import Image

# Step 1: 上传围棋图片
uploaded = files.upload()  # 上传清晰围棋图像，如 PNG/JPG

# Step 2: 加载图像
img_path = list(uploaded.keys())[0]
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 3: 显示原图
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.title("原始围棋图片")
plt.axis('off')
plt.show()

# Step 4: 假设整张图已经是棋盘（如果没有歪斜的话）
# 若不是完整棋盘图，可以先做 perspective transform，后续补上
height, width, _ = img.shape
grid_size = 19
cell_h = height // grid_size
cell_w = width // grid_size

# Step 5: 遍历每个交叉点，裁剪小格并保存
output_dir = "cropped_cells"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 12))
counter = 0

for i in range(grid_size):
    for j in range(grid_size):
        y = i * cell_h
        x = j * cell_w
        cell = img_rgb[y:y+cell_h, x:x+cell_w]

        # 保存为单独图像
        cell_img = Image.fromarray(cell)
        cell_img.save(f"{output_dir}/cell_{i:02d}_{j:02d}.png")

        # 可视化其中几个格子
        if counter < 25:
            plt.subplot(5, 5, counter + 1)
            plt.imshow(cell)
            plt.axis('off')
        counter += 1

plt.suptitle("前 25 个切割格子预览")
plt.show()