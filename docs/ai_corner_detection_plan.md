# 基于 Gomrade 数据集的深度学习棋盘角检测方案
(Gomrade Dataset Utilization for AI-based Corner Detection)

## 1. 核心发现：数据集中的“宝藏”
在 `/Users/tuxy/Codes/AI/Data/Gomrade-dataset1/` 下的各个子目录中，除了原始图片（`.png`），最重要的辅助文件是：
*   **`board_extractor_state.yml`**: 这个文件保存了人工标注的 **棋盘四个角点坐标 (`pts_clicks`)**。
    *   例如，在文件夹 `6` 中，坐标点为：`(569, 72), (1737, 53), (1790, 1339), (553, 1331)`。
*   **`board_state_classifier_state.yml`**: 保存了棋盘的颜色特征和网格分步。
*   **`.txt` 文件**: 保存了 19x19 棋盘的真实棋子分布（Ground Truth Board State）。

**利用价值**：这些数据完美解决了深度学习中最难的“标签（Label）”问题。我们可以利用这些成千上万张已标注好角点的图片，训练一个**关键点检测网络（Keypoint Detector）**，从而彻底解决 OpenCV 传统方法容易被手臂、光影干扰的问题。

---

## 2. AI 训练方案：棋盘角点回归

### A. 数据准备 (Data Preparation)
1.  **遍历解析**：编写 Python 脚本遍历所有 39 个子文件夹。
2.  **标签提取**：从每个文件夹的 `board_extractor_state.yml` 中提取 `pts_clicks` 作为 Ground Truth (Y)。
3.  **多图关联**：
    *   有些文件夹（如 `20_03_27_19_18_09`）包含几十张图片（`0.png`, `1.png`...），但只有一个 `board_extractor_state.yml`。这意味着在这些序列中，相机是固定的，所有图片共享同一组角点。
    *   这大大增加了训练样本量。

### B. 模型选择 (Model Selection)
推荐两种架构：
1.  **坐标回归模型 (Coordinate Regression)**:
    *   **Backbone**: ResNet-18 或 MobileNetV3 (轻量且快)。
    *   **Head**: 全连接层直接输出 8 个数值（代表 4 个 (x, y) 坐标）。
    *   **优点**：结构简单，推理极快。
2.  **热力图检测模型 (Heatmap-based Detection)**:
    *   **架构**：类似 YOLOv8-pose 或 HRNet。
    *   **输出**：为 4 个角点各生成一张概率热力图。
    *   **优点**：精度极高，即使角点被轻微遮挡也能根据上下文“预测”位置。

### C. 关键增强技术 (Augmentation)
模型能否在你的真实相机环境下运行，取决于**数据增强**：
*   **色彩增强**：模拟不同光照、阴影、白平衡。
*   **几何变换**：随机旋转、适度剪裁，确保模型不依赖棋盘在相机中的绝对位置。
*   **合成遮挡**：在图片上随机叠加“黑色块”模拟手臂遮挡，强迫模型根据剩余线条推算角点。

---

## 3. 工程实施建议 (Action Plan)

### 第一阶段：验证数据集 (Validation)
先写一个可视化脚本，把 `pts_clicks` 画在对应的 `.png` 上，确认我们理解的坐标系与图片一致。

### 第二阶段：训练 Pipeline
1.  将数据转换为标准格式（如 COCO-Keypoints 或简单格式）。
2.  使用迁移学习（Transfer Learning），加载已经在 ImageNet 预训练过的权重。
3.  **损失函数**：使用 MSE Loss (均方误差) 或 SmoothL1 Loss。

### 第三阶段：端到端识别流程 (Inference Pipeline)
1.  **AI 粗定位**：AI 模型预测 4 个角点的大致坐标 $(\hat{x}, \hat{y})$。
2.  **几何精校准 (可选)**：以 AI 预测的点为中心开一个小窗口，在窗口内用 OpenCV 寻找真实的交点线段（如你之前的 LSD/Hough 逻辑），进一步提升到亚像素级精度。
3.  **透视变换**：使用 `cv2.getPerspectiveTransform` 将棋盘拉伸为标准方形。
4.  **棋子识别**：利用数据集中的 `.txt` 标注，训练一个 3 分类器（黑子、白子、空格）。

---

## 4. 结论：为什么这能解决你现在的问题？

你目前在 `BoardCornerDetect` 遇到的瓶颈是“逻辑复杂且容易断”。
*   OpenCV 逻辑是 **Top-Down**：试图从碎线拼接全局。
*   AI 逻辑是 **Bottom-Up**：它看过几千张棋盘，知道“长成这样的东西通常四个角在哪里”，它对噪声和遮挡的容忍度远高于基于规则的 OpenCV。

**下一步行动建议**：
我可以为你写一个 **Gomrade 数据解析与可视化脚本**，先帮你确认这些数据是否好用。你意下如何？
