# Checkpoint v9: Dataset2 Discovery & Auto-Cleaning Logic
================================================================

### 🏁 训练重点 (Focus)
*   **黑盒验证**：将 v8 在 Dataset1 上学到的高性能模型，作为“检察官”去扫描未清理的 Dataset2。
*   **异常检测**：首次引入“自动嫌疑名单 (Suspects)”机制。
*   **工业化提速**：全面开启 AMP (自动混合精度) 训练。

### 🛠 主要修改 (Key Changes)
*   **数据集切换**：配置修改为 **Dataset2 ONLY**。
*   **Suspect Detection**：在 `evaluate` 环节，自动检测 `Conf > 0.9` 且 `Pred != Label` 的样本并记录到 CSV。
*   **Heuristic Filter**：增加了白棋区域亮度启发式过滤逻辑 (W-Area brightness filter)。

### 📊 训练结果评价 (Results)
*   **验证集准确率**：**~63%**。
*   **关键发现**：
    1. 生成了包含 **14,483** 个样本的嫌疑名单 (`suspects_latest.csv`)。
    2. 发现 Dataset2 中 White 类别极度稀缺且伴随大量噪声（光点/假目标）。
    3. 确认了“脏数据比模型架构更限制性能”。

### ⚠️ 下一步行动 (Action)
*   由本版本衍生出了 `auto_clean_dataset.py`。
*   计划：利用模型置信度 + 物理亮度阈值，对 Dataset2 进行地毯式自动清洗。
