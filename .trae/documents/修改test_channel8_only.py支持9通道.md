# 修改test_channel8_only.py支持9通道训练和预测

## 目标
将 `test_channel8_only.py` 从单通道改为支持9个通道，参考 `ultrasonic_hdc_emg_style.py` 的实现。

## 主要修改内容

### 1. 参数配置
- 添加 `CHANNELS = 9` 参数
- 修改窗口大小、滑动步长、N_GRAM_SIZE为最佳值
- `DEFAULT_WINDOW = 256`
- `DEFAULT_STEP = 8`
- `DEFAULT_N_GRAM_SIZE = 31`

### 2. 数据集类修改
参考 `ultrasonic_hdc_emg_style.py` 的 `UltrasonicDataset`：
- 从单通道改为多通道
- 支持多通道标签（label_0到label_8）
- 窗口标签使用投票机制（窗口内出现最多的标签）

### 3. 编码器类修改
参考 `ultrasonic_hdc_emg_style.py` 的 `Encoder`：
- 从单通道改为多通道
- `self.channels = embeddings.Random(CHANNELS, out_features)`
- 支持多通道输入

### 4. 训练函数修改
参考 `ultrasonic_hdc_emg_style.py` 的 `train_model`：
- 检测多通道标签
- 为每个通道创建独立的Centroid模型
- 分别训练每个通道的模型

### 5. 预测函数修改
参考 `ultrasonic_hdc_emg_style.py` 的 `predict_model`：
- 加载多通道模型
- 为每个通道进行预测
- 保存多通道预测结果（predicted_label_0到predicted_label_8）

### 6. 准确率计算修改
- 支持多通道准确率计算
- 分别计算每个通道的精确率、召回率、F1分数

## 修改后的文件结构
```
test_channel8_only.py (修改为test_all_channels.py)
├── 参数配置（支持9通道）
├── Channel8Dataset → AllChannelsDataset（多通道数据集）
├── Channel8Encoder → AllChannelsEncoder（多通道编码器）
├── train_channel8_model → train_all_channels_model（多通道训练）
├── predict_channel8_model → predict_all_channels_model（多通道预测）
└── calculate_channel8_accuracy → calculate_all_channels_accuracy（多通道准确率）
```

## 关键改进点
1. **多通道支持**：从单通道扩展到9通道
2. **独立模型**：每个通道一个独立的Centroid模型
3. **投票机制**：窗口标签使用窗口内出现最多的标签
4. **最佳参数**：使用测试得到的最佳配置（window=256, step=8, n_gram=31）

## 文件清单
- 修改 `test_channel8_only.py` 为 `test_all_channels.py`（或直接修改原文件）