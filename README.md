# 超声波探伤HDC系统

基于超维计算(HDC)的钢轨超声波探伤智能识别系统。

## 系统特点

- **点级分类**: 基于点级分类策略，每个滑动窗口的中心点作为分类对象
- **多通道处理**: 支持9个超声波通道的独立编码和分类
- **增量学习**: 支持增量学习，仅训练新添加的数据
- **BeamOffset调整**: 考虑BeamOffset调整，确保各通道数据正确对齐
- **丰富评估**: 提供精确率、召回率和F1分数等评估指标
- **可视化支持**: 提供B显数据可视化和标注管理功能

## 系统架构

```
d:\HDC\
├── gui/                         # GUI相关模块
│   ├── main_window.py           # 主窗口
│   ├── annotation_manager.py    # 标注管理
│   ├── point_selector.py        # 点选择器
│   └── type_selector.py         # 类型选择器
├── torchhd/                     # torchhd库
├── utils/                       # 工具函数
│   └── coordinate_converter.py  # 坐标转换器
├── analyze_similarity_diff.py   # 相似度差异分析
├── config.py                    # 配置管理
├── hdc_point_level_classifier.py  # HDC核心分类器
├── main.py                      # GUI主入口
├── stats_annotation_encodings.py # 标注编码统计
├── test_hdc_point_level_full.py  # 完整测试流程
├── ultrasonic_data_converter.py  # 数据转换器
├── README.md                    # 本文件
└── 软件详细设计.md              # 详细设计文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchhd numpy pandas matplotlib PyQt5
```

### 2. 数据准备

将JSON格式的原始数据放入`data/`目录：

```
data/
├── train/                       # 训练数据
│   ├── sample1.json
│   └── sample1标注.json
└── test/                        # 测试数据
    └── sample2.json
```

### 3. 运行完整测试流程

```python
python test_hdc_point_level_full.py
```

### 4. 运行GUI应用

```python
python main.py
```

## 核心功能

### 1. 数据转换

将JSON格式的超声波数据转换为CSV格式：

```python
from ultrasonic_data_converter import UltrasonicDataConverter

converter = UltrasonicDataConverter()
converter.generate_all_csvs("data/sample.json", "data/sample标注.json")
```

### 2. 模型训练

```python
from hdc_point_level_classifier import HDCPointLevelModel
import torch

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hdc_sys = HDCPointLevelModel(device=device)

# 加载训练数据
train_ds = HDCPointLevelDataset("data/train", is_train=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

# 训练模型
hdc_sys.train(train_ld, epochs=5, lr=0.1)

# 保存模型
hdc_sys.save("models/hdc_point_level_9ch.pth")
```

### 3. 模型预测

```python
from hdc_point_level_classifier import HDCPointLevelModel
import torch

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hdc_sys = HDCPointLevelModel(device=device)

# 加载模型
hdc_sys.load("models/hdc_point_level_9ch.pth")

# 加载测试数据
test_ds = HDCPointLevelDataset("data/test", is_train=False)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

# 预测
predictions, normal_sims, max_defect_sims = hdc_sys.predict(test_ld, threshold=0, margin=0)
```

### 4. 结果转换

将预测结果CSV转换为GUI可识别的JSON格式：

```python
from test_hdc_point_level_full import convert_predicted_csv_to_json

convert_predicted_csv_to_json(
    "data/test/sample_predicted_label.csv",
    "data/test/sample.json",
    "data/test/sample_predicted.json",
    "data/test/sample_hdc.csv"
)
```

## 配置说明

在`config.py`中可以调整以下参数：

### 数据配置

```python
"data": {
    "input_dir": "data",        # 输入数据目录
    "output_dir": "output",      # 输出目录
    "train_ratio": 0.7,          # 训练集比例
    "window_size": 128,          # 滑动窗口大小
    "normalize": True             # 是否归一化
}
```

### 模型配置

```python
"model": {
    "dimensions": 10000,         # 超维空间维度
    "vsa_type": "MAP",           # VSA类型
    "classifier": "OnlineHD",      # 分类器类型
    "learning_rate": 0.1,        # 学习率
    "epochs": 5                  # 训练轮数
}
```

### 评估配置

```python
"evaluation": {
    "metrics": ["precision", "recall", "f1_score"],
    "cross_validation": False
}
```

## 伤损类型

系统支持识别以下伤损类型：

| ID | 伤损类型 |
|----|----------|
| 0 | 正常 |
| 1 | 轨头外侧核伤 |
| 2 | 轨头中心核伤 |
| 3 | 轨头内侧核伤 |
| 4 | 轨头内、外、中心核伤 |
| 5 | 轨头内、中心核伤 |
| 6 | 轨头外、中心核伤 |
| 7 | 轨头水平分裂 |
| 8 | 轨头、轨腰分裂 |
| 9 | 轨腰水平裂 |
| 10 | 轨底断裂 |
| 11 | 螺孔裂纹 |
| 12 | 接头 |
| 13 | 螺孔 |
| 14 | 导线孔 |
| 15 | 焊缝 |
| 16 | 鱼鳞伤 |
| 17 | 轨底锈蚀 |
| 18 | 轨底焊筋 |
| 19 | 道岔 |

## 评估指标

### 核心指标

- **精确率(Precision)**: 预测为正的样本中真正为正的比例
- **召回率(Recall)**: 实际为正的样本中被正确预测的比例
- **F1分数(F1-Score)**: 精确率和召回率的调和平均

## 技术特点

### 1. 点级分类

采用点级分类策略，每个滑动窗口的中心点作为分类对象：

```python
# 确定窗口标签，使用窗口中心点的标签
center_idx = (start + end) // 2
window_label = labels[center_idx]  # 直接使用中心点的所有通道标签
```

### 2. 多通道独立处理

9个超声波通道独立编码和分类，每个通道使用独立的标签：

```python
# 预测结果格式：[batch_size, NUM_CHANNELS]
predictions, normal_sims, max_defect_sims = hdc_sys.predict(test_loader, threshold=0, margin=0)
```

### 3. BeamOffset调整

考虑BeamOffset调整，确保各通道数据正确对齐：

```python
# 获取 BeamOffset
for i in range(0, len(bo), 2):
    beam_offset_map[int(bo[i])] = bo[i+1]

# 应用 BeamOffset 调整
offset = beam_offset_map.get(ch, 0)
original_e = int(encoding + offset)
```

### 4. 增量学习

支持增量学习，仅训练新添加的数据：

```python
# 筛选出真正需要训练的新文件
actual_new_files = [f for f in get_json_files(TRAIN_DATA_DIR) if f.name not in learned_files]
if actual_new_files:
    # 训练新数据
    # ...
    # 更新已学习列表
    for f in actual_new_files:
        learned_files.add(f.name)
    save_learned_files(learned_files)
```

## 性能优化建议

### 1. 窗口大小优化

- **128**: 平衡精度和效率，推荐使用
- **64**: 更快速度，但可能降低精度
- **256**: 更高精度，但计算量增加

### 2. 滑动步长选择

- **窗口大小的一半**: 推荐使用，平衡重叠度和效率

### 3. 批量大小调整

- **GPU环境**: 128，充分利用GPU并行计算能力
- **CPU环境**: 8-16，平衡内存占用和计算效率

## 常见问题

### Q1: 如何提高识别准确率？

A: 可以尝试以下方法：
1. 增加超维空间维度（如20,000维）
2. 使用更多训练数据
3. 调整窗口大小和滑动步长
4. 优化学习率和训练轮数

### Q2: 如何处理类别不平衡？

A: 系统已内置处理：
1. 使用F1分数平衡精确率和召回率
2. 支持增量学习，适应新数据分布

### Q3: 如何进行增量学习？

A: 系统支持自动增量学习，新数据会自动添加到训练集中：
```python
# test_hdc_point_level_full.py会自动检测新文件并进行增量学习
python test_hdc_point_level_full.py
```

### Q4: 如何调整相似度阈值？

A: 在hdc_point_level_classifier.py的predict方法中调整threshold和margin参数：
```python
predictions, normal_sims, max_defect_sims = hdc_sys.predict(test_ld, threshold=0, margin=0)
```

## 参考文献

1. 超维计算基础理论
2. OnlineHD: Robust, Efficient, and Single-Pass Online Learning
3. PyQt5官方文档
4. matplotlib官方文档

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件

## 更新日志

### v2.0.0 (2026-01-24)
- 重构系统架构，简化模块结构
- 实现点级分类策略
- 支持9通道独立编码和分类
- 支持增量学习
- 增加BeamOffset调整
- 完善GUI可视化功能
- 提供完整的测试流程
- 更新详细设计文档

### v1.0.0 (2024-01-20)
- 初始版本发布
- 实现完整的HDC分类系统
- 支持9通道超声波数据处理
- 提供丰富的评估指标
- 支持实时识别和增量学习