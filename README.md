# 超声波探伤HDC系统

基于超维计算(HDC)的钢轨超声波探伤智能识别系统。

## 系统特点

- **准确性优先**: 专为离线数据分析设计，追求高准确率
- **多通道处理**: 支持9个超声波通道的独立编码和融合
- **在线学习**: 支持增量学习，适应现场数据变化
- **丰富评估**: 提供F1、F2、MCC、AUC等多种评估指标
- **实时识别**: 支持实时数据流处理和伤损识别

## 系统架构

```
d:\HDC\
├── config.py                    # 配置管理
├── ultrasonic_data_converter.py  # 数据转换器
├── feature_extractor.py         # 特征提取器
├── hd_encoder.py               # HDC编码器
├── hd_classifier.py            # HDC分类器
├── trainer.py                  # 训练器
├── evaluator.py                # 评估器
├── realtime_processor.py        # 实时处理器
├── main_controller.py           # 主控制器
├── examples.py                 # 使用示例
└── README.md                   # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchhd numpy scikit-learn scipy
```

### 2. 数据准备

将JSON格式的原始数据放入`data/`目录：

```
data/
├── sample1.json
├── sample1标注.json
├── sample2.json
├── sample2标注.json
└── ...
```

### 3. 运行完整流水线

```python
python main_controller.py pipeline data/
```

### 4. 交互模式

```python
python main_controller.py interactive
```

## 核心功能

### 数据转换

将JSON格式的超声波数据转换为CSV格式：

```python
from main_controller import MainController

controller = MainController()
csv_files = controller.convert_data("data/", "output/")
```

### 模型训练

```python
# 训练模型
train_results = controller.train_model(
    train_csv="output/train.csv",
    model_path="output/hd_model.pth"
)
```

### 模型评估

```python
# 评估模型
eval_results = controller.evaluate_model(
    test_csv="output/test.csv",
    model_path="output/hd_model.pth"
)
```

### 实时识别

```python
# 实时识别数据流
results = controller.realtime_recognition(
    data_stream=data_stream,
    model_path="output/hd_model.pth"
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
    "window_size": 64,           # 滑动窗口大小
    "normalize": True             # 是否归一化
}
```

### 模型配置

```python
"model": {
    "dimensions": 10000,         # 超维空间维度
    "vsa_type": "MAP",           # VSA类型: MAP, HRR, FHRR
    "classifier": "OnlineHD",      # 分类器类型
    "learning_rate": 0.035,       # 学习率
    "epochs": 120                # 训练轮数
}
```

### 评估配置

```python
"evaluation": {
    "metrics": ["accuracy", "precision", "recall", "f1_score", "mcc", "auc"],
    "cross_validation": True,
    "folds": 5
}
```

### 实时配置

```python
"realtime": {
    "threshold": 0.5,            # 相似度阈值
    "update_interval": 100,       # 模型更新间隔
    "batch_size": 1               # 批处理大小
}
```

## 伤损类型

系统支持识别以下伤损类型：

| ID | 伤损类型 |
|----|----------|
| 0 | 未识别 |
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

- **F2分数**: 优先保证召回率（少漏检），阈值≥0.85
- **MCC**: 马修斯相关系数，最公平的单值指标，阈值≥0.85
- **PR-AUC**: 精确率-召回率曲线下面积，阈值≥0.9
- **漏检率(FNR)**: 一票否决指标，阈值≤1%

### 辅助指标

- **准确率(Accuracy)**: 整体准确率
- **精确率(Precision)**: 预测为正的样本中真正为正的比例
- **召回率(Recall)**: 实际为正的样本中被正确预测的比例
- **ROC-AUC**: 受试者工作特征曲线下面积，阈值≥0.95
- **误报率(FPR)**: 实际为负的样本中被误判为正的比例

## 使用示例

### 示例1: 基本使用流程

```python
from main_controller import MainController

controller = MainController()

# 1. 数据转换
csv_files = controller.convert_data("data/", "output/")

# 2. 模型训练
train_results = controller.train_model(
    train_csv="output/train.csv",
    model_path="output/hd_model.pth"
)

# 3. 模型评估
eval_results = controller.evaluate_model(
    test_csv="output/test.csv",
    model_path="output/hd_model.pth"
)
```

### 示例2: 实时识别

```python
from main_controller import MainController
import numpy as np

controller = MainController()

# 模拟实时数据流
data_stream = []
for i in range(100):
    sample = np.random.rand(10) * 200
    sample[0] = np.random.randint(0, 999)
    data_stream.append(sample)

# 实时识别
results = controller.realtime_recognition(
    data_stream=data_stream,
    model_path="output/hd_model.pth"
)

# 生成标注文件
controller.realtime_processor.generate_annotation_json(
    results,
    "output/realtime_annotations.json"
)
```

### 示例3: 增量训练

```python
from realtime_processor import RealtimeProcessor
import numpy as np

processor = RealtimeProcessor(model_path="output/hd_model.pth")

# 模拟新数据
new_data = np.random.rand(50, 10) * 200
new_labels = np.random.randint(1, 20, 50)

# 增量训练
for i in range(len(new_data)):
    processor.process_sample(new_data[i], new_labels[i])
    
    # 定期更新模型
    if processor.total_samples % 100 == 0:
        processor.update_model_incremental(new_data, new_labels)
```

### 示例4: 交叉验证

```python
from trainer import Trainer
from ultrasonic_data_converter import UltrasonicDataConverter

trainer = Trainer()
converter = UltrasonicDataConverter()

# 加载数据
data, labels = converter.load_csv("output/train.csv")

# 交叉验证
cv_results = trainer.cross_validate(data, labels, folds=5)

print(f"平均准确率: {cv_results['average_results']['accuracy']:.4f}")
```

## 技术特点

### 1. 通道分离处理

9个超声波通道独立编码，每个通道使用Level编码器：

```python
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor(dimensions=10000)
channel_hvs = extractor.encode_all_channels(data)
```

### 2. 循环移位编码

使用循环移位编码表示多个点的深度值：

```python
from hd_encoder import HDCEncoder

encoder = HDCEncoder(dimensions=10000)
sequence_hv = encoder.circular_shift_encode(hvs)
```

### 3. 特征融合

将编码值和通道数据通过bind操作融合：

```python
from hd_encoder import HDCEncoder

encoder = HDCEncoder(dimensions=10000)
final_hv = encoder.encode_multi_channel_sample(sample)
```

### 4. 在线学习

使用OnlineHD分类器，支持增量学习：

```python
from hd_classifier import HDCClasifier

classifier = HDCClasifier(classifier_type="OnlineHD")
classifier.train(train_data, train_labels)
```

## 性能优化建议

### 1. 维度选择

- **10,000维**: 平衡精度和效率，推荐使用
- **20,000维**: 更高精度，但计算量增加
- **5,000维**: 更快速度，但可能降低精度

### 2. VSA类型选择

- **MAP**: 推荐使用，适合大多数场景
- **FHRR**: 适合需要复数运算的场景
- **HRR**: 适合需要精确绑定的场景

### 3. 分类器选择

- **OnlineHD**: 推荐使用，支持在线学习
- **AdaptHD**: 自适应训练
- **Vanilla**: 简单快速

## 常见问题

### Q1: 如何提高识别准确率？

A: 可以尝试以下方法：
1. 增加超维空间维度（如20,000维）
2. 使用更多训练数据
3. 调整相似度阈值
4. 使用交叉验证优化参数

### Q2: 如何处理类别不平衡？

A: 系统已内置处理：
1. 使用F2分数优先保证召回率
2. MCC指标对不平衡数据鲁棒
3. 支持加权平均计算

### Q3: 如何进行增量学习？

A: 使用实时处理器的增量训练功能：
```python
processor.update_model_incremental(new_data, new_labels)
```

### Q4: 如何调整相似度阈值？

A: 在配置文件中设置：
```python
config.set("realtime.threshold", 0.6)
```

## 参考文献

1. 超维计算基础理论
2. OnlineHD: Robust, Efficient, and Single-Pass Online Learning
3. AdaptHD: Adaptive Efficient Training for Brain-Inspired Hyperdimensional Computing

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件

## 更新日志

### v1.0.0 (2024-01-20)
- 初始版本发布
- 实现完整的HDC分类系统
- 支持9通道超声波数据处理
- 提供丰富的评估指标
- 支持实时识别和增量学习