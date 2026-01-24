# 超声波探伤HDC系统 - 系统架构文档

## 系统概述

本系统基于超维计算(HDC)技术，专门用于钢轨超声波探伤的智能识别。系统采用模块化设计，支持9个超声波通道的独立编码和融合，提供高准确率的伤损识别能力。

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    超声波探伤HDC系统                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  数据转换模块  │    │  HDC核心模块   │    │  GUI可视化模块  │
│  Converter    │    │  Classifier   │    │  GUI         │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ JSON → CSV   │    │  模型训练     │    │  B显可视化    │
│ 转换         │    │  与预测       │    │  标注管理     │
└──────────────┘    └──────────────┘    └──────────────┘
                              │
                              │
                              ▼
                    ┌──────────────┐
                    │  测试管道     │
                    │  Pipeline    │
                    └──────────────┘
                              │
                              │
                              ▼
                    ┌──────────────┐
                    │  辅助工具     │
                    │  Tools       │
                    └──────────────┘
```

## 模块详细说明

### 1. 配置管理模块 (config.py)

**功能**:
- 统一管理系统配置参数
- 支持配置文件的加载和保存
- 提供配置项的读写接口

**核心类**:
- `Config`: 配置管理类

**主要配置项**:
```python
{
    "data": {
        "input_dir": "data",        # 输入数据目录
        "output_dir": "output",      # 输出目录
        "train_ratio": 0.7,          # 训练集比例
        "window_size": 128,          # 滑动窗口大小
        "normalize": True             # 是否归一化
    },
    "model": {
        "dimensions": 10000,         # 超维空间维度
        "vsa_type": "MAP",           # VSA类型
        "classifier": "OnlineHD",      # 分类器类型
        "learning_rate": 0.1,        # 学习率
        "epochs": 5                  # 训练轮数
    },
    "evaluation": {
        "metrics": ["precision", "recall", "f1_score"],
        "cross_validation": False
    }
}
```

### 2. 数据转换模块 (ultrasonic_data_converter.py)

**功能**:
- JSON到CSV格式转换
- 支持9个超声波通道的数据处理
- 考虑BeamOffset调整
- 生成特征CSV和标签CSV

**核心类**:
- `UltrasonicDataConverter`: 数据转换器

**主要方法**:
```python
# 生成所有CSV文件
generate_all_csvs(json_file, annotation_path=None)

# JSON到特征CSV转换
convert_json_to_hdc_csv(json_file)

# 标注到标签CSV转换
convert_annotation_to_label_csv(annotation_file)
```

**数据流程**:
```
JSON数据 → 解析B扫描数据 → 应用BeamOffset调整 → 按编码值分组 → 生成特征CSV和标签CSV
```

### 3. HDC核心模块 (hdc_point_level_classifier.py)

**功能**:
- 数据集加载和管理
- 超维编码和分类
- 模型训练和预测
- 模型保存和加载
- 评估指标计算

**核心类**:
- `HDCPointLevelDataset`: 数据集类，处理训练和测试数据
- `HDCPointLevelModel`: HDC模型类，包含训练和预测逻辑

**主要方法**:
```python
# 训练方法
def train(self, train_loader, epochs=5, lr=0.1):
    # 模型训练逻辑
    pass

# 预测方法
def predict(self, test_loader, threshold=0, margin=0):
    # 模型预测逻辑
    pass

# 模型保存
def save(self, path):
    # 保存模型到文件
    pass

# 模型加载
def load(self, path):
    # 从文件加载模型
    pass
```

**分类流程**:
```
输入数据 → 滑动窗口生成 → 超维编码 → 相似度计算 → 类别判决 → 输出结果
```

### 4. GUI可视化模块 (gui/)

**功能**:
- B显数据可视化
- 标注管理
- HDC识别结果可视化
- 文件操作

**核心类**:
- `MainWindow`: 主窗口类，管理GUI布局和事件
- `AnnotationManager`: 标注管理类，处理标注的创建、保存和加载
- `BScanCanvas`: B显画布类，负责绘制B显数据
- `PointSelector`: 点选择器类，支持在画布上选择点
- `TypeSelectorDialog`: 类型选择对话框，支持选择标注类型

**主要方法**:
```python
# 主窗口方法
def open_file_by_path(self, file_path):
    # 打开并加载JSON文件
    pass

def update_bscan(self):
    # 更新B显图像
    pass

# 标注管理方法
def load_annotations(self, file_path):
    # 加载标注数据
    pass

def save_annotations(self, file_path):
    # 保存标注数据
    pass
```

### 5. 测试管道模块 (test_hdc_point_level_full.py)

**功能**:
- 整合数据转换、模型训练、预测和结果转换
- 支持增量学习
- 生成评估报告
- 清理旧的中间文件

**核心方法**:
```python
# 完整测试流程
def run_full_test():
    # 1. 环境准备与数据转换
    # 2. 模型训练 (增量)
    # 3. 执行预测与转换
    # 4. 汇总训练集指标
    pass

# 预测结果转换
def convert_predicted_csv_to_json(csv_path, original_json_path, output_json_path, hdc_csv_path=None):
    # 将预测结果CSV转换为GUI可识别的_predicted.json格式
    pass

# 清理旧文件
def clean_old_files(data_dirs, file_types="all"):
    # 清理旧的中间文件
    pass

# 计算评估指标
def calculate_metrics(y_true, y_pred):
    # 计算精确率、召回率和F1分数
    pass
```

### 6. 辅助工具模块

**功能**:
- 相似度差异分析
- 标注编码统计

**核心文件**:
- `analyze_similarity_diff.py`: 分析伤损相似度大于正常相似度的数据点
- `stats_annotation_encodings.py`: 统计标注信息和拼孔后的位置

## 数据流图

```
┌─────────────┐
│  原始数据    │ (JSON格式)
└─────────────┘
      │
      ▼
┌─────────────┐
│  数据转换    │ (JSON → CSV)
└─────────────┘
      │
      ├─────────────┬─────────────┐
      │             │             │
      ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  模型训练    │ │  模型预测    │ │  标注统计    │
└─────────────┘ └─────────────┘ └─────────────┘
      │             │             │
      │             ▼             │
      │       ┌─────────────┐     │
      │       │  结果转换    │     │
      │       │ (CSV → JSON) │     │
      │       └─────────────┘     │
      │             │             │
      └─────────────┼─────────────┘
                    │
                    ▼
┌─────────────────────────────────┐
│           GUI可视化            │
└─────────────────────────────────┘
```

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

## 性能优化

### 1. 窗口大小优化

- **128**: 平衡精度和效率，推荐使用
- **64**: 更快速度，但可能降低精度
- **256**: 更高精度，但计算量增加

### 2. 滑动步长选择

- **窗口大小的一半**: 推荐使用，平衡重叠度和效率
- **窗口大小**: 更快速度，但可能降低精度

### 3. 批量大小调整

- **GPU环境**: 128，充分利用GPU并行计算能力
- **CPU环境**: 8-16，平衡内存占用和计算效率

## 扩展性

### 添加新的伤损类型

在hdc_point_level_classifier.py中更新FLAW_ID_TO_NAME字典：

```python
FLAW_ID_TO_NAME = {
    1: "轨头外侧核伤",
    # ... 现有类型
    20: "新伤损类型",  # 添加新类型
}
```

### 扩展评估指标

在calculate_metrics函数中添加新指标：

```python
def calculate_metrics(y_true, y_pred):
    # 现有指标计算
    # ...
    
    # 添加新指标
    # ...
    
    return {
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        # 新增指标
        # ...
    }
```

### 扩展GUI功能

继承现有GUI类，添加新功能：

```python
class EnhancedMainWindow(MainWindow):
    def new_feature(self):
        # 新功能实现
        pass
```

## 部署建议

### 1. 训练环境

- 使用GPU加速训练
- 足够的内存用于加载和处理数据
- 推荐使用10,000维超向量

### 2. 测试环境

- 可以使用CPU进行预测
- 推荐使用10,000维超向量
- 设置合适的阈值和Margin参数

### 3. GUI环境

- 安装PyQt5和matplotlib
- 确保足够的内存用于加载大型B显数据

## 总结

本系统采用模块化设计，各个模块职责清晰，易于扩展和维护。系统充分利用了超维计算的优势，在保证高准确率的同时，支持点级分类和增量学习，非常适合钢轨超声波探伤的应用场景。

系统的核心模块包括数据转换、HDC分类、GUI可视化和测试管道，各模块之间通过明确的数据格式进行交互，便于独立开发和测试。

通过合理的配置和优化，系统可以在不同的硬件环境和应用场景中发挥最佳性能，为钢轨探伤提供可靠的智能识别支持。