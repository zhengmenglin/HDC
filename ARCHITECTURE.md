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
│  数据处理层   │    │  特征提取层   │    │  模型训练层   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 数据转换器    │    │ 特征提取器    │    │  训练器      │
│ Converter    │    │ Extractor    │    │  Trainer     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ HDC编码器     │    │ HDC分类器    │    │  评估器      │
│ Encoder      │    │ Classifier   │    │  Evaluator   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │ 实时处理器    │
                    │ Processor    │
                    └──────────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │ 主控制器      │
                    │ Controller   │
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
        "window_size": 64,           # 滑动窗口大小
        "normalize": True             # 是否归一化
    },
    "model": {
        "dimensions": 10000,         # 超维空间维度
        "vsa_type": "MAP",           # VSA类型
        "classifier": "OnlineHD",      # 分类器类型
        "learning_rate": 0.035,       # 学习率
        "epochs": 120                # 训练轮数
    },
    "evaluation": {
        "metrics": ["accuracy", "precision", "recall", "f1_score", "mcc", "auc"],
        "cross_validation": True,
        "folds": 5
    },
    "realtime": {
        "threshold": 0.5,            # 相似度阈值
        "update_interval": 100,       # 模型更新间隔
        "batch_size": 1               # 批处理大小
    }
}
```

### 2. 数据处理模块 (ultrasonic_data_converter.py)

**功能**:
- JSON到CSV格式转换
- 数据归一化处理
- 滑动窗口采样
- 训练集/测试集划分

**核心类**:
- `UltrasonicDataConverter`: 数据转换器

**主要方法**:
```python
# 数据转换
convert_json_to_csv(json_file, annotation_file)
convert_directory(input_dir, output_dir)

# 数据加载
load_csv(csv_file)

# 数据预处理
normalize_data(data)
apply_window_sampling(data, labels, window_size)
split_train_test(data, labels, train_ratio)
```

**数据流程**:
```
JSON数据 → 解析B扫描数据 → 按编码值分组 → 生成CSV → 归一化 → 划分训练/测试集
```

### 3. 特征提取模块 (feature_extractor.py)

**功能**:
- 9个通道的独立编码
- 超维向量生成
- 特征融合
- 序列编码

**核心类**:
- `FeatureExtractor`: 特征提取器

**主要方法**:
```python
# 通道分离
separate_channels(data)

# 通道编码
encode_channel(channel_data, encoder)
encode_all_channels(data)

# 特征融合
bundle_channels(channel_hvs)

# 序列编码
create_sequence_encoding(data, sequence_length)
```

**编码流程**:
```
输入数据 → 通道分离 → Level编码 → 特征融合 → 超维向量
```

### 4. HDC编码模块 (hd_encoder.py)

**功能**:
- Level编码（连续数值）
- 循环移位编码
- 绑定和捆绑操作
- 相似度计算

**核心类**:
- `HDCEncoder`: HDC编码器

**主要方法**:
```python
# 数值编码
encode_depth_continuous(depth, min_depth, max_depth)
encode_amplitude_continuous(amplitude, min_amp, max_amp)
encode_channel(channel_idx)
encode_encoding_value(encoding_value)

# 高级编码
circular_shift_encode(hvs)
bundle_encode(hvs)
bind_encode(hvs)

# 相似度计算
compute_similarity(hv1, hv2)
```

**编码策略**:
1. **Level编码**: 将连续数值映射到超维空间
2. **循环移位编码**: 表示序列中的位置信息
3. **绑定操作**: 组合不同的特征
4. **捆绑操作**: 聚合多个超向量

### 5. HDC分类模块 (hd_classifier.py)

**功能**:
- 基于torchhd的分类器封装
- 支持多种分类器类型
- 模型训练和预测
- 模型保存和加载

**核心类**:
- `HDCClasifier`: HDC分类器

**支持的分类器**:
- `OnlineHD`: 在线学习，支持增量更新
- `AdaptHD`: 自适应训练
- `Vanilla`: 基线分类器

**主要方法**:
```python
# 训练
train(data, labels, epochs, lr)

# 预测
predict(data, threshold)
predict_with_confidence(data)

# 数据编码
encode_data(data)

# 模型管理
save_model(filepath)
load_model(filepath)
```

**分类流程**:
```
输入数据 → HDC编码 → 相似度计算 → 类别判决 → 输出结果
```

### 6. 训练模块 (trainer.py)

**功能**:
- 数据准备和预处理
- 模型训练
- 交叉验证
- 增量训练

**核心类**:
- `Trainer`: 训练器

**主要方法**:
```python
# 数据准备
prepare_data(input_dir, output_dir, normalize, window_sampling)

# 模型训练
train_model(train_data, train_labels, test_data, test_labels)
train_from_files(train_csv, test_csv, model_path, save_path)

# 交叉验证
cross_validate(data, labels, folds)

# 增量训练
train_incremental(model_path, new_data, new_labels)
```

**训练流程**:
```
数据准备 → 数据转换 → 模型训练 → 模型评估 → 保存模型
```

### 7. 评估模块 (evaluator.py)

**功能**:
- 多指标评估
- 混淆矩阵生成
- F1/F2分数计算
- MCC和AUC计算

**核心类**:
- `Evaluator`: 评估器

**评估指标**:
- **准确率(Accuracy)**: 整体准确率
- **精确率(Precision)**: 预测为正的样本中真正为正的比例
- **召回率(Recall)**: 实际为正的样本中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **F2分数**: 召回率权重更高的F分数
- **MCC**: 马修斯相关系数
- **AUC**: ROC曲线下面积
- **漏检率(FNR)**: 实际为正但预测为负的比例
- **误报率(FPR)**: 实际为负但预测为正的比例

**主要方法**:
```python
# 评估
evaluate(classifier, data, labels)

# 指标计算
calculate_f2_score(y_true, y_pred, beta)
calculate_mcc(y_true, y_pred, num_classes)
calculate_auc(y_true, y_pred, num_classes)

# 结果输出
print_evaluation_report(results, classifier)
save_evaluation_results(results, filepath)
```

### 8. 实时处理模块 (realtime_processor.py)

**功能**:
- 实时数据流处理
- 在线识别
- 增量学习
- 统计信息管理

**核心类**:
- `RealtimeProcessor`: 实时处理器

**主要方法**:
```python
# 数据处理
process_sample(sample, true_label)
process_batch(samples, true_labels)
process_stream(data_stream, true_labels)

# 模型更新
update_model_incremental(new_data, new_labels)

# 统计管理
get_statistics()
reset_statistics()

# 结果输出
save_predictions(filepath)
generate_annotation_json(predictions, output_path)
```

**实时处理流程**:
```
数据流 → 单样本处理 → 批量处理 → 模型更新 → 结果输出
```

### 9. 主控制器模块 (main_controller.py)

**功能**:
- 整合所有模块
- 提供统一接口
- 交互模式
- 流水线执行

**核心类**:
- `MainController`: 主控制器

**主要方法**:
```python
# 数据转换
convert_data(input_dir, output_dir)

# 模型训练
train_model(train_csv, test_csv, model_path, save_path)

# 模型评估
evaluate_model(test_csv, model_path)

# 实时识别
realtime_recognition(data_stream, model_path)

# 完整流水线
run_pipeline(input_dir, model_path)

# 交互模式
interactive_mode()
```

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
      ▼
┌─────────────┐
│  数据预处理   │ (归一化、窗口采样)
└─────────────┘
      │
      ▼
┌─────────────┐
│  特征提取    │ (通道分离、Level编码)
└─────────────┘
      │
      ▼
┌─────────────┐
│  HDC编码     │ (超维向量生成)
└─────────────┘
      │
      ▼
┌─────────────┐
│  模型训练    │ (OnlineHD)
└─────────────┘
      │
      ▼
┌─────────────┐
│  模型评估    │ (多指标评估)
└─────────────┘
      │
      ▼
┌─────────────┐
│  实时识别    │ (数据流处理)
└─────────────┘
      │
      ▼
┌─────────────┐
│  结果输出    │ (标注JSON)
└─────────────┘
```

## 技术特点

### 1. 通道分离处理

9个超声波通道独立编码，每个通道使用Level编码器：

```python
for channel in range(9):
    channel_hv = level_encoder.encode(channel_data[channel])
```

### 2. 循环移位编码

使用循环移位编码表示多个点的深度值：

```python
for i in range(len(depths)):
    depth_hv = level_encoder.encode(depths[i])
    shifted_hv = permute(depth_hv, shifts=i)
    fused_hv = bundle(fused_hv, shifted_hv)
```

### 3. 特征融合

将编码值和通道数据通过bind操作融合：

```python
encoding_hv = encode_encoding_value(encoding)
channels_hv = bundle(channel_hvs)
final_hv = bind(encoding_hv, channels_hv)
```

### 4. 在线学习

使用OnlineHD分类器，支持增量学习：

```python
for new_sample in new_data:
    classifier.update(new_sample, new_label)
```

## 性能优化

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

## 扩展性

### 添加新的编码器

```python
class CustomEncoder(HDCEncoder):
    def custom_encode(self, data):
        # 自定义编码逻辑
        pass
```

### 添加新的分类器

```python
class CustomClassifier(HDCClasifier):
    def __init__(self):
        self.model = torchhd.YourCustomClassifier(...)
```

### 添加新的评估指标

```python
class CustomEvaluator(Evaluator):
    def custom_metric(self, y_true, y_pred):
        # 自定义指标计算
        pass
```

## 部署建议

### 1. 离线分析环境

- 使用高维度（20,000）
- 使用完整的评估指标
- 支持交叉验证
- 优先保证准确率

### 2. 在线检测环境

- 使用中等维度（10,000）
- 使用OnlineHD分类器
- 支持增量学习
- 设置合适的相似度阈值

### 3. 边缘计算环境

- 使用低维度（5,000）
- 使用量化模型
- 优化推理速度
- 降低内存占用

## 总结

本系统采用模块化设计，各个模块职责清晰，易于扩展和维护。系统充分利用了超维计算的优势，在保证高准确率的同时，支持实时处理和增量学习，非常适合超声波探伤的应用场景。

通过合理的配置和优化，系统可以在不同的硬件环境和应用场景中发挥最佳性能。