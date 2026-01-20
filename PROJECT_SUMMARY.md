# 超声波探伤HDC系统 - 项目总结

## 项目概述

本项目实现了一个基于超维计算(HDC)的钢轨超声波探伤智能识别系统。系统采用模块化设计，支持9个超声波通道的独立编码和融合，提供高准确率的伤损识别能力。

## 项目文件结构

```
d:\HDC\
├── config.py                    # 配置管理模块
├── ultrasonic_data_converter.py  # 数据转换器
├── feature_extractor.py         # 特征提取器
├── hd_encoder.py               # HDC编码器
├── hd_classifier.py            # HDC分类器
├── trainer.py                  # 训练器
├── evaluator.py                # 评估器
├── realtime_processor.py        # 实时处理器
├── main_controller.py           # 主控制器
├── examples.py                 # 使用示例
├── test_system.py              # 系统测试脚本
├── generate_sample_data.py      # 示例数据生成器
├── requirements.txt            # 依赖包列表
├── README.md                  # 项目说明文档
├── QUICKSTART.md             # 快速开始指南
├── ARCHITECTURE.md           # 系统架构文档
└── PROJECT_SUMMARY.md        # 本文件
```

## 核心模块说明

### 1. config.py - 配置管理模块

**功能**:
- 统一管理系统配置参数
- 支持配置文件的加载和保存
- 提供配置项的读写接口

**主要类**:
- `Config`: 配置管理类

**关键方法**:
- `load_config()`: 加载配置文件
- `save_config()`: 保存配置文件
- `get(key, default)`: 获取配置项
- `set(key, value)`: 设置配置项

### 2. ultrasonic_data_converter.py - 数据转换器

**功能**:
- JSON到CSV格式转换
- 数据归一化处理
- 滑动窗口采样
- 训练集/测试集划分

**主要类**:
- `UltrasonicDataConverter`: 数据转换器

**关键方法**:
- `convert_json_to_csv()`: 将JSON转换为CSV
- `convert_directory()`: 批量转换目录中的文件
- `load_csv()`: 加载CSV文件
- `normalize_data()`: 数据归一化
- `apply_window_sampling()`: 滑动窗口采样
- `split_train_test()`: 划分训练集和测试集

### 3. feature_extractor.py - 特征提取器

**功能**:
- 9个通道的独立编码
- 超维向量生成
- 特征融合
- 序列编码

**主要类**:
- `FeatureExtractor`: 特征提取器

**关键方法**:
- `separate_channels()`: 分离9个通道
- `encode_channel()`: 编码单个通道
- `encode_all_channels()`: 编码所有通道
- `bundle_channels()`: 捆绑多个通道
- `create_sequence_encoding()`: 创建序列编码
- `extract_features()`: 提取特征

### 4. hd_encoder.py - HDC编码器

**功能**:
- Level编码（连续数值）
- 循环移位编码
- 绑定和捆绑操作
- 相似度计算

**主要类**:
- `HDCEncoder`: HDC编码器

**关键方法**:
- `encode_depth_continuous()`: 编码深度值
- `encode_amplitude_continuous()`: 编码幅度值
- `encode_channel()`: 编码通道号
- `encode_encoding_value()`: 编码编码值
- `circular_shift_encode()`: 循环移位编码
- `bundle_encode()`: 捆绑编码
- `bind_encode()`: 绑定编码
- `compute_similarity()`: 计算相似度

### 5. hd_classifier.py - HDC分类器

**功能**:
- 基于torchhd的分类器封装
- 支持多种分类器类型
- 模型训练和预测
- 模型保存和加载

**主要类**:
- `HDCClasifier`: HDC分类器

**支持的分类器**:
- `OnlineHD`: 在线学习，支持增量更新
- `AdaptHD`: 自适应训练
- `Vanilla`: 基线分类器

**关键方法**:
- `train()`: 训练模型
- `predict()`: 预测数据
- `predict_with_confidence()`: 预测并返回置信度
- `encode_data()`: 编码数据
- `save_model()`: 保存模型
- `load_model()`: 加载模型
- `get_class_name()`: 获取类别名称
- `get_model_info()`: 获取模型信息

### 6. trainer.py - 训练器

**功能**:
- 数据准备和预处理
- 模型训练
- 交叉验证
- 增量训练

**主要类**:
- `Trainer`: 训练器

**关键方法**:
- `prepare_data()`: 准备训练数据
- `train_model()`: 训练模型
- `train_from_files()`: 从CSV文件训练
- `cross_validate()`: 交叉验证
- `train_incremental()`: 增量训练

### 7. evaluator.py - 评估器

**功能**:
- 多指标评估
- 混淆矩阵生成
- F1/F2分数计算
- MCC和AUC计算

**主要类**:
- `Evaluator`: 评估器

**评估指标**:
- 准确率(Accuracy)
- 精确率(Precision)
- 召回率(Recall)
- F1分数
- F2分数
- MCC（马修斯相关系数）
- AUC（ROC曲线下面积）
- 漏检率(FNR)
- 误报率(FPR)

**关键方法**:
- `evaluate()`: 评估模型
- `calculate_f2_score()`: 计算F2分数
- `calculate_mcc()`: 计算MCC
- `calculate_auc()`: 计算AUC
- `print_evaluation_report()`: 打印评估报告
- `save_evaluation_results()`: 保存评估结果

### 8. realtime_processor.py - 实时处理器

**功能**:
- 实时数据流处理
- 在线识别
- 增量学习
- 统计信息管理

**主要类**:
- `RealtimeProcessor`: 实时处理器

**关键方法**:
- `process_sample()`: 处理单个样本
- `process_batch()`: 批量处理
- `process_stream()`: 处理数据流
- `update_model_incremental()`: 增量更新模型
- `get_statistics()`: 获取统计信息
- `save_predictions()`: 保存预测结果
- `generate_annotation_json()`: 生成标注JSON

### 9. main_controller.py - 主控制器

**功能**:
- 整合所有模块
- 提供统一接口
- 交互模式
- 流水线执行

**主要类**:
- `MainController`: 主控制器

**关键方法**:
- `convert_data()`: 转换数据格式
- `train_model()`: 训练模型
- `evaluate_model()`: 评估模型
- `realtime_recognition()`: 实时识别
- `run_pipeline()`: 运行完整流水线
- `interactive_mode()`: 交互模式

## 辅助脚本

### examples.py - 使用示例

提供了7个详细的使用示例：
1. 基本使用流程
2. 实时识别
3. 增量训练
4. 交叉验证
5. 自定义配置
6. 特征提取
7. HDC编码

### test_system.py - 系统测试

测试所有模块的功能：
1. 配置模块测试
2. 数据转换器测试
3. 特征提取器测试
4. HDC编码器测试
5. HDC分类器测试
6. 训练器测试
7. 评估器测试
8. 实时处理器测试

### generate_sample_data.py - 示例数据生成器

生成模拟的超声波探伤数据用于测试系统：
- 生成训练集和测试集
- 支持自定义伤损比例
- 生成JSON格式的原始数据和标注数据

## 文档说明

### README.md - 项目说明文档

包含以下内容：
- 系统特点
- 系统架构
- 快速开始
- 核心功能
- 配置说明
- 伤损类型
- 评估指标
- 使用示例
- 技术特点
- 性能优化建议
- 常见问题

### QUICKSTART.md - 快速开始指南

提供快速上手指南：
- 安装步骤
- 快速使用
- 数据格式
- 测试系统
- 运行示例
- 关键配置
- 常见问题

### ARCHITECTURE.md - 系统架构文档

详细的系统架构说明：
- 系统架构图
- 模块详细说明
- 数据流图
- 技术特点
- 性能优化
- 扩展性
- 部署建议

### PROJECT_SUMMARY.md - 项目总结

本文件，提供项目整体概览。

## 技术栈

### 核心依赖
- **torch**: PyTorch深度学习框架
- **torchhd**: 超维计算库
- **numpy**: 数值计算
- **scikit-learn**: 机器学习工具
- **scipy**: 科学计算

### HDC技术
- **Level编码**: 连续数值编码
- **循环移位编码**: 序列编码
- **绑定操作**: 特征组合
- **捆绑操作**: 特征聚合
- **余弦相似度**: 相似度计算

### 分类器
- **OnlineHD**: 在线学习
- **AdaptHD**: 自适应训练
- **Vanilla**: 基线分类器

## 支持的伤损类型

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

## 核心评估指标

### 核心指标
- **F2分数**: ≥0.85（优先保证召回率）
- **MCC**: ≥0.85（最公平的单值指标）
- **PR-AUC**: ≥0.9（不平衡数据黄金指标）
- **漏检率**: ≤1%（一票否决指标）

### 辅助指标
- **准确率(Accuracy)**: 整体准确率
- **精确率(Precision)**: 预测为正的样本中真正为正的比例
- **召回率(Recall)**: 实际为正的样本中被正确预测的比例
- **ROC-AUC**: ≥0.95（模型区分能力）
- **误报率(FPR)**: 实际为负的样本中被误判为正的比例

## 使用流程

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据准备
```bash
python generate_sample_data.py
```

### 3. 运行测试
```bash
python test_system.py
```

### 4. 交互模式
```bash
python main_controller.py interactive
```

### 5. 完整流水线
```bash
python main_controller.py pipeline data/
```

## 项目特点

### 1. 模块化设计
- 各模块职责清晰
- 易于扩展和维护
- 支持独立测试

### 2. 高度可配置
- 统一的配置管理
- 灵活的参数调整
- 支持多种VSA类型

### 3. 完整的评估体系
- 多种评估指标
- 详细的评估报告
- 支持交叉验证

### 4. 实时处理能力
- 支持数据流处理
- 在线识别
- 增量学习

### 5. 丰富的文档
- 详细的使用说明
- 完整的架构文档
- 丰富的代码示例

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

## 扩展性

### 添加新的编码器
继承`HDCEncoder`类，实现自定义编码逻辑

### 添加新的分类器
继承`HDCClasifier`类，使用torchhd的其他分类器

### 添加新的评估指标
继承`Evaluator`类，实现自定义指标计算

## 总结

本项目实现了一个完整的超声波探伤HDC系统，具有以下特点：

1. **完整性**: 包含数据处理、特征提取、模型训练、评估和实时识别等完整流程
2. **模块化**: 各模块职责清晰，易于扩展和维护
3. **可配置**: 统一的配置管理，灵活的参数调整
4. **高性能**: 基于torchhd的高效HDC实现
5. **易用性**: 提供交互模式和丰富的示例代码
6. **文档完善**: 详细的使用说明和架构文档

系统充分利用了超维计算的优势，在保证高准确率的同时，支持实时处理和增量学习，非常适合超声波探伤的应用场景。