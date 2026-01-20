# 超声波探伤HDC系统 - 快速开始指南

## 系统概述

本系统基于超维计算(HDC)技术，专门用于钢轨超声波探伤的智能识别。系统支持9个超声波通道的独立编码和融合，提供高准确率的伤损识别能力。

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository_url>
cd HDC
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 创建目录结构
```bash
mkdir -p data output
```

## 快速使用

### 方式1: 交互模式（推荐新手）
```bash
python main_controller.py interactive
```
然后按照提示输入命令即可。

### 方式2: 完整流水线
```bash
python main_controller.py pipeline data/
```
这将自动完成数据转换、模型训练、评估和实时识别。

### 方式3: 分步执行
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

## 数据格式

### 输入数据格式
将JSON格式的原始数据放入`data/`目录：

```json
{
  "B1": [6, 125.0, 80.0, 257.0, 176.0],
  "B2": [6, 130.0, 75.0, 260.0, 180.0],
  "E": 5
}
```

### 标注数据格式
```json
{
  "annotations": [
    {
      "id": "1",
      "type": "结构",
      "subtype": {
        "id": "101",
        "name": "螺孔"
      },
      "points": [
        {
          "e": 592,
          "channel": 6,
          "depth": 90.0,
          "amplitude": 100.0
        }
      ]
    }
  ]
}
```

## 测试系统

运行测试脚本验证系统功能：
```bash
python test_system.py
```

## 运行示例

查看各种使用示例：
```bash
python examples.py
```

## 关键配置

### 修改配置
在`config.py`中调整参数：

```python
# 增加维度提高精度
config.set("model.dimensions", 20000)

# 调整相似度阈值
config.set("realtime.threshold", 0.6)

# 更改分类器类型
config.set("model.classifier", "AdaptHD")
```

### 推荐配置
- **离线分析**: 维度=20000, 分类器=OnlineHD, 阈值=0.5
- **实时检测**: 维度=10000, 分类器=OnlineHD, 阈值=0.6

## 常见问题

### Q: 如何提高识别准确率？
1. 增加超维空间维度到20000
2. 使用更多训练数据
3. 调整相似度阈值
4. 使用交叉验证优化参数

### Q: 如何处理类别不平衡？
系统已内置处理：
1. F2分数优先保证召回率
2. MCC指标对不平衡数据鲁棒
3. 支持加权平均计算

### Q: 如何进行增量学习？
```python
from realtime_processor import RealtimeProcessor

processor = RealtimeProcessor(model_path="output/hd_model.pth")
processor.update_model_incremental(new_data, new_labels)
```

## 输出结果

### 模型文件
- `hd_model.pth`: 训练好的模型
- `hd_model_info.json`: 模型信息
- `evaluation_results.json`: 评估结果

### 标注文件
- `realtime_annotations.json`: 实时识别结果
- `*_标注.json`: 手动标注结果

## 性能指标

### 核心指标
- **F2分数**: ≥0.85（优先保证召回率）
- **MCC**: ≥0.85（最公平的单值指标）
- **PR-AUC**: ≥0.9（不平衡数据黄金指标）
- **漏检率**: ≤1%（一票否决指标）

### 辅助指标
- **准确率**: 整体准确率
- **精确率**: 预测为正的样本中真正为正的比例
- **召回率**: 实际为正的样本中被正确预测的比例
- **ROC-AUC**: ≥0.95（模型区分能力）

## 技术支持

如有问题，请：
1. 查看README.md详细文档
2. 运行test_system.py诊断问题
3. 查看examples.py学习用法
4. 提交Issue或联系开发团队

## 下一步

1. 准备训练数据（JSON格式）
2. 运行交互模式开始使用
3. 根据需要调整配置参数
4. 评估模型性能
5. 部署到实际应用场景

祝您使用愉快！