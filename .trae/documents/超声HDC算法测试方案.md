# 超声HDC算法测试方案

## 问题分析
EMG手势识别成功的原因：
1. 数据已按手势段预先分割
2. 每个窗口内标签一致
3. 窗口标签 = 窗口内任意点的标签

超声探伤当前问题：
1. 一段数据中有多个不同的伤损类型
2. 滑动窗口内标签不一致
3. 只取中心点标签导致信息丢失

## 测试方案

### 步骤1：创建简化的测试数据
基于 `d:\HDC\data\train\接头3.cap_hdc.csv`：
- **简化标注规则**：channel_8 > 170 → 伤损(label=1)，否则正常(label=0)
- 按标签变化点分割数据（仿EMG方式）
- 确保每段内标签一致

### 步骤2：实现两种数据输入方式对比

**方式A：EMG风格（按段分割）**
```python
# 找到标签变化点
indexes = [i for i in range(1, len(labels)) if labels[i] != labels[i-1]]
# 每个连续段作为一个样本
for i in indexes:
    segment = data[prev:i]
    label = labels[prev]  # 段内所有点共享一个标签
```

**方式B：滑动窗口（当前方式）**
```python
# 固定窗口大小，滑动步长
for start in range(0, len(data) - window + 1, step):
    window_data = data[start:start+window]
    label = labels[(start + start+window)//2]  # 中心点标签
```

### 步骤3：测试参数组合
- 窗口大小：32, 64, 128, 256
- 滑动步长：1, 4, 8, 16
- N-gram大小：2, 4, 8

### 步骤4：评估指标
- 训练准确率
- 测试准确率
- 精确率（Precision）
- 召回率（Recall）
- F1分数

### 步骤5：创建测试脚本
创建 `test_hdc_simple.py`：
1. 加载接头3.cap_hdc.csv
2. 应用简化标注规则
3. 实现两种数据集类
4. 训练和测试HDC模型
5. 输出对比结果

## 预期结果
- 找到最适合超声数据的数据输入方式
- 确定最佳参数组合
- 验证HDC算法在超声伤损检测中的有效性

## 文件清单
- `test_hdc_simple.py`: 主测试脚本
- `test_results.csv`: 测试结果汇总