# 修改 `d:\HDC\HDC.py` 详细计划

## 目标
根据 `HDC算法需求.md` 修改 `HDC.py`，保留原有超维计算算法逻辑，不使用聚类和二值化，实现循环移位编码。

## 主要修改内容

### 1. 添加数据结构类

#### 1.1 BScanPoint类
```python
class BScanPoint:
    """B显数据点类，存储深度值、原始dB值、幅度值"""
    - depth: 深度值（mm）
    - db_value: 原始超声信号强度（dB）
    - amplitude: 信号幅度值（%）
```

#### 1.2 ChannelConfig类
```python
class ChannelConfig:
    """通道配置类，存储通道的参数信息"""
    - pitch_angle: 俯仰角度（度）
    - horizontal_angle: 水平角度（度）
    - offset: 拼孔值（mm）
```

#### 1.3 BScanMatrix类
```python
class BScanMatrix:
    """B显矩阵管理类"""
    - init_bscan_matrix(max_encoding): 初始化B显矩阵
    - update_bscan_matrix(channel, encoding, depth, db_value, amplitude): 更新矩阵（右到左填充）
    - calculate_x_coordinate(current_x, channel, encoding_value): 计算X坐标
    - calculate_y_coordinate(depth, y_max): 计算Y坐标
    - get_bscan_data(data_type): 获取B显数据
```

### 2. 增强HyperdimensionalEncoder类

#### 2.1 保留现有方法
- `generate_binary_hypervector()`: 生成随机二元超维向量
- `generate_address_vector(key, memory_type)`: 生成地址超维向量
- `continuous_mapping(min_val, max_val, num_levels, base_key)`: 连续映射编码
- `bind(address_vector, value_vector)`: 绑定运算
- `bundle(vectors, normalize)`: 捆绑运算
- `hamming_similarity(vec1, vec2)`: 汉明相似度
- `cosine_similarity(vec1, vec2)`: 余弦相似度

#### 2.2 新增方法
- `circular_shift_encode(depth_values)`: 循环移位编码，表示多个点的深度值序列
- `encode_sample_enhanced(channel, code, depth, amplitude)`: 增强样本编码，包含更多特征
- `encode_value_with_segments(value, value_range, num_segments)`: 分段编码
- `encode_feature_interaction(feature1, feature2)`: 特征交互编码

### 3. 改进HDCClassifier类

#### 3.1 保留现有方法
- `train(samples, labels)`: 训练分类器
- `predict(sample)`: 预测样本类别

#### 3.2 改进方法
- `retrain(samples, labels, iterations)`: 改进重训练逻辑
  - 使用非二元向量和自适应学习率
  - 避免类别混淆问题
  - 在迭代结束后统一归一化

#### 3.3 新增方法
- `determine_flaw_type(channel, depth, amplitude)`: 根据通道和深度判断伤损类型
  - 核伤类：通道1-6，深度>100，幅度>70
  - 水平裂纹类：通道7-8，深度>120，幅度>60
  - 轨底裂纹类：通道9，深度>150，幅度>50
  - 螺孔裂纹类：特定深度范围
  - 其他类：其他情况
- `determine_normal_structure(channel, depth, amplitude)`: 判断正常结构类型
  - 接头、螺孔、导线孔、焊缝、鱼鳞伤、轨底锈蚀、轨底焊筋、道岔、其他

### 4. 重写BScanDataProcessor类

#### 4.1 移除的功能
- 移除二值化处理（binary_value）
- 移除聚类相关功能

#### 4.2 新增/改进的功能
- `load_json_data(json_file)`: 从JSON文件加载数据，格式：{"B1":[1,125,80]}
- `create_bscan_matrix(max_encoding)`: 创建B显矩阵
- `update_matrix_with_data(data_points)`: 用数据点更新矩阵
- `calculate_coordinates(data_points)`: 计算所有数据点的坐标
- `extract_samples_from_matrix()`: 从矩阵提取样本
- `calculate_changes(data_points)`: 计算数据变化率

### 5. 更新main函数

#### 5.1 数据加载流程
1. 从JSON文件加载B显数据
2. 创建B显矩阵
3. 更新矩阵数据
4. 计算坐标和变化率
5. 提取训练/测试样本

#### 5.2 训练流程
1. 初始化编码器
2. 生成数值向量（深度、幅度、编码值）
3. 准备训练数据和标签
4. 训练分类器

#### 5.3 测试流程
1. 准备测试数据
2. 预测测试样本
3. 输出分类结果和相似度
4. 重训练优化（可选）

## 技术要点

### 1. 连续映射编码
- 确保汉明距离均匀变化
- 公式：`flip_bits = self.d // (2 * (num_levels - 1))`
- 用于编码深度值、幅度值等连续数值

### 2. 循环移位编码
- 用于表示多个点的深度值序列
- 通过循环移位操作将多个深度值编码为一个向量
- 保留序列的顺序关系

### 3. 坐标系统
- 原点：左上角
- X轴：向右，X = X + b + Δx
- Y轴：向下，Y = Ymax * d / 3

### 4. 矩阵填充
- 从右到左填充
- 起始编码值（0）放在最右边
- 新编码值放在左边

### 5. 数据格式
- JSON格式：{"B1":[1,125,80]}
- 字段含义：[通道号, 深度值, 幅度值]

### 6. 分类逻辑
- 根据通道号和深度范围自动判断伤损类型
- 根据特征判断正常结构类型
- 支持五大类伤损和九类正常结构

## 预期效果

1. 完整支持需求文档中的所有功能要求
2. 保留HDC.py的核心算法逻辑
3. 不使用聚类和二值化处理
4. 支持JSON格式数据读取
5. 实现循环移位编码
6. 实现完整的伤损和正常结构分类
7. 改进重训练逻辑，避免类别混淆