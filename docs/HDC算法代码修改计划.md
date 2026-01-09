# HDC算法代码修改计划

基于评审结果，我制定了以下详细的代码修改计划：

## 第一阶段：基础架构搭建（高优先级）

### 1. 创建数据目录结构
```bash
# 创建训练和测试数据目录
New-Item -ItemType Directory -Path "d:\HDC\data\train"
New-Item -ItemType Directory -Path "d:\HDC\data\test"
```

### 2. 完善HDC.py的数据预处理模块
**修改文件**：`d:\HDC\utils\HDC.py`

**新增功能**：
- 实现角度矫正计算函数
- 添加数学计算导入（math模块）
- 完善BScanPoint类的深度矫正功能

**具体修改**：
```python
import math  # 新增导入

def calculate_actual_depth(acoustic_range, horizontal_angle, pitch_angle):
    """计算实际深度值（经角度矫正）"""
    return acoustic_range * math.cos(math.radians(horizontal_angle)) * math.cos(math.radians(pitch_angle))

def calculate_encoding_offset(acoustic_range, horizontal_angle, pitch_angle):
    """计算编码偏差"""
    return acoustic_range * math.cos(math.radians(horizontal_angle)) * math.sin(math.radians(pitch_angle))
```

### 3. 实现通道配置管理
**新增类**：ChannelManager
**功能**：
- 从JSON数据解析通道配置
- 管理9个通道的角度参数和拼孔值
- 提供通道特定的数据处理方法

**实现代码**：
```python
class ChannelManager:
    """通道管理器，负责解析和管理9个通道的配置"""
    
    def __init__(self, config_data):
        self.channels = {}
        self.parse_config(config_data)
    
    def parse_config(self, config_data):
        """从JSON配置数据解析通道参数"""
        # 解析BeamAngle、BeamOffset等配置
        pass
    
    def get_channel_config(self, channel_id):
        """获取指定通道的配置"""
        return self.channels.get(channel_id)
```

## 第二阶段：核心算法实现（中优先级）

### 4. 实现超维编码核心算法
**新增类**：HyperdimensionalEncoder
**功能**：
- 连续映射编码（汉明距离均匀变化）
- 循环移位编码
- 余弦距离计算

**实现代码**：
```python
class HyperdimensionalEncoder:
    """超维编码器，负责将数值映射到高维空间"""
    
    def __init__(self, dimensions=10000):
        self.dimensions = dimensions
    
    def continuous_mapping(self, value, min_val, max_val):
        """连续映射数值到超维向量"""
        normalized = (value - min_val) / (max_val - min_val)
        # 实现汉明距离均匀变化的编码逻辑
        pass
    
    def cyclic_shift_encoding(self, points, base_vector):
        """循环移位编码表示多个点"""
        encoded_vectors = []
        for i, point in enumerate(points):
            # 对每个点进行循环移位
            shifted_vector = np.roll(base_vector, i)
            encoded_vectors.append(shifted_vector)
        return encoded_vectors
```

### 5. 实现训练样本提取逻辑
**新增功能**：
- 从标注JSON文件解析伤损区域
- 提取对应的B显数据点作为训练样本
- 考虑空间关联性（相近编码值）

**实现代码**：
```python
class TrainingSampleExtractor:
    """训练样本提取器"""
    
    def extract_samples(self, bscan_data, annotation_data):
        """从B显数据和标注数据中提取训练样本"""
        # 解析标注文件中的伤损区域坐标
        # 提取对应B显数据点
        # 返回伤损样本和正常样本
        pass
```

## 第三阶段：模型训练与推理（中优先级）

### 6. 实现HDC模型训练功能
**新增类**：HDCModel
**功能**：
- 训练21种分类（12种伤损 + 9种正常结构）
- 支持现场检测和回放分析两种模式
- 模型保存和加载

**实现代码**：
```python
class HDCModel:
    """HDC模型类，负责训练和推理"""
    
    def train(self, training_samples, labels):
        """训练HDC模型"""
        # 实现超维向量的训练逻辑
        # 构建分类器
        pass
    
    def predict(self, test_sample):
        """对测试样本进行预测"""
        # 返回21种分类的概率分布
        pass
```

### 7. 实现测试数据推理功能
**新增功能**：
- 加载训练好的HDC模型
- 对测试JSON数据进行预处理
- 执行推理并生成识别结果

### 8. 实现测试标注JSON生成
**新增功能**：
- 将推理结果转换为标注JSON格式
- 包含每个点的识别情况和置信度
- 输出格式与训练标注保持一致

## 第四阶段：测试与优化（低优先级）

### 9. 测试整个训练-推理流程
**测试内容**：
- 数据预处理是否正确
- 超维编码是否有效
- 模型训练是否收敛
- 推理结果是否合理

## 实施步骤

1. **立即执行**：创建数据目录（任务1）
2. **并行开发**：数据预处理 + 通道配置（任务2-3）
3. **顺序开发**：超维编码 → 样本提取 → 模型训练 → 推理功能（任务4-8）
4. **最终测试**：端到端流程验证（任务9）

## 预期成果

完成修改后，HDC算法将能够：
- ✅ 正确处理角度矫正和通道配置
- ✅ 实现有效的超维编码
- ✅ 从标注数据中提取训练样本
- ✅ 训练21种分类的HDC模型
- ✅ 对测试数据生成标注JSON文件
- ✅ 支持现场检测和回放分析两种模式

## 优先级说明

- **高优先级**：基础架构搭建，直接影响后续开发
- **中优先级**：核心算法实现，决定模型性能
- **低优先级**：测试优化，确保系统稳定性

这个计划将系统性地解决评审中发现的问题，确保HDC算法的完整实现。