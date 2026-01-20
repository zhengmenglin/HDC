## 📚 核心模块概览
### 1. functional.py - 核心操作函数
这是最基础的模块，包含所有超维计算的核心操作：
 🔹 超向量生成函数
函数 功能 适用场景 
empty() 创建空集超向量（元素全为0） 初始化，作为捆绑操作的恒等元 
identity() 创建单位超向量（元素全为1） 作为绑定操作的恒等元 
random() 创建随机超向量 生成随机标识符、特征向量 
level() 创建层级相关的超向量 编码连续数值（如温度、价格） 
thermometer() 温度计编码 编码有序数值，保持相似性 
circular() 循环编码 编码周期性数据（如角度、时间）
 🔹 核心操作
函数 功能 适用场景 
bind() 绑定操作（关联信息） 键值对、特征组合 
bundle() 捆绑操作（聚合信息） 集合、特征融合 
permute() 置换操作（分配顺序） 序列编码、位置信息 
inverse() 绑定操作的逆 解绑、信息提取 
negative() 捆绑操作的逆 从集合中移除元素
 🔹 相似度计算
函数 功能 适用场景 
cosine_similarity() / cos() 余弦相似度 大多数值型超向量 
dot_similarity() / dot() 点积相似度 MAP、HRR 类型 
hamming_similarity() 汉明距离 二进制超向量（BSC）
 🔹 高级结构函数
函数 功能 适用场景 
multiset() 多重集 词袋模型、特征集合 bundle_sequence() 基于捆绑的序列 时间序列、文本序列 bind_sequence() 基于绑定的序列 需要精确顺序的序列 hash_table() 哈希表 键值存储、特征映射 graph() 图结构 社交网络、知识图谱 ngrams() n-gram 统计 文本特征提取
 🔹 其他工具
- normalize() - 归一化
- cleanup() - 清理操作（找到最相似的向量）
- resonator() - 共振网络（用于分解）
- ridge_regression() - 岭回归
### 2. embeddings.py - 编码层
将输入数据转换为超向量的神经网络层：

类 功能 适用场景 Empty 空向量嵌入 填充、占位符 Random 随机嵌入 离散特征、类别编码 Level 层级编码 连续数值特征 Thermometer 温度计编码 有序数值特征 Circular 循环编码 周期性特征（角度、时间） Projection 随机投影 高维特征降维 Sinusoid 正弦编码 非线性特征变换 Density 密度编码 IntRVFL 模型专用 FractionalPower 分数幂编码 相似性保持的核函数

### 3. classifiers.py - 分类器
完整的分类模型：

分类器 特点 适用场景 Vanilla 基线中心分类器 简单分类任务、快速原型 AdaptHD 自适应训练 需要迭代优化的任务 OnlineHD 在线学习 流式数据、增量学习 NeuralHD 神经网络风格 边缘计算、资源受限 DistHD 动态编码 需要特征选择的任务 LeHDC 梯度学习 复杂分类任务 CompHD 模型压缩 存储受限场景 SparseHD 稀疏化 计算效率优先 QuantHD 量化 低精度计算 IntRVFL 随机向量函数链接 快速训练、岭回归

### 4. memory.py - 记忆模型
类/函数 功能 适用场景 SparseDistributed 稀疏分布记忆 大规模关联记忆 hopfield() 经典 Hopfield 网络 模式恢复、联想记忆 Hopfield Hopfield 类封装 面向对象使用 modern_hopfield() 现代 Hopfield 网络 密集联想记忆 attention() 注意力机制 Transformer 风格

### 5. structures.py - 数据结构
类 功能 适用场景 Memory 关联记忆 键值存储、近似查找 Multiset 多重集 词袋、特征集合 HashTable 哈希表 特征映射、属性存储 BundleSequence 捆绑序列 时间序列、文本 BindSequence 绑定序列 需要精确顺序 Graph 图结构 社交网络、知识图谱 Tree 树结构 层次数据、决策树 FiniteStateAutomata 有限状态自动机 序列建模、模式识别

### 6. models.py - 基础模型
类 功能 适用场景 Centroid 中心分类模型 类原型、最近邻 IntRVFL IntRVFL 模型 随机特征网络

## 🎯 如何选择适合你项目的函数？
### 根据任务类型选择： 1. 分类任务
- 简单分类 → Vanilla 或 AdaptHD
- 在线学习 → OnlineHD
- 边缘计算 → NeuralHD
- 高精度需求 → LeHDC
- 资源受限 → CompHD 、 SparseHD 、 QuantHD 2. 序列数据
- 文本/时间序列 → bundle_sequence() 、 bind_sequence()
- n-gram 特征 → ngrams()
- 序列建模 → BundleSequence 、 BindSequence 、 FiniteStateAutomata 3. 图/网络数据
- 社交网络 → graph() 、 Graph
- 知识图谱 → hash_table() 、 HashTable 4. 数值特征编码
- 连续数值 → level() 、 Level
- 有序数值 → thermometer() 、 Thermometer
- 周期性数据 → circular() 、 Circular 5. 记忆/联想
- 模式恢复 → hopfield() 、 modern_hopfield()
- 大规模记忆 → SparseDistributed
- 注意力机制 → attention() 6. 特征嵌入
- 离散特征 → Random
- 高维特征 → Projection
- 非线性变换 → Sinusoid 、 FractionalPower
## 💡 推荐组合
### 场景1：文本分类
```
# 使用 OnlineHD + Level 编码
classifier = OnlineHD(n_features, 
n_dimensions, n_classes)
encoder = Level(n_levels, 
n_dimensions)
```
### 场景2：时间序列预测
```
# 使用 bundle_sequence + thermometer
sequence = bundle_sequence
(thermometer_encoded_data)
```
### 场景3：图神经网络
```
# 使用 graph() + attention
graph_hv = graph(edges_hv)
output = attention(query, keys, 
values)
```
### 场景4：在线学习系统
```
# 使用 OnlineHD + Sinusoid 编码
classifier = OnlineHD(n_features, 
n_dimensions, n_classes)
encoder = Sinusoid(n_features, 
n_dimensions)
```