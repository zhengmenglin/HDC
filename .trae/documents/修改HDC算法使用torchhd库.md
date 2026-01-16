# 修改HDC算法使用torchhd库

## 1. 分析当前实现
- 当前工程使用自定义的HDC实现，包括编码器、分类器和数据处理器
- 主要文件：`utils/HDC.py`、`hdc_train.py`、`hdc_test.py`
- 核心功能：训练HDC模型、使用滑动窗口扫描检测伤损

## 2. torchhd库优势
- 提供多种VSA张量类型（BSCTensor, MAPTensor, HRRTensor等）
- 实现了多种高效分类器（Vanilla, AdaptHD, OnlineHD, NeuralHD等）
- 丰富的功能函数（bind, bundle, permute, inverse等）
- 支持GPU加速，性能更优

## 3. 修改计划

### 3.1 修改`utils/HDC.py`
- 使用torchhd的编码器替换当前的`HyperdimensionalEncoder`类
- 使用torchhd的分类器替换当前的`HDCClassifier`类
- 保留`BScanDataProcessor`类的核心功能，适配新的HDC实现

### 3.2 修改`hdc_train.py`
- 适配新的HDC编码器和分类器
- 使用torchhd提供的训练方法
- 优化模型保存和加载机制

### 3.3 修改`hdc_test.py`
- 适配新的HDC模型格式
- 使用torchhd提供的预测方法
- 优化滑动窗口扫描算法

## 4. 具体实现步骤

1. **集成torchhd库**
   - 确保torchhd库已正确安装
   - 在代码中导入torchhd相关模块

2. **重构HDC编码器**
   - 使用torchhd的`Level`或`Projection`嵌入替换当前的连续映射
   - 使用torchhd的`hash_table`或`bind`/`bundle`操作替换当前的编码逻辑

3. **重构HDC分类器**
   - 使用torchhd的`Vanilla`或`OnlineHD`分类器替换当前的`HDCClassifier`
   - 实现基于torchhd的训练和预测方法

4. **优化滑动窗口扫描**
   - 使用torchhd的相似度计算方法
   - 优化伤损区域检测逻辑

5. **测试和验证**
   - 确保修改后的算法能够正确训练模型
   - 确保修改后的算法能够正确检测伤损
   - 比较修改前后的性能和准确性

## 5. 预期效果
- 提高训练速度和效率
- 提高伤损检测的准确性
- 代码更加简洁和可维护
- 支持GPU加速，进一步提升性能