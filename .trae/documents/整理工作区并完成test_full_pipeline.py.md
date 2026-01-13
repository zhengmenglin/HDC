# 整理工作区并完成test_full_pipeline.py

## 1. 整理工作区 - 删除过时测试程序

### 需要删除的过时测试文件
- `hdc_test.py`
- `hdc_train.py`
- `main.py`
- `run_pipeline.py`
- `run_test.py`
- `run_test_capture.py`
- `test_conversion.csv`
- `test_conversion.py`
- `test_error.txt`
- `test_file_find.py`
- `test_hdc_modifications.py`
- `test_load_data.py`
- `test_output.log`
- `test_output.txt`
- `test_output_captured.txt`
- `test_output_full.log`
- `test_results.log`
- `test_results.txt`
- `test_torchhd_import.py`
- `pipeline_test.csv` (如果存在)

## 2. 完成test_full_pipeline.py的正式版本

### 修改内容
1. **使用正式数据**：
   - 从 `data/train` 目录加载多个训练文件
   - 从 `data/test` 目录加载测试文件

2. **完整测试流程**：
   - 数据转换：批量转换训练和测试数据
   - 模型训练：使用转换后的训练数据训练模型
   - 模型预测：使用训练好的模型对测试数据进行预测
   - 结果评估：计算预测准确率和其他指标
   - 结果转换：将预测结果转换为JSON格式

3. **优化测试逻辑**：
   - 添加更详细的日志记录
   - 实现完整的训练-测试-评估流程
   - 使用torchhd库的最佳实践
   - 确保测试结果的准确性和可靠性

4. **清理机制**：
   - 自动清理测试过程中生成的临时文件
   - 保留最终的模型文件和预测结果

### 预期输出
- 训练好的模型文件
- 预测结果文件（CSV和JSON格式）
- 详细的测试日志
- 预测准确率和其他评估指标

## 3. 执行步骤

1. 首先删除过时的测试程序
2. 修改test_full_pipeline.py文件
3. 运行修改后的测试程序
4. 验证测试结果
5. 清理临时文件

通过以上步骤，我们将得到一个完整的、使用正式数据的测试管道，用于验证HDC算法在超声波探伤中的性能。