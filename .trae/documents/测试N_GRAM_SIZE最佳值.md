# 测试N_GRAM_SIZE最佳值的方案

## 目标
创建一个测试脚本，循环测试N_GRAM_SIZE从1到32的所有值，找出最佳配置。

## 实现方案

### 1. 创建测试脚本 `test_ngram_search.py`
基于之前的测试脚本，添加N_GRAM_SIZE的循环：
- 固定窗口大小=256，滑动步长=8（基于之前测试的最佳配置）
- 循环测试N_GRAM_SIZE从1到32
- 记录每个N_GRAM_SIZE的F1分数、精确率、召回率
- 输出最佳N_GRAM_SIZE

### 2. 测试参数
- 窗口大小：256（固定）
- 滑动步长：8（固定）
- N_GRAM_SIZE：1, 2, 4, 8, 16, 32（测试所有值）

### 3. 输出结果
- 每个N_GRAM_SIZE的测试结果
- 按F1分数排序
- 推荐最佳N_GRAM_SIZE
- 保存结果到CSV文件

## 文件清单
- `test_ngram_search.py`: N_GRAM_SIZE搜索测试脚本
- `test_ngram_results.csv`: 测试结果汇总