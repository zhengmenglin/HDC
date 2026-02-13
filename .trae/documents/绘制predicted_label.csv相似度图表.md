# 绘制predicted_label.csv相似度图表

## 任务描述
编写一个Python脚本，找到train和test目录下所有的predicted_label.csv文件，为每个文件绘制相似度图表。

## 实现步骤

### 1. 导入必要的库
- os：用于文件路径操作
- pandas：用于读取和处理CSV数据
- matplotlib：用于绘制图表

### 2. 读取配置信息
- 从config.py文件中读取第46行的训练轮数和第76行的训练策略
- 这些信息将用于图表标题

### 3. 查找predicted_label.csv文件
- 遍历train和test目录
- 找到所有名为`*_predicted_label.csv`的文件

### 4. 处理每个文件
- 读取CSV文件数据
- 提取encoding、class_0_similarity和class_13_similarity列
- 绘制散点图，使用encoding作为横坐标，两个相似度值作为纵坐标
- 设置图表标题，包含配置信息
- 保存图表，使用原文件名（去掉.csv后缀）作为图片名

### 5. 保存图表
- 将生成的图表保存到适当的目录

## 技术细节
- 使用pandas读取CSV文件，确保正确处理数据类型
- 使用matplotlib绘制散点图，设置合适的图表大小和样式
- 为每个相似度值使用不同的颜色和图例
- 确保图表标题包含从config.py读取的训练轮数和策略信息

## 预期输出
- 每个predicted_label.csv文件对应一个PNG格式的图表文件
- 图表显示encoding与两个相似度值的关系
- 图表标题包含训练配置信息