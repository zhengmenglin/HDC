## 实现计划：根据新方案编写1.json转1.csv转换函数

### 新方案分析
根据文档第50-54行，新方案的核心逻辑是：
1. 读取1.json文件，获取所有通道的编码值和声程值，生成基础1.csv
2. 将数据复制三份得到sheet1, sheet2, sheet3
3. 对sheet1和sheet2，将channel_0-8共9列数据分别上移对应的BeamOffset行
4. 空缺位置填充为0，E按照规律填充

### 实现步骤

1. **在 `ultrasonic_data_converter.py` 中添加新方法 `convert_to_csv_with_beam_offset`**
   - 读取JSON数据，提取BeamOffset配置
   - 生成基础DataFrame（包含encoding和channel_0-8列）
   - 复制三份数据（sheet1, sheet2, sheet3）
   - 对sheet1和sheet2应用BeamOffset上移操作
   - 合并所有sheet，保存为CSV

2. **BeamOffset上移逻辑**
   - 对每个通道，获取对应的BeamOffset值
   - 将该列数据向上移动指定行数
   - 顶部空缺位置填充0
   - E列（encoding）按规律填充（0, 1, 2, ...）

3. **添加命令行接口**
   - 支持指定输入JSON文件和输出CSV文件
   - 可选参数控制是否使用新方案

### 关键技术点
- 使用 `pandas.shift()` 或 `numpy.roll()` 实现数据上移
- BeamOffset格式：`[通道号, 偏移值, 通道号, 偏移值, ...]`
- 确保数据类型正确（encoding为整数，channel值为浮点数）