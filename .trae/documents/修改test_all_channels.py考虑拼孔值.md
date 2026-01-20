# 修改test_all_channels.py考虑拼孔值

## 目标
在 `test_all_channels.py` 中添加对拼孔值的支持，参考 `csv_to_hdc_json.py` 的实现。

## 主要修改内容

### 1. 添加拼孔值提取函数
从原始JSON文件中提取：
- `beam_offset_map`: 通道到拼孔值的映射
- `beam_way_map`: 通道到射向的映射（-1后、0直、1前）
- `beam_angle_map`: 通道到声束角度的映射

### 2. 修改CSV到HDC JSON转换调用
在调用 `CsvToHdcJsonConverter.batch_convert` 时：
- 传递拼孔值映射参数
- 确保聚类时考虑拼孔值

### 3. 添加坐标转换工具导入
导入 `coordinate_converter` 工具，用于计算显示坐标

## 修改后的文件结构
```
test_all_channels.py
├── 添加拼孔值提取函数
├── 修改CSV到HDC JSON转换调用（传递拼孔值）
└── 添加坐标转换工具导入
```

## 关键改进点
1. **拼孔值支持**：从原始JSON文件中提取拼孔值映射
2. **坐标计算**：使用拼孔值计算显示坐标
3. **聚类优化**：基于显示坐标进行聚类，考虑拼孔值
4. **完整流程**：确保从训练到预测都考虑拼孔值