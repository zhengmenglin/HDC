## 修复数据转换逻辑

### 问题分析
当前代码在 [`convert_to_csv_format`](file:///d:\HDC\ultrasonic_data_converter.py#L202-290) 方法中存在以下问题：

1. **编码范围计算错误**（第226-228行）：
   - 当前使用原始E值范围：`min_e = sorted_es[0] + min_offset` 和 `max_e = sorted_es[-1] + max_offset`
   - 应该使用调整后的编码值范围：基于 `encoded_e = e - beam_offset[channel]` 的实际值

2. **数据行生成逻辑错误**（第253行）：
   - 当前使用原始E值范围遍历：`for encoding in range(min_e, max_e + 1)`
   - 应该使用调整后的编码值范围遍历

### 修复方案

**修改位置**：`ultrasonic_data_converter.py` 的 `convert_to_csv_format` 方法

**具体修改**：

1. **修复编码范围计算**（第226-228行）：
   - 遍历所有编码数据，找到实际的编码值最小值和最大值
   - 使用 `encoded_data` 字典的键来确定范围

2. **修复数据行生成**（第253行）：
   - 使用调整后的编码值范围：`sorted(encoded_data.keys())`
   - 确保遍历的是经过 `e - beam_offset[channel]` 计算后的编码值

这样就能确保CSV文件中的encoding列正确地反映了经过BeamOffset调整的编码值。