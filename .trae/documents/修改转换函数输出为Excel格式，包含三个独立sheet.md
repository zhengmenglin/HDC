# 修改超声波数据转换器的计划

## 需求分析
1. **不需要三个sheet**：当前`convert_to_csv_with_beam_offset`生成三个sheet（sheet1, sheet2, sheet3），需要改为只生成一份
2. **统一转换方式**：将`convert_single_file`的数据转换替换为新的方式（使用`convert_to_csv_with_beam_offset`的逻辑）
3. **删除旧函数**：删除旧的`convert_to_csv_format`函数
4. **标注文件处理**：确保标注文件也能正确使用相同的转换方式

## 修改步骤

### 1. 修改`convert_to_csv_with_beam_offset`函数
- 移除三个sheet的生成逻辑（sheet1, sheet2, sheet3）
- 只生成一份应用BeamOffset调整的数据
- 保留BeamOffset调整的核心逻辑
- 输出格式：encoding + channel_0-8

### 2. 重构`convert_single_file`函数
- 替换当前的`convert_to_csv_format`调用
- 使用`convert_to_csv_with_beam_offset`的逻辑
- 保留标注文件的处理功能
- 支持标注数据与数据文件的转换

### 3. 删除`convert_to_csv_format`函数
- 移除旧的转换函数（第202-284行）
- 该函数已被新的转换方式取代

### 4. 更新相关调用
- 检查`batch_convert`函数是否需要调整
- 确保所有调用都使用新的转换方式

## 预期效果
- 只有一个转换函数，逻辑清晰
- 输出格式统一：encoding + channel_0-8
- 标注文件和数据文件使用相同的转换方式
- 代码更简洁，易于维护