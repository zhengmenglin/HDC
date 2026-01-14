# 实现同时显示两种标注并避免ID冲突

## 问题分析
1. 目前`main_window.py`中加载标注时默认使用`replace=True`，导致HDC标注会替换常规标注
2. `csv_to_hdc_json.py`中HDC标注ID固定从100开始，可能与常规标注ID冲突
3. `annotation_manager.py`的`load_annotations`方法在合并标注时会忽略ID重复的标注

## 解决方案

### 1. 修改main_window.py中的标注加载逻辑
- 加载常规标注时使用`replace=True`（默认值，保持不变）
- 加载HDC标注时使用`replace=False`，实现标注合并

### 2. 增强annotation_manager.py的ID冲突处理
- 修改`load_annotations`方法，当合并标注遇到ID冲突时，自动生成新的唯一ID
- 确保新生成的ID大于当前所有标注的最大ID

### 3. 优化csv_to_hdc_json.py的ID生成逻辑（可选）
- 考虑从更高的起始ID开始生成HDC标注ID，减少冲突概率

## 具体实现步骤

1. **修改main_window.py**：
   - 在第1068行加载HDC标注时添加`replace=False`参数

2. **修改annotation_manager.py**：
   - 修改`load_annotations`方法，当合并标注时，检查并处理ID冲突
   - 对于冲突的ID，生成新的唯一ID并更新标注

3. **测试验证**：
   - 确保同时加载常规标注和HDC标注时不会丢失数据
   - 确保所有标注ID唯一
   - 确保标注列表中能显示所有标注

## 预期效果
- 常规标注和HDC标注同时显示在GUI中
- 所有标注ID唯一，无冲突
- 标注列表中可选择所有标注

## 文件修改清单
1. `d:\HDC\gui\main_window.py`
2. `d:\HDC\gui\annotation_manager.py`
3. 可选：`d:\HDC\csv_to_hdc_json.py`