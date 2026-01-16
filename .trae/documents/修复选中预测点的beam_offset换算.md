## 修复选中预测点的beam_offset换算问题

### 问题分析

1. **CSV文件中的 `encoding` 值**：经过拼孔处理（原始E值 - beam_offset）
2. **displayed_points 中的 `e` 值**：原始E值（未经过拼孔处理）
3. **csv_to_hdc_json.py 中的处理**：生成HDC JSON时进行逆拼孔处理
   ```python
   # 逆拼孔处理：原始E值 = CSV编码值 + 对应通道的拼孔值
   original_e = encoding + offset
   ```
4. **当前问题**：`select_hdc_predicted_points()` 直接使用CSV中的 `encoding` 值与 `displayed_points` 匹配，没有进行逆拼孔换算

### 修改方案

在 `select_hdc_predicted_points()` 方法中，获取通道8的拼孔值，对CSV中的 `encoding` 值进行逆拼孔处理：

```python
def select_hdc_predicted_points(self):
    """选中HDC预测的点（直接从CSV文件中选择predicted_label_8 != 0的点）"""
    # 获取当前打开的CSV文件路径
    if self.parser is None or not hasattr(self.parser, 'json_file'):
        QMessageBox.warning(self, "警告", "请先打开一个JSON文件")
        return
    
    # 构造CSV文件路径（与JSON文件同目录，文件名相同，后缀为_hdc.csv）
    json_file_path = Path(self.parser.json_file)
    csv_file_path = json_file_path.parent / f"{json_file_path.stem}_hdc.csv"
    
    if not csv_file_path.exists():
        QMessageBox.warning(self, "警告", f"找不到CSV文件: {csv_file_path}")
        return
    
    print(f"调试：从CSV文件读取预测点: {csv_file_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        QMessageBox.critical(self, "错误", f"读取CSV文件失败: {str(e)}")
        return
    
    # 检查是否有predicted_label_8列
    if 'predicted_label_8' not in df.columns:
        QMessageBox.warning(self, "警告", "CSV文件中没有predicted_label_8列")
        return
    
    # 找到所有predicted_label_8 != 0的点
    pred_mask = df['predicted_label_8'] != 0
    pred_points_df = df[pred_mask]
    
    print(f"调试：CSV中预测点数: {len(pred_points_df)}")
    
    # 获取通道8的拼孔值
    beam_offset = self.parser.params['beam_offset'].get(8, 0)
    print(f"调试：通道8的拼孔值: {beam_offset}")
    
    # 在displayed_points中查找匹配的点
    selected_points = []
    for _, pred_row in pred_points_df.iterrows():
        # 逆拼孔处理：原始E值 = CSV编码值 + 拼孔值
        pred_e = int(pred_row['encoding']) + beam_offset
        pred_channel = 8  # 固定通道8
        pred_depth = float(pred_row['channel_8'])
        pred_amplitude = 100.0  # 默认幅度值
        
        # 在displayed_points中查找匹配的点（只匹配E值和channel，不检查depth和amplitude）
        for displayed_point in self.bscan_canvas.displayed_points:
            orig = displayed_point['original']
            if orig['e'] == pred_e and orig['channel'] == pred_channel:
                selected_points.append(displayed_point)
                print(f"调试： ✓ 找到匹配点 E:{pred_e} Ch:{pred_channel}")
                break
    
    print(f"调试：总共选中 {len(selected_points)} 个点")
    
    if selected_points:
        # 设置选中的点
        self.point_selector.set_selected_points(selected_points)
        self.status_bar.showMessage(f"已选中 {len(selected_points)} 个HDC预测点")
    else:
        QMessageBox.warning(self, "警告", "无法找到HDC预测的点")
```

### 主要修改

1. **获取通道8的拼孔值**：
   ```python
   beam_offset = self.parser.params['beam_offset'].get(8, 0)
   ```

2. **对CSV中的encoding值进行逆拼孔处理**：
   ```python
   # 逆拼孔处理：原始E值 = CSV编码值 + 拼孔值
   pred_e = int(pred_row['encoding']) + beam_offset
   ```

3. **添加调试信息**：
   ```python
   print(f"调试：通道8的拼孔值: {beam_offset}")
   ```

### 预期效果

- CSV中的 `encoding` 值会加上通道8的 `beam_offset`
- 转换后的 `pred_e` 值可以与 `displayed_points` 中的原始E值正确匹配
- 选中预测点功能可以正常工作