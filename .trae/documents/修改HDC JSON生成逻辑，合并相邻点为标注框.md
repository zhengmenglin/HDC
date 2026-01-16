## 修改HDC JSON生成逻辑，合并相邻点为标注框

### 问题分析

当前 `generate_hdc_json_direct()` 方法：
- 每个预测点生成一个单独的标注
- points数组只有一个点
- start_E和end_E都是同一个E值
- 画框时就是一个点，不是框

手动标注格式：
- 每个标注有多个连续的点
- points数组包含多个点
- start_E和end_E是第一个和最后一个点的E值
- 画框时连接这些点形成框

### 修改方案

#### 1. 修改 `generate_hdc_json_direct()` 方法

将单独的点合并成标注框：

```python
def generate_hdc_json_direct(self, defect_points: pd.DataFrame, original_csv_path: str, target_channel: int = None) -> Dict:
    """
    直接生成hdc.json格式的数据，合并相邻点为标注框
    
    Args:
        defect_points: 缺陷点DataFrame
        original_csv_path: 原始CSV文件路径
        target_channel: 目标通道号，只提取该通道的预测结果（如8表示只提取通道8）
        
    Returns:
        hdc.json格式的字典
    """
    annotations = []
    annotation_id = 1000  # 自动标注ID从1000开始
    
    # 检查是否有多通道标签
    has_multichannel_pred = all([f'predicted_label_{i}' in defect_points.columns for i in range(9)])
    has_multichannel_true = all([f'label_{i}' in defect_points.columns for i in range(9)])
    
    # 从原始JSON文件中提取超声波参数
    csv_path = Path(original_csv_path)
    original_stem = csv_path.stem.replace('_hdc', '')
    original_json_path = str(csv_path.parent / f"{original_stem}.json")
    beam_offset_map, beam_way_map, beam_angle_map = self.extract_beam_parameters(original_json_path)
    print(f"Using beam offset map: {beam_offset_map} for {original_csv_path}")
    
    # 按通道分组并合并相邻点
    if has_multichannel_pred or has_multichannel_true:
        # 确定要处理的通道范围
        channels_to_process = [target_channel] if target_channel is not None else range(9)
        
        for channel in channels_to_process:
            # 筛选当前通道的预测点
            channel_mask = defect_points[f'predicted_label_{channel}'] != 0
            channel_points = defect_points[channel_mask]
            
            if channel_points.empty:
                continue
            
            # 按E值排序
            channel_points = channel_points.sort_values(by='encoding')
            
            # 合并相邻点为标注框
            current_annotation_points = []
            current_label = None
            
            for _, row in channel_points.iterrows():
                encoding = int(row['encoding'])
                label = row[f'predicted_label_{channel}'] if has_multichannel_pred else row[f'label_{channel}']
                depth = row[f'channel_{channel}']
                
                # 获取该通道的拼孔值
                offset = beam_offset_map.get(channel, 0)
                # 逆拼孔处理：原始E值 = CSV编码值 + 对应通道的拼孔值
                original_e = encoding + offset
                
                # 生成点
                point = {
                    "e": original_e,
                    "channel": channel,
                    "depth": float(depth),
                    "amplitude": 100.0
                }
                
                # 如果当前点与上一个点的E值差小于2，并且标签相同，合并到当前标注
                if (current_annotation_points and 
                    current_label is not None and 
                    label == current_label and
                    abs(original_e - current_annotation_points[-1]['e']) < 2):
                    current_annotation_points.append(point)
                else:
                    # 保存上一个标注
                    if current_annotation_points:
                        annotation = self._create_annotation(
                            current_annotation_points, 
                            current_label, 
                            annotation_id, 
                            channel
                        )
                        annotations.append(annotation)
                        annotation_id += 1
                    
                    # 开始新的标注
                    current_annotation_points = [point]
                    current_label = label
            
            # 保存最后一个标注
            if current_annotation_points:
                annotation = self._create_annotation(
                    current_annotation_points, 
                    current_label, 
                    annotation_id, 
                    channel
                )
                annotations.append(annotation)
    
    return {
        "annotations": annotations,
        "timestamp": datetime.now().isoformat()
    }

def _create_annotation(self, points, label, annotation_id, channel):
    """创建标注对象"""
    if len(points) == 0:
        return None
    
    # 确定伤损类型
    if label == 0:
        annotation_type = "结构"
    else:
        annotation_type = "伤损"
    
    # 计算start_E和end_E
    start_E = min(p['e'] for p in points)
    end_E = max(p['e'] for p in points)
    
    return {
        "id": str(annotation_id),
        "type": annotation_type,
        "subtype": {
            "id": str(label),
            "name": self.flaw_types.get(label, "未知类型")
        },
        "points": points,
        "start_E": start_E,
        "end_E": end_E,
        "channels": [channel],
        "timestamp": datetime.now().isoformat(),
        "source": "predicted"
    }
```

### 主要改进

1. **按通道分组**：将预测点按通道分组
2. **按E值排序**：在每个通道内按E值排序
3. **合并相邻点**：
   - 如果当前点与上一个点的E值差 < 2，并且标签相同，合并到当前标注
   - 否则创建新的标注
4. **生成标注框**：
   - start_E = 第一个点的E值
   - end_E = 最后一个点的E值
   - points = 所有合并的点

### 预期效果

- HDC预测的标注格式与手动标注一致
- 可以像手动标注那样画框
- 多个相邻的点合并成一个标注框