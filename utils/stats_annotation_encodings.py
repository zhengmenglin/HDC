
import json
import os
import glob
import pandas as pd
from config import config

def get_beam_offsets(data_file):
    """从原始JSON文件中提取拼孔值"""
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                config = data[0]
                if 'BeamOffset' in config and isinstance(config['BeamOffset'], list):
                    offsets = config['BeamOffset']
                    return {int(offsets[i]): offsets[i+1] for i in range(0, len(offsets), 2)}
    except Exception as e:
        print(f"提取拼孔值失败 {data_file}: {e}")
    return {}

def stats_annotations(data_dir):
    # 查找所有的标注.json文件
    pattern = os.path.join(data_dir, "**", "*标注.json")
    json_files = glob.glob(pattern, recursive=True)
    
    all_stats = []
    
    for json_file in json_files:
        # 寻找对应的原始数据文件
        data_file = json_file.replace("标注.json", ".json")
        if not os.path.exists(data_file):
            data_file = json_file.replace("标注.json", ".xxc")
            
        offsets = {}
        if os.path.exists(data_file):
            offsets = get_beam_offsets(data_file)
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            annotations = data.get('annotations', [])
            file_name = os.path.basename(json_file)
            
            for ann in annotations:
                ann_id = ann.get('id')
                ann_type = ann.get('type')
                subtype = ann.get('subtype', {}).get('name', '未知')
                start_e = ann.get('start_E')
                end_e = ann.get('end_E')
                channels = ann.get('channels', [])
                
                # 如果字段缺失，尝试从 points 中计算
                if start_e is None or end_e is None:
                    points = ann.get('points', [])
                    if points:
                        e_values = [p.get('e') for p in points if p.get('e') is not None]
                        if e_values:
                            start_e = min(e_values)
                            end_e = max(e_values)
                
                # 为每个通道计算拼孔后的位置
                channel_info = []
                points = ann.get('points', [])
                
                # 按通道分组点
                ch_points = {}
                for p in points:
                    ch = p.get('channel')
                    e = p.get('e')
                    if ch is not None and e is not None:
                        if ch not in ch_points:
                            ch_points[ch] = []
                        ch_points[ch].append(e)
                
                all_corrected_es = []
                
                # 如果没有点数据，退而求其次使用全局 start_e/end_e
                if not ch_points and start_e is not None and end_e is not None:
                    for ch in channels:
                        offset = offsets.get(ch, 0)
                        corrected_start = start_e - offset
                        corrected_end = end_e - offset
                        channel_info.append(f"Ch{ch}({corrected_start}-{corrected_end})")
                        all_corrected_es.extend([corrected_start, corrected_end])
                else:
                    # 使用每个通道实际点的范围
                    for ch, e_list in ch_points.items():
                        offset = offsets.get(ch, 0)
                        ch_min_e = min(e_list)
                        ch_max_e = max(e_list)
                        corrected_start = ch_min_e - offset
                        corrected_end = ch_max_e - offset
                        channel_info.append(f"Ch{ch}({corrected_start}-{corrected_end})")
                        all_corrected_es.extend([corrected_start, corrected_end])
                
                # 计算跨所有通道的总跨度
                all_ch_span = 0
                if all_corrected_es:
                    all_ch_span = max(all_corrected_es) - min(all_corrected_es) + 1
                
                if start_e is not None and end_e is not None:
                    span = end_e - start_e + 1
                    
                    all_stats.append({
                        '文件名': file_name,
                        '标注ID': ann_id,
                        '类型': ann_type,
                        '子类型': subtype,
                        '原始起始E': start_e,
                        '原始结束E': end_e,
                        '跨度(点数)': span,
                        '涉及通道': ",".join(map(str, channels)),
                        '拼孔后位置(CSV坐标)': " | ".join(channel_info),
                        '全通道总跨度(CSV行数)': all_ch_span
                    })
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            
    return all_stats

def main():
    # 从配置中获取数据目录
    base_dir = config.get("data.input_dir", "data")
    # 确保路径是绝对路径
    base_dir = os.path.abspath(base_dir)
    
    print(f"正在统计 {base_dir} 目录下的标注信息...")
    
    stats = stats_annotations(base_dir)
    
    if not stats:
        print("未发现任何标注信息。")
        return
        
    df = pd.DataFrame(stats)
    
    # 按照文件名和原始起始位置排序
    df = df.sort_values(by=['文件名', '原始起始E'])
    
    # 从配置中获取输出目录
    output_dir = config.get("data.output_dir", "output")
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到 Excel
    output_path = os.path.join(os.path.abspath(output_dir), "annotation_encoding_stats_v4.xlsx")
    try:
        df.to_excel(output_path, index=False)
        print(f"\n统计完成！共发现 {len(df)} 条标注记录。")
        print(f"结果已保存至: {output_path}")
        
        # 打印一些摘要信息
        print("\n--- 跨度统计摘要 ---")
        print(df['跨度(点数)'].describe())
    except Exception as e:
        print(f"保存 Excel 失败: {e}")
        csv_path = output_path.replace(".xlsx", ".csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存为CSV文件: {csv_path}")

if __name__ == "__main__":
    main()
