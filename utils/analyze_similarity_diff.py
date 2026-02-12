
import sys
from pathlib import Path
import pandas as pd
import os
import glob

# 获取项目根目录并添加到搜索路径
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from config import config

def analyze_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    results = []
    num_channels = 9
    
    for i in range(num_channels):
        sim_col = f'similarity_{i}'
        if sim_col not in df.columns:
            continue
            
        # 提取 encoding 和 label 用于显示
        encodings = df['encoding'] if 'encoding' in df.columns else range(len(df))
        
        for idx, row in df.iterrows():
            sim_str = str(row[sim_col])
            if ',' not in sim_str:
                continue
                
            try:
                normal_sim, defect_sim = map(float, sim_str.split(','))
                if defect_sim > normal_sim:
                    diff = defect_sim - normal_sim
                    results.append({
                        'file': os.path.basename(file_path),
                        'channel': i,
                        'index': idx,
                        'encoding': encodings[idx],
                        'normal_sim': normal_sim,
                        'defect_sim': defect_sim,
                        'diff': diff
                    })
            except ValueError:
                continue
                
    return results

def main():
    # 从配置中获取数据目录
    data_dir = config.get("data.input_dir", "data")
    # 确保路径是绝对路径
    data_dir = os.path.abspath(data_dir)
    
    # 查找所有预测结果文件
    pattern = os.path.join(data_dir, "**", "*_predicted_label.csv")
    files = glob.glob(pattern, recursive=True)
    
    all_results = []
    for f in files:
        print(f"Processing {f}...")
        all_results.extend(analyze_csv(f))
        
    if not all_results:
        print("未发现伤损相似度大于正常相似度的数据点。")
        return
        
    # 转换为 DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 排序：差值从大到小
    results_df = results_df.sort_values(by='diff', ascending=False)
    
    # 从配置中获取输出目录
    output_dir = config.get("data.output_dir", "output")
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为 Excel
    output_path = os.path.join(os.path.abspath(output_dir), "similarity_analysis_results.xlsx")
    try:
        # 使用 openpyxl 引擎保存为 Excel
        results_df.to_excel(output_path, index=False)
        print(f"\n分析完成！结果已保存至: {output_path}")
    except Exception as e:
        print(f"\n保存 Excel 失败: {e}")
        print("尝试保存为 CSV 备份...")
        results_df.to_csv(output_path.replace(".xlsx", ".csv"), index=False)

    # 终端预览输出
    print("\n" + "="*80)
    print(f"{'文件':<30} {'通道':<5} {'索引':<8} {'编码':<10} {'正常':<8} {'伤损':<8} {'差值':<8}")
    print("-" * 80)
    
    display_limit = 20
    for _, res in results_df.head(display_limit).iterrows():
        print(f"{res['file']:<30} {res['channel']:<5} {res['index']:<8} {res['encoding']:<10} {res['normal_sim']:<8.4f} {res['defect_sim']:<8.4f} {res['diff']:<8.4f}")
    
    if len(results_df) > display_limit:
        print(f"... 还有 {len(results_df) - display_limit} 条结果已保存至 Excel。")
    print("="*80)

if __name__ == "__main__":
    main()
