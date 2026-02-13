import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def read_config_info():
    """读取config.py文件中的配置信息"""
    epochs = ""
    train_strategy = ""
    channels = []
    
    try:
        with open('config.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        epochs_match = re.search(r'"epochs"\s*:\s*(\d+)', content)
        if epochs_match:
            epochs = epochs_match.group(1)
        
        strategy_match = re.search(r'"train_strategy"\s*:\s*"([^"]+)"', content)
        if strategy_match:
            train_strategy = strategy_match.group(1)
        
        channels_match = re.search(r'"channels"\s*:\s*\[([^\]]+)\]', content)
        if channels_match:
            channels_str = channels_match.group(1)
            channels = [int(x.strip()) for x in channels_str.split(',') if x.strip().isdigit()]
    except Exception as e:
        print(f"读取配置文件失败: {e}")
    
    return epochs, train_strategy, channels

def find_predicted_label_files():
    """查找train和test目录下的predicted_label.csv文件"""
    files = []
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        for file in os.listdir(train_dir):
            if 'predicted_label.csv' in file:
                files.append(os.path.join(train_dir, file))
    test_dir = 'data/test'
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if 'predicted_label.csv' in file:
                files.append(os.path.join(test_dir, file))
    return files

def plot_similarity_chart(file_path, epochs, train_strategy, channel_mode, save_dir):
    """为单个文件绘制相似度图表"""
    try:
        df = pd.read_csv(file_path)
        
        encoding = df['encoding']
        class_0_similarity = df['class_0_similarity']
        class_13_similarity = df['class_13_similarity']
        
        plt.figure(figsize=(12, 6))
        
        plt.scatter(encoding, class_0_similarity, color='blue', label='class_0_similarity', alpha=0.6)
        plt.scatter(encoding, class_13_similarity, color='orange', label='class_13_similarity', alpha=0.6)
        
        title = f"{train_strategy}+{epochs}轮+{channel_mode}"
        plt.title(title)
        plt.xlabel('Encoding')
        plt.ylabel('Similarity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        file_name = os.path.basename(file_path)
        chart_name = file_name.replace('.csv', '.png')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        full_save_path = os.path.join(save_dir, chart_name)
        plt.savefig(full_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"图表保存成功: {full_save_path}")
        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    print("开始绘制相似度图表...")
    
    epochs, train_strategy, channels = read_config_info()
    
    if len(channels) == 1:
        channel_mode = f"单通道{channels[0]}"
    else:
        channel_mode = f"{len(channels)}通道"
    
    print(f"配置信息: 训练轮数={epochs}, 训练策略={train_strategy}, 通道模式={channel_mode}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('hdc_test_results', 'similarity_charts', timestamp)
    
    files = find_predicted_label_files()
    print(f"找到 {len(files)} 个predicted_label.csv文件")
    
    success_count = 0
    for file_path in files:
        print(f"处理文件: {file_path}")
        if plot_similarity_chart(file_path, epochs, train_strategy, channel_mode, save_dir):
            success_count += 1
    
    report_path = os.path.join(save_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"相似度图表报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"训练策略: {train_strategy}\n")
        f.write(f"通道模式: {channel_mode}\n")
        f.write(f"处理文件数: {len(files)}\n")
        f.write(f"成功绘制: {success_count}\n")
    
    print(f"处理完成: 成功绘制 {success_count} 个图表")
    print(f"结果保存至: {save_dir}")

if __name__ == "__main__":
    main()