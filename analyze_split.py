#!/usr/bin/env python3
"""分析数据划分问题"""

import csv

# 读取数据
data = []
with open(r"d:\HDC\data\train\接头3.cap_hdc.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        channel_8 = float(row['channel_8'])
        label = 1 if channel_8 > 170 else 0
        data.append({
            'encoding': int(row['encoding']),
            'channel_8': channel_8,
            'label': label
        })

print("=" * 60)
print("数据划分分析")
print("=" * 60)
print(f"总数据点数: {len(data)}")
print()

# 划分训练集和测试集
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")
print()

# 统计标签分布
train_labels = [item['label'] for item in train_data]
test_labels = [item['label'] for item in test_data]

print("训练集标签分布:")
print(f"  正常点 (label=0): {train_labels.count(0)} ({train_labels.count(0)/len(train_labels)*100:.2f}%)")
print(f"  伤损点 (label=1): {train_labels.count(1)} ({train_labels.count(1)/len(train_labels)*100:.2f}%)")
print()

print("测试集标签分布:")
print(f"  正常点 (label=0): {test_labels.count(0)} ({test_labels.count(0)/len(test_labels)*100:.2f}%)")
print(f"  伤损点 (label=1): {test_labels.count(1)} ({test_labels.count(1)/len(test_labels)*100:.2f}%)")
print()

# 找出伤损点的位置
defect_positions = [i for i, item in enumerate(data) if item['label'] == 1]
print(f"伤损点位置（前20个）: {defect_positions[:20]}")
print(f"伤损点位置（后20个）: {defect_positions[-20:]}")
print()

# 分析伤损点分布
if defect_positions:
    print(f"伤损点范围: {min(defect_positions)} - {max(defect_positions)}")
    print(f"训练集范围: 0 - {train_size}")
    print(f"测试集范围: {train_size} - {len(data)}")
    print()
    
    # 检查测试集中是否有伤损点
    test_defects = [pos for pos in defect_positions if pos >= train_size]
    print(f"测试集中的伤损点数: {len(test_defects)}")
    if len(test_defects) == 0:
        print("⚠️  警告：测试集中没有伤损点！")
        print("   这导致模型无法正确评估伤损检测能力。")
