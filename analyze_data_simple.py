#!/usr/bin/env python3
"""分析接头3.cap_hdc.csv的数据分布（不使用pandas）"""

import csv
from collections import Counter
import statistics

# 读取数据
data = []
with open(r"d:\HDC\data\train\接头3.cap_hdc.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print("=" * 60)
print("数据基本信息")
print("=" * 60)
print(f"总行数: {len(data)}")
print(f"列名: {list(data[0].keys())}")
print()

print("=" * 60)
print("channel_8 数据分布")
print("=" * 60)
channel_8 = [float(row['channel_8']) for row in data]
print(f"最小值: {min(channel_8)}")
print(f"最大值: {max(channel_8)}")
print(f"平均值: {statistics.mean(channel_8):.2f}")
print(f"中位数: {statistics.median(channel_8):.2f}")
print(f"标准差: {statistics.stdev(channel_8):.2f}")
print()

# 统计不同值的出现次数
value_counts = Counter(channel_8)
print("值分布（前20个）:")
for value, count in sorted(value_counts.items())[:20]:
    print(f"  {value}: {count}")
print()

print("=" * 60)
print("label_8 数据分布")
print("=" * 60)
if 'label_8' in data[0]:
    label_8 = [int(row['label_8']) for row in data]
    unique_labels = sorted(set(label_8))
    print(f"唯一标签数: {len(unique_labels)}")
    print(f"标签值: {unique_labels}")
    print()
    
    # 统计每个标签的数量
    label_counts = Counter(label_8)
    print("标签分布:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        print(f"  标签 {label}: {count} 个点 ({count/len(label_8)*100:.2f}%)")
    print()
    
    # 找出标签变化点
    label_changes = [i for i in range(1, len(label_8)) if label_8[i] != label_8[i-1]]
    print(f"标签变化点数量: {len(label_changes)}")
    print(f"标签变化点位置（前20个）: {label_changes[:20]}")
    print()
    
    # 分析每个标签段的长度
    if len(label_changes) > 0:
        segment_lengths = []
        prev = 0
        for change in label_changes:
            segment_lengths.append(change - prev)
            prev = change
        segment_lengths.append(len(label_8) - prev)
        
        print(f"标签段长度统计:")
        print(f"  最小段长度: {min(segment_lengths)}")
        print(f"  最大段长度: {max(segment_lengths)}")
        print(f"  平均段长度: {statistics.mean(segment_lengths):.2f}")
        print(f"  中位数段长度: {statistics.median(segment_lengths):.2f}")
        print()
        
        # 统计不同长度的段
        length_counts = Counter(segment_lengths)
        print("段长度分布（前20个）:")
        for length in sorted(length_counts.keys())[:20]:
            print(f"  长度 {length}: {length_counts[length]} 个段")

print("=" * 60)
print("简化标注规则测试")
print("=" * 60)
# 应用简化标注规则：channel_8 > 170 → 伤损(label=1)，否则正常(label=0)
simple_labels = [1 if v > 170 else 0 for v in channel_8]
print(f"正常点数 (label=0): {simple_labels.count(0)} ({simple_labels.count(0)/len(simple_labels)*100:.2f}%)")
print(f"伤损点数 (label=1): {simple_labels.count(1)} ({simple_labels.count(1)/len(simple_labels)*100:.2f}%)")
print()

# 找出简化标签的变化点
simple_changes = [i for i in range(1, len(simple_labels)) if simple_labels[i] != simple_labels[i-1]]
print(f"简化标签变化点数量: {len(simple_changes)}")
print(f"简化标签变化点位置（前20个）: {simple_changes[:20]}")
print()

# 分析简化标签段的长度
if len(simple_changes) > 0:
    simple_segment_lengths = []
    prev = 0
    for change in simple_changes:
        simple_segment_lengths.append(change - prev)
        prev = change
    simple_segment_lengths.append(len(simple_labels) - prev)
    
    print(f"简化标签段长度统计:")
    print(f"  最小段长度: {min(simple_segment_lengths)}")
    print(f"  最大段长度: {max(simple_segment_lengths)}")
    print(f"  平均段长度: {statistics.mean(simple_segment_lengths):.2f}")
    print(f"  中位数段长度: {statistics.median(simple_segment_lengths):.2f}")
    print()
    
    # 统计不同长度的段
    simple_length_counts = Counter(simple_segment_lengths)
    print("简化标签段长度分布（前20个）:")
    for length in sorted(simple_length_counts.keys())[:20]:
        print(f"  长度 {length}: {simple_length_counts[length]} 个段")
