
import json
import os
import random
from pathlib import Path

def create_sample_data(json_path, anno_path, n_samples=500):
    """同时生成JSON数据和对应的标注，确保它们一致"""
    config = {
        "E": 1,
        "BeamOffset": [0, 35, 1, 229, 2, 115, 3, 175, 4, 123, 5, 150, 6, 20, 7, 268, 8, 279],
        "RailType": "60",
        "RailHgt": 176
    }
    
    data = [config]
    annotations = []
    
    # 预先定义 3-5 个伤损区域
    damage_regions = []
    for _ in range(random.randint(3, 5)):
        start_e = random.randint(150, 450)
        length = random.randint(10, 30)
        channel = random.randint(0, 8)
        damage_type = random.randint(1, 10)
        damage_regions.append({
            "start_e": start_e,
            "end_e": start_e + length,
            "channel": channel,
            "type": damage_type
        })
        
        # 为标注准备点
        points = []
        for e in range(start_e, start_e + length):
            points.append({
                "e": e,
                "channel": channel,
                "depth": 100.0, # 固定深度
                "amplitude": 100.0 # 高增益
            })
            
        annotations.append({
            "id": str(random.randint(1000, 9999)),
            "type": "伤损",
            "subtype": {
                "id": str(damage_type),
                "name": f"模拟伤损_{damage_type}"
            },
            "points": points
        })

    # 生成每一帧的数据
    for e in range(100, 100 + n_samples):
        b1_data = []
        
        # 检查当前 e 是否在任何伤损区域内
        active_damages = [d for d in damage_regions if d["start_e"] <= e < d["end_e"]]
        active_channels = {d["channel"] for d in active_damages}
        
        for ch in range(9):
            if ch in active_channels:
                # 伤损信号：高振幅，固定深度
                depth = 100.0
                amplitude = random.uniform(90, 100)
                b1_data.extend([float(ch), depth, amplitude, 250.0, 176.0])
            elif random.random() > 0.8: # 20% 概率产生背景噪声
                # 正常信号：低振幅，随机深度
                depth = random.uniform(50, 200)
                amplitude = random.uniform(40, 60)
                b1_data.extend([float(ch), depth, amplitude, 250.0, 176.0])
        
        if b1_data:
            data.append({
                "E": e,
                "B1": b1_data
            })
            
    # 写入文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    with open(anno_path, 'w', encoding='utf-8') as f:
        json.dump({"annotations": annotations}, f, ensure_ascii=False, indent=2)

def main():
    train_dir = Path(r"d:\HDC\data\train")
    test_dir = Path(r"d:\HDC\data\test")
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成 5 个训练文件
    for i in range(1, 6):
        create_sample_data(train_dir / f"train_{i}.json", train_dir / f"train_{i}标注.json")
        
    # 生成 3 个测试文件
    for i in range(1, 4):
        create_sample_data(test_dir / f"test_{i}.json", test_dir / f"test_{i}标注.json")
        
    print("一致性测试数据生成完成")

if __name__ == "__main__":
    main()
