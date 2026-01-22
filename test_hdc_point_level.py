import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from hdc_point_level_classifier import HDCPointLevelDataset, HDCPointLevelModel, WINDOW_SIZE, NUM_CHANNELS, CENTER_IDX

def create_mock_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建一个模拟的 1_hdc.csv
    # 至少需要 WINDOW_SIZE + 10 行
    num_rows = WINDOW_SIZE + 50
    encodings = np.arange(1000, 1000 + num_rows)
    channels = {f'channel_{i}': np.random.randint(0, 256, num_rows) for i in range(NUM_CHANNELS)}
    
    hdc_df = pd.DataFrame({'encoding': encodings, **channels})
    hdc_df.to_csv(os.path.join(data_dir, "test_sample_hdc.csv"), index=False)
    
    # 创建一个模拟的 1_label.csv
    # 只标注几个点
    label_indices = [CENTER_IDX, CENTER_IDX + 10, CENTER_IDX + 20]
    labels = []
    for idx in label_indices:
        labels.append({
            'encoding': encodings[idx],
            'label_0': 1, # 通道 0 有伤损
            'label_1': 5, # 通道 1 有伤损
            'label_2': 0
        })
    
    label_df = pd.DataFrame(labels)
    # 补齐其他通道列
    for i in range(3, NUM_CHANNELS):
        label_df[f'label_{i}'] = 0
        
    label_df.to_csv(os.path.join(data_dir, "test_sample_label.csv"), index=False)
    print(f"模拟数据已创建在: {data_dir}")

def test_hdc_point_level():
    test_dir = "temp_test_data"
    create_mock_data(test_dir)
    
    try:
        # 1. 测试数据集加载
        print("\n--- 测试数据集 ---")
        dataset = HDCPointLevelDataset(test_dir, is_train=True)
        print(f"数据集样本数: {len(dataset)}")
        assert len(dataset) > 0, "数据集应该包含样本"
        
        sample_x, sample_y = dataset[0]
        print(f"样本输入形状: {sample_x.shape}") # 应为 [256, 9]
        print(f"样本标签形状: {sample_y.shape}") # 应为 [9]
        assert sample_x.shape == (WINDOW_SIZE, NUM_CHANNELS)
        assert sample_y.shape == (NUM_CHANNELS,)

        # 2. 测试模型训练
        print("\n--- 测试训练 ---")
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataset, batch_size=2)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HDCPointLevelModel(dimensions=1000, device=device) # 使用较小维度加速测试
        model.train(train_loader)
        
        # 3. 测试模型预测
        print("\n--- 测试预测 ---")
        test_dataset = HDCPointLevelDataset(test_dir, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=2)
        
        predictions = model.predict(test_loader)
        print(f"预测结果形状: {predictions.shape}") # 应为 [N, 9]
        assert predictions.shape == (len(test_dataset), NUM_CHANNELS)
        
        # 4. 测试模型保存与加载
        print("\n--- 测试保存与加载 ---")
        model_path = "temp_model.pth"
        model.save(model_path)
        
        new_model = HDCPointLevelModel(dimensions=1000, device=device)
        new_model.load(model_path, window_size=WINDOW_SIZE)
        
        new_predictions = new_model.predict(test_loader)
        assert np.array_equal(predictions, new_predictions), "加载后的模型预测结果应一致"
        
        print("\n所有测试通过!")
        
    finally:
        # 清理临时文件
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists("temp_model.pth"):
            os.remove("temp_model.pth")

if __name__ == "__main__":
    test_hdc_point_level()
