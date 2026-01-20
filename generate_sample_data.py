"""
示例数据生成器
生成模拟的超声波探伤数据用于测试系统
"""

import os
import json
import random
import numpy as np
from typing import List, Dict

class DataGenerator:
    """示例数据生成器"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 伤损类型
        self.damage_types = [
            "轨头外侧核伤", "轨头中心核伤", "轨头内侧核伤",
            "轨头内、外、中心核伤", "轨头内、中心核伤",
            "轨头外、中心核伤", "轨头水平分裂",
            "轨头、轨腰分裂", "轨腰水平裂", "轨底断裂",
            "螺孔裂纹", "接头", "螺孔", "导线孔",
            "焊缝", "鱼鳞伤", "轨底锈蚀", "轨底焊筋", "道岔"
        ]
        
        # 正常结构类型
        self.structure_types = ["接头", "螺孔", "导线孔", "焊缝", "鱼鳞伤", "轨底锈蚀", "轨底焊筋", "道岔"]
    
    def generate_sample_data(self, n_samples: int = 100, 
                          damage_ratio: float = 0.3) -> Dict:
        """生成示例数据
        
        Args:
            n_samples: 样本数量
            damage_ratio: 伤损比例
        
        Returns:
            JSON格式的数据
        """
        data = {}
        annotations = []
        
        # 生成编码值序列
        encoding_values = list(range(1, n_samples + 1))
        
        # 随机选择伤损样本
        n_damages = int(n_samples * damage_ratio)
        damage_indices = random.sample(range(n_samples), n_damages)
        
        for i, e_value in enumerate(encoding_values):
            # 生成B扫描数据
            bscan_data = self._generate_bscan_data(e_value)
            data[f"B{i+1}"] = bscan_data
            
            # 如果是伤损样本，添加标注
            if i in damage_indices:
                damage_type = random.choice(self.damage_types)
                annotation = self._generate_annotation(e_value, damage_type)
                annotations.append(annotation)
        
        # 添加编码值
        data["E"] = encoding_values[-1]
        
        return data, annotations
    
    def _generate_bscan_data(self, e_value: int) -> List:
        """生成B扫描数据
        
        Args:
            e_value: 编码值
        
        Returns:
            B扫描数据列表 [channel, depth, amplitude, ...]
        """
        # 随机选择通道
        channel = random.randint(0, 8)
        
        # 生成深度值
        depth = random.uniform(50.0, 200.0)
        
        # 生成幅度值
        amplitude = random.uniform(60.0, 100.0)
        
        # 生成额外的dB值
        db_value = random.uniform(200.0, 300.0)
        
        return [channel, depth, amplitude, db_value]
    
    def _generate_annotation(self, e_value: int, damage_type: str) -> Dict:
        """生成标注数据
        
        Args:
            e_value: 编码值
            damage_type: 伤损类型
        
        Returns:
            标注字典
        """
        # 获取伤损ID
        damage_id = self.damage_types.index(damage_type) + 1
        
        # 生成点数据
        channel = random.randint(0, 8)
        depth = random.uniform(50.0, 200.0)
        amplitude = random.uniform(60.0, 100.0)
        
        annotation = {
            "id": str(e_value),
            "type": "伤损" if damage_type not in self.structure_types else "结构",
            "subtype": {
                "id": str(100 + damage_id),
                "name": damage_type
            },
            "points": [
                {
                    "e": e_value,
                    "channel": channel,
                    "depth": depth,
                    "amplitude": amplitude
                }
            ]
        }
        
        return annotation
    
    def generate_dataset(self, n_files: int = 10, 
                      samples_per_file: int = 100,
                      damage_ratio: float = 0.3) -> List[str]:
        """生成数据集
        
        Args:
            n_files: 文件数量
            samples_per_file: 每个文件的样本数
            damage_ratio: 伤损比例
        
        Returns:
            生成的文件路径列表
        """
        file_paths = []
        
        for i in range(n_files):
            # 生成数据
            data, annotations = self.generate_sample_data(samples_per_file, damage_ratio)
            
            # 保存原始数据
            data_filename = f"sample_{i+1}.json"
            data_path = os.path.join(self.output_dir, data_filename)
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 保存标注数据
            annotation_filename = f"sample_{i+1}标注.json"
            annotation_path = os.path.join(self.output_dir, annotation_filename)
            annotation_data = {"annotations": annotations}
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            file_paths.extend([data_path, annotation_path])
            print(f"已生成: {data_filename} 和 {annotation_filename}")
        
        return file_paths
    
    def generate_train_test_split(self, n_train_files: int = 7, 
                               n_test_files: int = 3,
                               samples_per_file: int = 100,
                               damage_ratio: float = 0.3) -> Dict:
        """生成训练集和测试集
        
        Args:
            n_train_files: 训练文件数量
            n_test_files: 测试文件数量
            samples_per_file: 每个文件的样本数
            damage_ratio: 伤损比例
        
        Returns:
            包含训练和测试文件路径的字典
        """
        # 生成训练数据
        print("生成训练数据...")
        train_dir = os.path.join(self.output_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        
        train_generator = DataGenerator(train_dir)
        train_files = train_generator.generate_dataset(
            n_train_files, samples_per_file, damage_ratio
        )
        
        # 生成测试数据
        print("\n生成测试数据...")
        test_dir = os.path.join(self.output_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        
        test_generator = DataGenerator(test_dir)
        test_files = test_generator.generate_dataset(
            n_test_files, samples_per_file, damage_ratio
        )
        
        return {
            "train_files": train_files,
            "test_files": test_files,
            "train_dir": train_dir,
            "test_dir": test_dir
        }

def main():
    """主函数"""
    print("="*50)
    print("超声波探伤HDC系统 - 示例数据生成器")
    print("="*50)
    
    # 创建数据生成器
    generator = DataGenerator("data")
    
    # 生成训练集和测试集
    print("\n生成训练集和测试集...")
    result = generator.generate_train_test_split(
        n_train_files=7,
        n_test_files=3,
        samples_per_file=100,
        damage_ratio=0.3
    )
    
    print("\n" + "="*50)
    print("数据生成完成！")
    print("="*50)
    print(f"训练数据目录: {result['train_dir']}")
    print(f"测试数据目录: {result['test_dir']}")
    print(f"训练文件数: {len(result['train_files'])}")
    print(f"测试文件数: {len(result['test_files'])}")
    print("\n现在可以运行以下命令:")
    print("  python main_controller.py interactive")
    print("  python main_controller.py pipeline data/")
    print("  python test_system.py")

if __name__ == "__main__":
    main()