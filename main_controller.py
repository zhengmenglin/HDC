"""
主控制器
整合所有模块，实现超声波探伤HDC系统
"""

import os
import json
from typing import Dict, List, Optional
from config import config
from ultrasonic_data_converter import UltrasonicDataConverter
from feature_extractor import FeatureExtractor
from hd_encoder import HDCEncoder
from hd_classifier import HDCClasifier
from trainer import Trainer
from evaluator import Evaluator
from realtime_processor import RealtimeProcessor

class MainController:
    """主控制器"""
    
    def __init__(self):
        # 创建输出目录
        output_dir = config.get("data.output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化各个模块
        self.converter = UltrasonicDataConverter()
        self.feature_extractor = FeatureExtractor()
        self.hd_encoder = HDCEncoder()
        self.classifier = HDCClasifier()
        self.trainer = Trainer()
        self.evaluator = Evaluator()
        self.realtime_processor = RealtimeProcessor()
    
    def convert_data(self, input_dir: str = None, 
                   output_dir: str = None) -> List[str]:
        """转换数据格式"""
        input_dir = input_dir or config.get("data.input_dir", "data")
        output_dir = output_dir or config.get("data.output_dir", "output")
        
        return self.converter.convert_directory(input_dir, output_dir)
    
    def train_model(self, train_csv: str = None, test_csv: str = None, 
                  model_path: str = None, save_path: str = None) -> Dict:
        """训练模型"""
        return self.trainer.train_from_files(train_csv, test_csv, model_path, save_path)
    
    def evaluate_model(self, test_csv: str = None, model_path: str = None) -> Dict:
        """评估模型"""
        # 加载测试数据
        test_data, test_labels = self.converter.load_csv(test_csv)
        
        # 加载模型
        self.classifier.load_model(model_path)
        
        # 评估模型
        eval_results = self.evaluator.evaluate(self.classifier, test_data, test_labels)
        
        # 打印评估报告
        self.evaluator.print_evaluation_report(eval_results, self.classifier)
        
        # 保存评估结果
        eval_path = os.path.join(
            os.path.dirname(model_path) if model_path else config.get("data.output_dir", "output"),
            "evaluation_results.json"
        )
        self.evaluator.save_evaluation_results(eval_results, eval_path)
        
        return eval_results
    
    def realtime_recognition(self, data_stream: List[np.ndarray], 
                           model_path: str = None) -> List[Dict]:
        """实时识别"""
        # 加载模型
        self.realtime_processor.load_model(model_path)
        
        # 处理数据流
        all_results = []
        for data_batch in data_stream:
            batch_results = self.realtime_processor.process_batch(data_batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def run_pipeline(self, input_dir: str = None, 
                    model_path: str = None) -> None:
        """运行完整流水线"""
        # 1. 数据转换
        print("步骤1: 数据转换")
        csv_files = self.convert_data(input_dir)
        
        # 2. 模型训练
        print("步骤2: 模型训练")
        train_results = self.train_model(csv_files[0])  # 使用第一个CSV文件训练
        
        # 3. 模型评估
        print("步骤3: 模型评估")
        eval_results = self.evaluate_model(test_csv=csv_files[1], model_path=train_results["model_path"])
        
        # 4. 实时识别（模拟）
        print("步骤4: 实时识别（模拟）")
        # 模拟数据流
        import random
        data_stream = []
        for _ in range(100):  # 模拟100个样本
            # 生成随机数据
            sample = np.random.rand(10) * 200  # 10个特征值，范围0-200
            sample[0] = random.randint(0, 999)  # 编码值
            for j in range(1, 10):
                sample[j] = random.uniform(0.0, 200.0)  # 深度值
            
            data_stream.append(sample)
        
        # 实时识别
        realtime_results = self.realtime_recognition(data_stream, train_results["model_path"])
        
        # 生成标注文件
        output_dir = config.get("data.output_dir", "output")
        self.realtime_processor.generate_annotation_json(realtime_results, 
            os.path.join(output_dir, "realtime_annotations.json")
        )
        
        print("流水线执行完成！")
        
        return {
            "csv_files": csv_files,
            "train_results": train_results,
            "eval_results": eval_results,
            "realtime_results": realtime_results
        }
    
    def interactive_mode(self) -> None:
        """交互模式"""
        print("超声波探伤HDC系统 - 交互模式")
        print("="*50)
        
        while True:
            print("\n可用命令:")
            print("1. convert - 转换数据格式")
            print("2. train - 训练模型")
            print("3. evaluate - 评估模型")
            print("4. pipeline - 运行完整流水线")
            print("5. realtime - 实时识别模式")
            print("6. config - 显示配置")
            print("7. help - 显示帮助")
            print("8. exit - 退出")
            print("="*50)
            
            command = input("\n请输入命令: ").strip().lower()
            
            if command == "convert":
                input_dir = input("\n请输入数据目录: ")
                output_dir = input("\n请输入输出目录: ")
                csv_files = self.convert_data(input_dir, output_dir)
                print(f"已转换 {len(csv_files)} 个文件")
            
            elif command == "train":
                train_csv = input("\n请输入训练CSV文件: ")
                model_path = input("\n请输入模型保存路径: ")
                save_path = input("\n请输入训练信息保存路径: ")
                train_results = self.train_model(train_csv, model_path, save_path)
                print(f"训练完成，模型已保存到: {train_results['model_path']}")
            
            elif command == "evaluate":
                test_csv = input("\n请输入测试CSV文件: ")
                model_path = input("\n请输入模型路径: ")
                eval_results = self.evaluate_model(test_csv, model_path)
                print(f"评估完成，结果已保存")
            
            elif command == "pipeline":
                input_dir = input("\n请输入数据目录: ")
                results = self.run_pipeline(input_dir)
                print("流水线执行完成")
            
            elif command == "realtime":
                model_path = input("\n请输入模型路径: ")
                print("进入实时识别模式，按Ctrl+C退出")
                self.realtime_recognition([], model_path)
            
            elif command == "config":
                print("\n当前配置:")
                print(f"数据目录: {config.get('data.input_dir', 'data')}")
                print(f"输出目录: {config.get('data.output_dir', 'output')}")
                print(f"模型维度: {config.get('model.dimensions', 10000)}")
                print(f"VSA类型: {config.get('model.vsa_type', 'MAP')}")
                print(f"分类器: {config.get('model.classifier', 'OnlineHD')}")
                print(f"学习率: {config.get('model.learning_rate', 0.035)}")
                print(f"训练轮数: {config.get('model.epochs', 120)}")
                print(f"实时阈值: {config.get('realtime.threshold', 0.5)}")
                print(f"更新间隔: {config.get('realtime.update_interval', 100)}")
                print(f"批处理大小: {config.get('realtime.batch_size', 1)}")
            
            elif command == "help":
                print("\n超声波探伤HDC系统使用说明:")
                print("\n1. 数据准备:")
                print("   - 将JSON格式的原始数据放入data目录")
                print("   - 对应的标注文件放入data目录")
                print("\n2. 模型训练:")
                print("   - 使用'train'命令训练模型")
                print("   - 支持从CSV文件训练或增量训练")
                print("\n3. 模型评估:")
                print("   - 使用'evaluate'命令评估模型性能")
                print("   - 支持多种评估指标")
                print("\n4. 实时识别:")
                print("   - 使用'realtime'命令进行实时识别")
                print("   - 可设置相似度阈值")
                print("\n5. 配置管理:")
                print("   - 使用'config'命令查看和修改配置")
                print("\n6. 退出:")
                print("   - 使用'exit'命令退出程序")
            
            elif command == "exit":
                print("退出程序")
                break
            
            else:
                print("未知命令，请重新输入")
    
    def show_help(self) -> None:
        """显示帮助信息"""
        print("\n超声波探伤HDC系统帮助:")
        print("\n命令说明:")
        print("  convert - 转换数据格式")
        print("  train - 训练模型")
        print("  evaluate - 评估模型")
        print("  pipeline - 运行完整流水线")
        print("  realtime - 实时识别模式")
        print("  config - 显示配置")
        print("  help - 显示帮助")
        print("  exit - 退出")
        print("\n使用示例:")
        print("  python main_controller.py pipeline data/")
        print("  python main_controller.py train data/train.csv")
        print("  python main_controller.py evaluate data/test.csv models/hd_model.pth")

if __name__ == "__main__":
    controller = MainController()
    
    if len(os.sys.argv) > 1:
        command = os.sys.argv[1]
        if command == "interactive":
            controller.interactive_mode()
        else:
            controller.interactive_mode()