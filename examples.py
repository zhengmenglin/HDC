"""
超声波探伤HDC系统使用示例
"""

import os
import numpy as np
from main_controller import MainController

def example_basic_usage():
    """基本使用示例"""
    print("示例1: 基本使用流程")
    print("="*50)
    
    # 创建控制器
    controller = MainController()
    
    # 1. 数据转换
    print("\n1. 数据转换")
    csv_files = controller.convert_data("data", "output")
    print(f"已转换 {len(csv_files)} 个文件")
    
    # 2. 模型训练
    print("\n2. 模型训练")
    train_results = controller.train_model(
        train_csv="output/train.csv",
        model_path="output/hd_model.pth",
        save_path="output/hd_model_info.json"
    )
    print(f"训练完成，样本数: {train_results['train_info']['n_samples']}")
    
    # 3. 模型评估
    print("\n3. 模型评估")
    eval_results = controller.evaluate_model(
        test_csv="output/test.csv",
        model_path="output/hd_model.pth"
    )
    print(f"评估完成，准确率: {eval_results['accuracy']:.4f}")

def example_realtime_recognition():
    """实时识别示例"""
    print("\n示例2: 实时识别")
    print("="*50)
    
    # 创建控制器
    controller = MainController()
    
    # 加载模型
    model_path = "output/hd_model.pth"
    
    # 模拟实时数据流
    print("\n模拟实时数据流...")
    data_stream = []
    for i in range(100):
        # 生成随机样本
        sample = np.random.rand(10) * 200
        sample[0] = np.random.randint(0, 999)  # 编码值
        data_stream.append(sample)
    
    # 实时识别
    results = controller.realtime_recognition(data_stream, model_path)
    
    print(f"已处理 {len(results)} 个样本")
    print(f"识别结果: {results[0]['class_name']}")

def example_incremental_training():
    """增量训练示例"""
    print("\n示例3: 增量训练")
    print("="*50)
    
    from realtime_processor import RealtimeProcessor
    
    # 创建实时处理器
    processor = RealtimeProcessor(model_path="output/hd_model.pth")
    
    # 模拟新数据
    new_data = np.random.rand(50, 10) * 200
    new_labels = np.random.randint(1, 20, 50)  # 随机标签
    
    # 增量训练
    for i in range(len(new_data)):
        processor.process_sample(new_data[i], new_labels[i])
    
    # 定期更新模型
    if processor.total_samples % processor.update_interval == 0:
        processor.update_model_incremental()
    
    # 获取统计信息
    stats = processor.get_statistics()
    print(f"总样本数: {stats['total_samples']}")
    print(f"准确率: {stats['accuracy']:.4f}")

def example_cross_validation():
    """交叉验证示例"""
    print("\n示例4: 交叉验证")
    print("="*50)
    
    from trainer import Trainer
    from ultrasonic_data_converter import UltrasonicDataConverter
    
    # 创建训练器
    trainer = Trainer()
    
    # 准备数据
    converter = UltrasonicDataConverter()
    data, labels = converter.load_csv("output/train.csv")
    
    # 交叉验证
    cv_results = trainer.cross_validate(data, labels, folds=5)
    
    print(f"交叉验证完成，总样本数: {cv_results['total_samples']}")
    if cv_results['average_results']:
        avg_accuracy = cv_results['average_results']['accuracy']
        print(f"平均准确率: {avg_accuracy:.4f}")

def example_custom_configuration():
    """自定义配置示例"""
    print("\n示例5: 自定义配置")
    print("="*50)
    
    from config import config
    
    # 修改配置
    config.set("model.dimensions", 20000)  # 增加维度
    config.set("model.epochs", 200)  # 增加训练轮数
    config.set("model.learning_rate", 0.05)  # 调整学习率
    config.set("realtime.threshold", 0.6)  # 提高阈值
    
    print("配置已更新:")
    print(f"模型维度: {config.get('model.dimensions')}")
    print(f"训练轮数: {config.get('model.epochs')}")
    print(f"学习率: {config.get('model.learning_rate')}")
    print(f"实时阈值: {config.get('realtime.threshold')}")

def example_feature_extraction():
    """特征提取示例"""
    print("\n示例6: 特征提取")
    print("="*50)
    
    from feature_extractor import FeatureExtractor
    
    # 创建特征提取器
    extractor = FeatureExtractor(dimensions=10000, vsa_type="MAP")
    
    # 模拟数据
    data = np.random.rand(10, 10) * 200
    data[:, 0] = np.random.randint(0, 999, 10)  # 编码值
    
    # 提取特征
    hv_batch = extractor.extract_features(data)
    
    print(f"输入数据形状: {data.shape}")
    print(f"超维向量形状: {hv_batch.shape}")
    print(f"特征信息: {extractor.get_feature_info()}")

def example_hd_encoding():
    """HDC编码示例"""
    print("\n示例7: HDC编码")
    print("="*50)
    
    from hd_encoder import HDCEncoder
    
    # 创建编码器
    encoder = HDCEncoder(dimensions=10000, vsa_type="MAP")
    
    # 编码深度值
    depth = 125.0
    depth_hv = encoder.encode_depth_continuous(depth)
    print(f"深度值 {depth} 编码为超向量")
    
    # 编码幅度值
    amplitude = 80.0
    amp_hv = encoder.encode_amplitude_continuous(amplitude)
    print(f"幅度值 {amplitude} 编码为超向量")
    
    # 编码通道
    channel = 6
    channel_hv = encoder.encode_channel(channel)
    print(f"通道 {channel} 编码为超向量")
    
    # 计算相似度
    similarity = encoder.compute_similarity(depth_hv, amp_hv)
    print(f"深度和幅度超向量的相似度: {similarity:.4f}")

if __name__ == "__main__":
    print("超声波探伤HDC系统使用示例")
    print("="*50)
    
    # 运行所有示例
    example_basic_usage()
    example_realtime_recognition()
    example_incremental_training()
    example_cross_validation()
    example_custom_configuration()
    example_feature_extraction()
    example_hd_encoding()
    
    print("\n所有示例运行完成！")