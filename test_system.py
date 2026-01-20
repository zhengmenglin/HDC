"""
测试脚本
验证超声波探伤HDC系统的各个模块
"""

import os
import numpy as np
from config import config
from ultrasonic_data_converter import UltrasonicDataConverter
from feature_extractor import FeatureExtractor
from hd_encoder import HDCEncoder
from hd_classifier import HDCClasifier
from trainer import Trainer
from evaluator import Evaluator
from realtime_processor import RealtimeProcessor

def test_config():
    """测试配置模块"""
    print("\n" + "="*50)
    print("测试1: 配置模块")
    print("="*50)
    
    print(f"数据目录: {config.get('data.input_dir')}")
    print(f"输出目录: {config.get('data.output_dir')}")
    print(f"模型维度: {config.get('model.dimensions')}")
    print(f"VSA类型: {config.get('model.vsa_type')}")
    print(f"分类器: {config.get('model.classifier')}")
    print("✓ 配置模块测试通过")

def test_converter():
    """测试数据转换器"""
    print("\n" + "="*50)
    print("测试2: 数据转换器")
    print("="*50)
    
    converter = UltrasonicDataConverter()
    
    # 测试类别映射
    print(f"类别映射: {converter.class_mapping}")
    print(f"反向映射: {converter.class_reverse_mapping}")
    
    # 测试归一化
    test_data = np.array([[1, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                          [2, 15, 25, 35, 45, 55, 65, 75, 85, 95]])
    normalized = converter.normalize_data(test_data)
    print(f"归一化前: {test_data[0]}")
    print(f"归一化后: {normalized[0]}")
    
    print("✓ 数据转换器测试通过")

def test_feature_extractor():
    """测试特征提取器"""
    print("\n" + "="*50)
    print("测试3: 特征提取器")
    print("="*50)
    
    extractor = FeatureExtractor(dimensions=1000, vsa_type="MAP")
    
    # 测试通道分离
    test_data = np.random.rand(10, 10) * 200
    test_data[:, 0] = np.random.randint(0, 999, 10)
    
    channels = extractor.separate_channels(test_data)
    print(f"输入数据形状: {test_data.shape}")
    print(f"通道数量: {len(channels)}")
    print(f"每个通道数据形状: {channels[0].shape}")
    
    # 测试特征提取
    hv_batch = extractor.extract_features(test_data)
    print(f"超维向量形状: {hv_batch.shape}")
    
    # 测试特征信息
    info = extractor.get_feature_info()
    print(f"特征信息: {info}")
    
    print("✓ 特征提取器测试通过")

def test_hd_encoder():
    """测试HDC编码器"""
    print("\n" + "="*50)
    print("测试4: HDC编码器")
    print("="*50)
    
    encoder = HDCEncoder(dimensions=1000, vsa_type="MAP")
    
    # 测试深度编码
    depth_hv = encoder.encode_depth_continuous(125.0)
    print(f"深度值 125.0 编码为超向量，形状: {depth_hv.shape}")
    
    # 测试幅度编码
    amp_hv = encoder.encode_amplitude_continuous(80.0)
    print(f"幅度值 80.0 编码为超向量，形状: {amp_hv.shape}")
    
    # 测试通道编码
    channel_hv = encoder.encode_channel(6)
    print(f"通道 6 编码为超向量，形状: {channel_hv.shape}")
    
    # 测试相似度计算
    similarity = encoder.compute_similarity(depth_hv, amp_hv)
    print(f"深度和幅度超向量的相似度: {similarity:.4f}")
    
    # 测试循环移位编码
    hvs = torch.stack([depth_hv, amp_hv, channel_hv], dim=0)
    fused_hv = encoder.circular_shift_encode(hvs)
    print(f"融合后的超向量形状: {fused_hv.shape}")
    
    # 测试编码器信息
    info = encoder.get_info()
    print(f"编码器信息: {info}")
    
    print("✓ HDC编码器测试通过")

def test_classifier():
    """测试HDC分类器"""
    print("\n" + "="*50)
    print("测试5: HDC分类器")
    print("="*50)
    
    classifier = HDCClasifier(n_classes=20, dimensions=1000, vsa_type="MAP")
    
    # 生成测试数据
    train_data = np.random.rand(100, 10) * 200
    train_data[:, 0] = np.random.randint(0, 999, 100)
    train_labels = np.random.randint(0, 20, 100)
    
    # 训练模型
    print("开始训练...")
    train_info = classifier.train(train_data, train_labels)
    print(f"训练完成，样本数: {train_info['n_samples']}")
    
    # 测试预测
    test_data = np.random.rand(10, 10) * 200
    test_data[:, 0] = np.random.randint(0, 999, 10)
    
    predictions = classifier.predict(test_data)
    print(f"预测结果: {predictions}")
    
    # 测试带置信度的预测
    predictions, confidences = classifier.predict_with_confidence(test_data)
    print(f"预测结果: {predictions}")
    print(f"置信度: {confidences}")
    
    # 测试类别名称
    class_name = classifier.get_class_name(1)
    print(f"类别1的名称: {class_name}")
    
    # 测试模型信息
    info = classifier.get_model_info()
    print(f"模型信息: {info}")
    
    print("✓ HDC分类器测试通过")

def test_trainer():
    """测试训练器"""
    print("\n" + "="*50)
    print("测试6: 训练器")
    print("="*50)
    
    trainer = Trainer()
    
    # 生成测试数据
    data = np.random.rand(200, 10) * 200
    data[:, 0] = np.random.randint(0, 999, 200)
    labels = np.random.randint(0, 20, 200)
    
    # 划分数据
    train_data, train_labels, test_data, test_labels = trainer.converter.split_train_test(data, labels)
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 训练模型
    print("开始训练...")
    results = trainer.train_model(train_data, train_labels, test_data, test_labels)
    print(f"训练完成，模型路径: {results['model_path']}")
    
    print("✓ 训练器测试通过")

def test_evaluator():
    """测试评估器"""
    print("\n" + "="*50)
    print("测试7: 评估器")
    print("="*50)
    
    evaluator = Evaluator()
    
    # 创建分类器
    classifier = HDCClasifier(n_classes=20, dimensions=1000, vsa_type="MAP")
    
    # 生成测试数据
    train_data = np.random.rand(100, 10) * 200
    train_data[:, 0] = np.random.randint(0, 999, 100)
    train_labels = np.random.randint(0, 20, 100)
    
    # 训练模型
    classifier.train(train_data, train_labels)
    
    # 生成测试数据
    test_data = np.random.rand(50, 10) * 200
    test_data[:, 0] = np.random.randint(0, 999, 50)
    test_labels = np.random.randint(0, 20, 50)
    
    # 评估模型
    eval_results = evaluator.evaluate(classifier, test_data, test_labels)
    
    # 打印评估报告
    evaluator.print_evaluation_report(eval_results, classifier)
    
    print("✓ 评估器测试通过")

def test_realtime_processor():
    """测试实时处理器"""
    print("\n" + "="*50)
    print("测试8: 实时处理器")
    print("="*50)
    
    # 创建分类器并训练
    classifier = HDCClasifier(n_classes=20, dimensions=1000, vsa_type="MAP")
    train_data = np.random.rand(100, 10) * 200
    train_data[:, 0] = np.random.randint(0, 999, 100)
    train_labels = np.random.randint(0, 20, 100)
    classifier.train(train_data, train_labels)
    
    # 保存模型
    model_path = "output/test_model.pth"
    os.makedirs("output", exist_ok=True)
    classifier.save_model(model_path)
    
    # 创建实时处理器
    processor = RealtimeProcessor(model_path=model_path)
    
    # 测试单样本处理
    sample = np.random.rand(10) * 200
    sample[0] = np.random.randint(0, 999)
    result = processor.process_sample(sample, true_label=5)
    print(f"处理结果: {result}")
    
    # 测试批量处理
    batch_data = np.random.rand(10, 10) * 200
    batch_data[:, 0] = np.random.randint(0, 999, 10)
    batch_labels = np.random.randint(0, 20, 10)
    
    batch_results = processor.process_batch(batch_data, batch_labels)
    print(f"批量处理结果数: {len(batch_results)}")
    
    # 获取统计信息
    stats = processor.get_statistics()
    print(f"统计信息: {stats}")
    
    print("✓ 实时处理器测试通过")

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*50)
    print("超声波探伤HDC系统 - 模块测试")
    print("="*50)
    
    try:
        test_config()
        test_converter()
        test_feature_extractor()
        test_hd_encoder()
        test_classifier()
        test_trainer()
        test_evaluator()
        test_realtime_processor()
        
        print("\n" + "="*50)
        print("所有测试通过！")
        print("="*50)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()