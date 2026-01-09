import numpy as np
from hdc_ultrasonic_bscan import HyperdimensionalComputing
from HDC import HyperdimensionalEncoder

def test_two_encoding_methods():
    """对比两种编码方法的相似度"""
    
    print("=" * 80)
    print("对比两种编码方法：连续映射 vs 分段编码")
    print("=" * 80)
    
    # 测试参数
    value_range = (0, 100)
    num_segments = 10
    
    print(f"\n测试参数: value_range={value_range}, num_segments={num_segments}")
    
    # 方法1：连续映射（HDC.py）
    print("\n" + "-" * 80)
    print("方法1：连续映射（HDC.py）")
    print("-" * 80)
    
    encoder1 = HyperdimensionalEncoder(d=10000, seed=42)
    encoder1.continuous_mapping(min_val=value_range[0], max_val=value_range[1], 
                               num_levels=num_segments, base_key='test')
    
    # 连续映射生成的向量是从min_val到min_val+num_levels-1
    min_vector1 = encoder1.item_memory['test'][value_range[0]]
    max_vector1 = encoder1.item_memory['test'][value_range[0] + num_segments - 1]
    
    hamming_sim1 = encoder1.hamming_similarity(min_vector1, max_vector1)
    cosine_sim1 = encoder1.cosine_similarity(min_vector1, max_vector1)
    
    print(f"最小值({value_range[0]})和最大值({value_range[0] + num_segments - 1})的相似度:")
    print(f"  汉明相似度: {hamming_sim1:.4f}")
    print(f"  余弦相似度: {cosine_sim1:.4f}")
    
    # 方法2：分段编码（hdc_ultrasonic_bscan.py）
    print("\n" + "-" * 80)
    print("方法2：分段编码（hdc_ultrasonic_bscan.py）")
    print("-" * 80)
    
    encoder2 = HyperdimensionalComputing(dimension=10000)
    np.random.seed(42)  # 设置相同的随机种子
    
    # 分段编码：最小值映射到段0，最大值映射到段num_segments-1
    min_vector2 = encoder2.encode_value(value_range[0], value_range, num_segments)
    max_vector2 = encoder2.encode_value(value_range[1], value_range, num_segments)
    
    # 计算相似度
    hamming_sim2 = np.sum(min_vector2 == max_vector2) / len(min_vector2)
    cosine_sim2 = np.dot(min_vector2, max_vector2) / (np.linalg.norm(min_vector2) * np.linalg.norm(max_vector2))
    
    print(f"最小值({value_range[0]})和最大值({value_range[1]})的相似度:")
    print(f"  汉明相似度: {hamming_sim2:.4f}")
    print(f"  余弦相似度: {cosine_sim2:.4f}")
    
    # 测试多个随机种子，观察分段编码的随机性
    print("\n" + "-" * 80)
    print("分段编码的随机性测试（不同随机种子）")
    print("-" * 80)
    
    seeds = [42, 123, 456, 789, 999]
    for seed in seeds:
        encoder_test = HyperdimensionalComputing(dimension=10000)
        np.random.seed(seed)
        
        min_vec = encoder_test.encode_value(value_range[0], value_range, num_segments)
        max_vec = encoder_test.encode_value(value_range[1], value_range, num_segments)
        
        hamming = np.sum(min_vec == max_vec) / len(min_vec)
        cosine = np.dot(min_vec, max_vec) / (np.linalg.norm(min_vec) * np.linalg.norm(max_vec))
        
        print(f"种子={seed:3d}: 汉明相似度={hamming:.4f}, 余弦相似度={cosine:.4f}")
    
    print("\n" + "=" * 80)
    print("分析结论")
    print("=" * 80)
    print("1. 连续映射（HDC.py）:")
    print("   - 采用逐步翻转的方式生成数值向量")
    print("   - 相邻数值的向量之间有连续性")
    print("   - 最小值和最大值的相似度相对稳定（约0.5-0.68）")
    print("   - 适合需要保持数值顺序关系的场景")
    print()
    print("2. 分段编码（hdc_ultrasonic_bscan.py）:")
    print("   - 每个段对应一个独立的随机向量")
    print("   - 相邻段的向量之间没有连续性关系")
    print("   - 最小值和最大值的相似度完全随机（可能为0，也可能接近1）")
    print("   - 适合离散分类场景，不需要保持数值顺序")
    print()
    print("3. 为什么有的相似度为0.5，有的为0:")
    print("   - 相似度0.5：使用连续映射，num_levels=2")
    print("   - 相似度0：使用分段编码，最小值和最大值对应完全不同的随机向量")
    print("   - 相似度0.68：使用连续映射，num_levels较大，重复翻转导致实际翻转位数减少")
    print()
    print("4. 选择建议:")
    print("   - 如果需要保持数值的连续性和顺序关系，使用连续映射")
    print("   - 如果只需要离散分类，不需要保持数值顺序，使用分段编码")
    print("   - 超声检测中，深度值建议使用连续映射（保持深度顺序）")
    print("   - 通道号、编码值等离散特征可以使用分段编码")

if __name__ == "__main__":
    test_two_encoding_methods()
