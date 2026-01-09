import numpy as np
from HDC import HyperdimensionalEncoder

def test_continuous_mapping_similarity():
    """测试连续映射的相似度问题"""
    
    print("=" * 60)
    print("测试连续映射的相似度问题")
    print("=" * 60)
    
    # 测试不同的 num_levels 值
    test_cases = [
        (0, 2, "只有最小值和最大值（2个级别）"),
        (0, 3, "3个级别"),
        (0, 5, "5个级别"),
        (0, 10, "10个级别"),
        (0, 11, "11个级别"),
        (0, 101, "101个级别"),
    ]
    
    for min_val, num_levels, description in test_cases:
        max_val = min_val + num_levels - 1
        print(f"\n测试用例: {description}")
        print(f"  参数: min_val={min_val}, max_val={max_val}, num_levels={num_levels}")
        
        encoder = HyperdimensionalEncoder(d=10000, seed=42)
        encoder.continuous_mapping(min_val, max_val, num_levels, 'test')
        
        # 获取最小值和最大值的向量
        min_vector = encoder.item_memory['test'][min_val]
        max_vector = encoder.item_memory['test'][max_val]
        
        # 计算相似度
        hamming_sim = encoder.hamming_similarity(min_vector, max_vector)
        cosine_sim = encoder.cosine_similarity(min_vector, max_vector)
        
        print(f"  汉明相似度: {hamming_sim:.4f}")
        print(f"  余弦相似度: {cosine_sim:.4f}")
        
        # 计算翻转位数
        flip_bits = encoder.d // (2 * (num_levels - 1))
        print(f"  每次翻转位数: {flip_bits}")
        
        # 计算总翻转位数（从最小值到最大值）
        total_flips = flip_bits * (num_levels - 1)
        print(f"  总翻转位数: {total_flips}")
        
        # 理论相似度
        theoretical_sim = 1 - (total_flips / encoder.d)
        print(f"  理论汉明相似度: {theoretical_sim:.4f}")
        
        # 检查是否完全翻转
        if total_flips >= encoder.d:
            print(f"  ⚠️  警告: 总翻转位数 >= 维度，可能导致相似度为0或接近0")
    
    print("\n" + "=" * 60)
    print("分析结论")
    print("=" * 60)
    print("1. 当 num_levels=2 时:")
    print("   - flip_bits = d // (2 * (2-1)) = d // 2")
    print("   - 总翻转位数 = d // 2")
    print("   - 相似度 = 1 - (d//2)/d ≈ 0.5")
    print("   - 这是最小值和最大值相似度为0.5的情况")
    print()
    print("2. 当 num_levels > 2 时:")
    print("   - flip_bits = d // (2 * (num_levels-1))")
    print("   - 总翻转位数 = flip_bits * (num_levels-1) = d // 2")
    print("   - 但由于随机选择翻转位，可能翻转同一位置多次")
    print("   - 实际翻转位数可能接近 d，导致相似度接近0")
    print()
    print("3. 当 num_levels 较大时（如101）:")
    print("   - flip_bits = d // (2 * 100) = d // 200")
    print("   - 总翻转位数 = (d // 200) * 100 = d // 2")
    print("   - 但累积翻转可能导致大部分位被翻转")
    print("   - 相似度可能接近0")
    print()
    print("4. 根本原因:")
    print("   - 连续映射采用逐步翻转的方式")
    print("   - 翻转位是随机选择的，可能重复翻转")
    print("   - 从最小值到最大值的累积翻转可能导致向量完全不同")

def test_multivariate_encoding_issues():
    """测试多元编码的问题"""
    
    print("\n" + "=" * 60)
    print("多元编码存在的问题")
    print("=" * 60)
    
    print("\n1. 维度爆炸问题:")
    print("   - 假设有4个特征：通道(9种)、编码(100种)、深度(301种)、幅度(101种)")
    print("   - 多元型编码需要处理所有可能的特征组合")
    print("   - 组合数量 = 9 × 100 × 301 × 101 = 27,399,900")
    print("   - 超维向量维度需要大幅增加才能容纳所有组合")
    print("   - 计算复杂度呈指数增长")
    
    print("\n2. 数据稀疏性问题:")
    print("   - 实际超声数据中，大部分特征组合不存在")
    print("   - 例如：通道1 + 编码50 + 深度200 + 幅度80 可能从未出现")
    print("   - 导致编码向量稀疏，影响分类效果")
    print("   - 浪费大量存储空间和计算资源")
    
    print("\n3. 过拟合风险:")
    print("   - 多元型编码容易学习到数据中的噪声")
    print("   - 超声检测数据存在大量正常结构")
    print("   - 过拟合会导致误判正常结构为伤损")
    print("   - 泛化能力差，难以适应新的检测场景")
    
    print("\n4. 可解释性差:")
    print("   - 多元型编码的特征组合复杂")
    print("   - 难以理解哪些特征对分类结果影响最大")
    print("   - 调试困难，现场应用不友好")
    
    print("\n5. 计算效率低:")
    print("   - 需要生成和存储大量特征组合向量")
    print("   - 编码和解码过程复杂")
    print("   - 实时检测的响应时间难以保证")
    
    print("\n6. 内存占用大:")
    print("   - 需要存储所有可能的特征组合")
    print("   - 对于上述示例，需要存储约2700万个向量")
    print("   - 每个向量10000维，约需要270GB内存")
    
    print("\n7. 训练样本需求大:")
    print("   - 多元型编码需要大量训练样本覆盖各种特征组合")
    print("   - 实际超声检测数据有限")
    print("   - 训练不充分会导致分类效果差")
    
    print("\n8. 泛化能力弱:")
    print("   - 多元型编码对训练数据依赖性强")
    print("   - 难以适应新的检测场景或设备参数变化")
    print("   - 需要重新训练才能适应新情况")

if __name__ == "__main__":
    test_continuous_mapping_similarity()
    test_multivariate_encoding_issues()
