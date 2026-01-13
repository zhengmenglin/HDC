#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试torchhd库导入
"""

import sys
import os

# 添加torchhd库所在目录到Python路径
sys.path.append(os.path.join('torchhd'))

try:
    import torchhd
    print(f"torchhd库导入成功，版本：{torchhd.__version__}")
    
    # 测试核心组件导入
    from torchhd import functional as F
    from torchhd.classifiers import Vanilla, OnlineHD
    from torchhd.embeddings import Level, Random
    
    print("核心组件导入成功")
    
    # 测试基本功能
    import torch
    n_dimensions = 10000
    
    # 创建随机超维向量
    rand_vec = torchhd.random(num_vectors=1, dimensions=n_dimensions)[0]
    print(f"随机超维向量创建成功，形状：{rand_vec.shape}")
    
    # 测试绑定和捆绑操作
    a = torchhd.random(num_vectors=1, dimensions=n_dimensions)[0]
    b = torchhd.random(num_vectors=1, dimensions=n_dimensions)[0]
    bound = torchhd.bind(a, b)
    bundled = torchhd.bundle(a, b)
    print(f"绑定和捆绑操作成功")
    
    # 测试相似度计算
    similarity = torchhd.cosine_similarity(a, b)
    print(f"相似度计算成功：{similarity.item()}")
    
    # 测试分类器创建
    n_features = 4  # 通道、编码、深度、幅度
    n_classes = 2   # 伤损、正常
    classifier = Vanilla(n_features, n_dimensions, n_classes)
    print(f"分类器创建成功：{classifier}")
    
    # 测试嵌入层
    level_embedding = Level(100, n_dimensions)
    print(f"嵌入层创建成功：{level_embedding}")
    
    print("torchhd库测试通过！")
    
except Exception as e:
    print(f"torchhd库导入或测试失败：{e}")
    import traceback
    traceback.print_exc()
