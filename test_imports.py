#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试导入是否正常
"""

import sys
import os

# 添加 torchhd 库路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

print("Testing imports...")

# 测试导入 torch
try:
    import torch
    print(f"✓ torch imported successfully: {torch.__version__}")
except Exception as e:
    print(f"✗ Failed to import torch: {e}")

# 测试导入 torchhd
try:
    import torchhd
    print(f"✓ torchhd imported successfully")
except Exception as e:
    print(f"✗ Failed to import torchhd: {e}")

# 测试导入其他模块
try:
    from hdc_point_level_classifier import (
        HDCPointLevelDataset, 
        HDCPointLevelModel, 
        WINDOW_SIZE, 
        NUM_CHANNELS, 
        NUM_TYPES, 
        CENTER_IDX
    )
    print("✓ hdc_point_level_classifier imported successfully")
except Exception as e:
    print(f"✗ Failed to import hdc_point_level_classifier: {e}")

try:
    from ultrasonic_data_converter import UltrasonicDataConverter
    print("✓ ultrasonic_data_converter imported successfully")
except Exception as e:
    print(f"✗ Failed to import ultrasonic_data_converter: {e}")

try:
    from config import config
    print("✓ config imported successfully")
except Exception as e:
    print(f"✗ Failed to import config: {e}")

print("Import test completed.")
