#!/usr/bin/env python3
"""
测试B1数据格式处理
验证代码是否能够正确处理扁平列表格式的B1数据
"""

import json
import tempfile
from ultrasonic_data_converter import UltrasonicDataConverter

# 测试数据：修正后的B1格式（扁平列表）
test_data = [
    {
        "E": 2,
        "B1": [2.0, 136.0, 100.0, 257.0, 176.0, 8.0, 126.0, 100.0, 257.0, 176.0]
    },
    {
        "E": 5,
        "B1": [3.0, 100.0, 90.0, 200.0, 150.0, 6.0, 110.0, 85.0, 220.0, 160.0, 9.0, 120.0, 80.0, 240.0, 170.0]
    }
]

def test_b1_format():
    """测试B1数据格式处理"""
    print("=== 测试B1数据格式处理 ===")
    
    # 创建UltrasonicDataConverter实例
    converter = UltrasonicDataConverter()
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        temp_file_path = f.name
    
    try:
        # 加载数据
        data_list, config = converter.load_json_data(temp_file_path)
        print(f"加载了 {len(data_list)} 个数据点")
        
        # 提取B扫描数据
        e_data = converter.extract_bscan_data(data_list)
        print(f"提取了 {len(e_data)} 个E值的数据")
        
        # 打印处理结果
        for e, channels in e_data.items():
            print(f"\nE值 {e} 的通道数据:")
            for channel, data in channels.items():
                print(f"  通道 {channel}: 深度={data[0]}, 振幅={data[1]}, dB值={data[2]}")
        
        # 验证结果
        expected_channels = {
            2: {2: [136.0, 100.0, 257.0], 8: [126.0, 100.0, 257.0]},
            5: {3: [100.0, 90.0, 200.0], 6: [110.0, 85.0, 220.0], 9: [120.0, 80.0, 240.0]}
        }
        
        success = True
        for e, expected_ch in expected_channels.items():
            if e not in e_data:
                print(f"❌ E值 {e} 缺失")
                success = False
                continue
            
            actual_ch = e_data[e]
            for ch, expected_data in expected_ch.items():
                if ch not in actual_ch:
                    print(f"❌ E值 {e} 的通道 {ch} 缺失")
                    success = False
                    continue
                
                actual_data = actual_ch[ch]
                if actual_data != expected_data:
                    print(f"❌ E值 {e} 的通道 {ch} 数据不匹配")
                    print(f"    预期: {expected_data}")
                    print(f"    实际: {actual_data}")
                    success = False
                    continue
        
        if success:
            print("\n✅ 测试通过：B1数据格式处理正确")
        else:
            print("\n❌ 测试失败：B1数据格式处理有误")
            
        return success
    
    finally:
        # 删除临时文件
        import os
        os.unlink(temp_file_path)

if __name__ == "__main__":
    test_b1_format()
