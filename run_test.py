#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import os

# 确保当前目录正确
print(f"当前工作目录: {os.getcwd()}")

# 直接运行测试脚本，查看输出
print("运行测试脚本...")
result = subprocess.run(['python', 'hdc_test.py'], capture_output=True, text=True, cwd='d:\\HDC')

# 打印结果到控制台
print('\n=== STDOUT ===')
print(result.stdout)
print('\n=== STDERR ===')
print(result.stderr)

# 写入输出到文件
file_path = 'd:\\HDC\\test_results.txt'
try:
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('STDOUT:\n')
        f.write(result.stdout)
        f.write('\n\nSTDERR:\n')
        f.write(result.stderr)
    print(f'\n测试结果已保存到 {file_path}')
    print(f'文件大小: {os.path.getsize(file_path)} 字节')
except Exception as e:
    print(f'\n保存文件失败: {e}')
