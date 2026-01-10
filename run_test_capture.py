#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys

# 运行测试脚本并捕获输出
result = subprocess.run([sys.executable, 'hdc_test.py'], capture_output=True, text=True, cwd='d:\\HDC')

# 打印输出到控制台
print('STDOUT:')
print(result.stdout)
print('\nSTDERR:')
print(result.stderr)

# 写入输出到文件
with open('test_output_captured.txt', 'w', encoding='utf-8') as f:
    f.write('STDOUT:\n')
    f.write(result.stdout)
    f.write('\n\nSTDERR:\n')
    f.write(result.stderr)

print('\n输出已保存到 test_output_captured.txt')
