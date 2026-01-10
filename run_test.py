#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本，直接运行hdc_test.py并输出结果
"""
import subprocess
import sys

# 运行hdc_test.py并捕获输出
print("开始运行hdc_test.py...")
result = subprocess.run([sys.executable, "hdc_test.py"], 
                       capture_output=True, 
                       text=True, 
                       cwd=".")

# 输出结果
print("stdout:")
print(result.stdout)

if result.stderr:
    print("stderr:")
    print(result.stderr)

print(f"返回码: {result.returncode}")
