# -*- coding: utf-8 -*-
"""双击本文件即可打开「命中检查器」（同时打开查询图与命中图，供人工检查）。

原理：本文件用"系统默认 Python"启动（自身不依赖任何第三方库），
然后把真正的工具交给下面 CANDIDATES 里找到的**解释器**运行——默认用
D:\HDC\.venv（PIL 等库在这里）。

换电脑 / 换虚拟环境时，只需修改下方的 CANDIDATES / TOOL_CANDIDATES 列表。
本文件放在「check_viewer.py」同目录或桌面均可。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 工具本体：先找同目录，再找 tools 目录（本文件被拷到桌面时用得上）
TOOL_CANDIDATES = [
    os.path.join(HERE, "check_viewer.py"),
    r"D:\HDC\tools\check_viewer.py",
]

# 解释器：按顺序取第一个存在的（虚拟环境优先，其次系统 Python 3.12）
CANDIDATES = [
    r"D:\HDC\.venv\Scripts\pythonw.exe",
    r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\pythonw.exe",
]


def _err(msg):
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("启动失败", msg)
        r.destroy()
    except Exception:
        print(msg)


def main():
    tool = next((t for t in TOOL_CANDIDATES if os.path.exists(t)), None)
    if tool is None:
        _err("没找到「check_viewer.py」。\n请把本文件放到工具同目录，或修改文件开头的 TOOL_CANDIDATES。")
        return 1
    runner = next((c for c in CANDIDATES if os.path.exists(c)), None)
    if runner is None:
        _err("没找到可用的 Python 解释器。\n请修改本文件开头的 CANDIDATES 列表。")
        return 1
    subprocess.Popen([runner, tool])
    return 0


if __name__ == "__main__":
    sys.exit(main())
