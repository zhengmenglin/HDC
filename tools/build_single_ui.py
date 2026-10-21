#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_single_ui.py — 组装单文件分发版「图案检索工具.py」

把三个源文件拼成一个自包含脚本（发给同事只需这一个文件）：
  1) 依赖自检（缺库时弹窗给出 pip 命令）
  2) find_pattern.py        核心算法（裁掉 __main__ 与 CLI 入口）
  3) find_pattern_batch.py  批量检索（裁掉 __main__；把对核心的 import 改为自引用）
  4) find_pattern_ui.py     图形界面（保留 __main__，作为单文件唯一入口）

算法更新后重跑本脚本即可再生成；请勿手改生成物。
"""
import datetime
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
SRC_CORE = BASE / "find_pattern.py"
SRC_BATCH = BASE / "find_pattern_batch.py"
SRC_UI = BASE / "find_pattern_ui.py"
OUT = BASE / "dist" / "图案检索工具.py"

# find_pattern_batch.py 中这两行在单文件版里要替换为"自引用"（核心已内联在同一模块）
IMPORT_BLOCK = ("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
                "import find_pattern as fp  # noqa: E402")


def cut_main(src):
    """裁掉 `if __name__ ==` 及其后的入口块（单文件只需要一个入口 = 界面）。"""
    i = src.find("\nif __name__ ==")
    return (src[:i].rstrip() if i >= 0 else src.rstrip())


DEP_CHECK = '''# ==================== 依赖自检（缺库时弹窗给出安装命令） ====================
import sys as _sys


def _check_deps():
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python"),
                     ("PIL", "pillow"), ("openpyxl", "openpyxl")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    cmd = _sys.executable + " -m pip install " + " ".join(missing)
    text = ("缺少以下运行库：\\n\\n  " + "\\n  ".join(missing)
            + "\\n\\n在 cmd 窗口执行下面这行命令安装后重试：\\n\\n" + cmd)
    print(text)
    try:
        import tkinter as _tkm
        from tkinter import messagebox as _mb
        _root = _tkm.Tk()
        _root.withdraw()
        _mb.showerror("图案检索工具 - 缺少依赖库", text)
        _root.destroy()
    except Exception:
        pass
    _sys.exit(1)


_check_deps()


'''


def main():
    core = cut_main(SRC_CORE.read_text(encoding="utf-8"))
    batch = cut_main(SRC_BATCH.read_text(encoding="utf-8"))
    ui = SRC_UI.read_text(encoding="utf-8")

    if IMPORT_BLOCK not in batch:
        print("[err] find_pattern_batch.py 的导入块与预期不符，请同步更新本脚本")
        return 1
    batch = batch.replace(
        IMPORT_BLOCK,
        "# 单文件版（由 build_single_ui.py 生成）：核心算法已内联在本模块，直接自引用\n"
        "fp = sys.modules[__name__]")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head = ('#!/usr/bin/env python\n'
            '# -*- coding: utf-8 -*-\n'
            '"""\n'
            '图案检索工具（单文件版） — 自动生成，请勿手改本文件\n'
            '\n'
            '用途：\n'
            '    在一批 B 显图片（语料）中查找指定的图片（查询）出现在哪个文件、哪一段，\n'
            '    用于超声探伤数据筛选。输出 Excel/CSV 总表 + 每张命中的复核图。\n'
            '\n'
            '给同事的使用方法：\n'
            '    1) 装好 Python 3（Windows 安装时勾选 Add Python to PATH）后，安装依赖：\n'
            '           pip install opencv-python numpy pillow openpyxl\n'
            '    2) 双击本文件（或在命令行执行：python 图案检索工具.py）\n'
            '    3) 界面里选『查询图片目录』和『语料图片目录』，点『开始检索』\n'
            '    4) 结果自动保存到输出目录；建议先用『高级设置 → 限制张数 = 5』试跑\n'
            '\n'
            '生成源：tools/find_pattern.py + tools/find_pattern_batch.py + tools/find_pattern_ui.py\n'
            '生成器：tools/build_single_ui.py（算法更新后重跑生成器，勿手改本文件）\n'
            f'生成时间：{ts}\n'
            '"""\n'
            '\n\n')

    parts = [head, DEP_CHECK, core,
             "\n\n# ================= 批量检索（内联自 find_pattern_batch.py） =================\n\n",
             batch,
             "\n\n# ================= 图形界面（内联自 find_pattern_ui.py） =================\n\n",
             ui, "\n"]
    text = "".join(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"[ok] 已生成 {OUT}")
    print(f"     {OUT.stat().st_size} 字节，{len(text.splitlines())} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
