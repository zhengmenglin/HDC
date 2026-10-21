#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_find_pattern.py — find_pattern.py 回归自测（合成裁片 + 真实样例）

用例全部可复现，真值已知：
  T1 真实样例（核伤导出图 vs data/2223）—— 回归既有结论：a有.png @1311
  T2 精确裁片 a有[600,1112)                —— 位移≈600，覆盖率 1.0
  T3 精确裁片 a无2[80,592)                 —— 位移≈80，覆盖率≥0.85
  T4 缩放裁片（T2 再 ×0.9）+ --scales auto —— 应自动找回（尺度≈1/0.9）
  T5 无标记裁片 a无[1200,1712)             —— 应报"未提取到标记"（exit 2）
  T6 单标记歧义 a有[1480,1680)             —— 真位移必须出现在候选里
  T7 相似图案区分 a无[386,898)             —— 必须命中 a无 本尊（而非 a无2）

输出：逐用例 PASS/FAIL 表；全部通过 exit 0。
注意：本脚本有独立输出目录 _out/_test，绝不触碰真机证据文件。
"""
import json
import os
import subprocess
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(ROOT, "find_pattern.py")
PY = sys.executable
DATA = r"D:/HDC/data"
QDIR = os.path.join(DATA, "2223")
REAL_Q = os.path.join(DATA, "核伤251003_0646_22232_20251110_084801",
                      "核伤251003_0646_22232_20251110_084801.png")
TMP = os.path.join(ROOT, "_out", "_test")
os.makedirs(TMP, exist_ok=True)


def load(path):
    im = Image.open(path).convert("RGBA")
    bgw = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bgw, im).convert("RGB")


def make_crop(src, x0, w=512, scale=1.0, name=None):
    im = load(src)
    c = im.crop((x0, 0, x0 + w, im.size[1]))
    if scale != 1.0:
        c = c.resize((int(round(c.size[0] * scale)), int(round(c.size[1] * scale))),
                     Image.LANCZOS)
    out = os.path.join(TMP, name)
    c.save(out)
    return out


def run_tool(query, extra, tag):
    outdir = os.path.join(TMP, "out_" + tag)
    cmd = [PY, "-X", "utf8", TOOL, "--query", query, "--corpus", QDIR,
           "--out", outdir, "--top", "3"] + extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    res = None
    jp = os.path.join(outdir, os.path.splitext(os.path.basename(query))[0], "result.json")
    if os.path.exists(jp):
        with open(jp, encoding="utf-8") as fp:
            res = json.load(fp)
    return r, res


RESULTS = []


def check(tag, cond, detail=""):
    RESULTS.append((tag, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {tag}" + (f"  [{detail}]" if detail else ""))


def top_hit(res):
    if not res:
        return None
    if res["accepted"]:
        name = res["accepted"][0]
        for h in res["hits"]:
            if h["file"] == name:
                return h
    return res["hits"][0] if res["hits"] else None


def bn(h):
    return os.path.basename(h["file"]) if h else "?"


def main():
    print("=" * 72)
    print("T1 真实样例（回归）：核伤导出图 → data/2223")
    r, res = run_tool(REAL_Q, [], "t1")
    h = top_hit(res)
    check("T1.1 top = a有.png", h and h["file"].endswith("a有.png"), bn(h))
    check("T1.2 位移≈1311", h and 1305 <= h["shift"] <= 1315, f"shift={h['shift'] if h else None}")
    check("T1.3 覆盖率=1.0", h and abs(h["cov"] - 1.0) < 1e-9, f"cov={h['cov'] if h else None}")
    check("T1.4 米区间≈1.28~3.07",
          h and h["m_range"] and abs(h["m_range"][0] - 1.28) < 0.06
          and abs(h["m_range"][1] - 3.07) < 0.06, f"m={h['m_range'] if h else None}")

    print("=" * 72)
    print("T2 精确裁片：a有.png [600,1112)")
    q2 = make_crop(os.path.join(QDIR, "a有.png"), 600, 512, 1.0, "crop_has_600.png")
    r, res = run_tool(q2, [], "t2")
    h = top_hit(res)
    check("T2.1 top = a有.png", h and h["file"].endswith("a有.png"), bn(h))
    check("T2.2 位移≈600", h and abs(h["shift"] - 600) <= 2, f"shift={h['shift'] if h else None}")
    check("T2.3 覆盖率≥0.99", h and h["cov"] >= 0.99, f"cov={h['cov'] if h else None}")
    check("T2.4 NCC≥0.90", h and h["ncc"] is not None and h["ncc"] >= 0.90,
          f"ncc={h['ncc'] if h else None}")

    print("=" * 72)
    print("T3 精确裁片：a无2.png [80,592)")
    q3 = make_crop(os.path.join(QDIR, "a无2.png"), 80, 512, 1.0, "crop_wu2_80.png")
    r, res = run_tool(q3, [], "t3")
    h = top_hit(res)
    check("T3.1 top = a无2.png", h and h["file"].endswith("a无2.png"), bn(h))
    check("T3.2 位移≈80", h and abs(h["shift"] - 80) <= 2, f"shift={h['shift'] if h else None}")
    check("T3.3 覆盖率≥0.85", h and h["cov"] >= 0.85, f"cov={h['cov'] if h else None}")

    print("=" * 72)
    print("T4 缩放裁片：a有.png [600,1112) ×0.9 + --scales auto")
    q4 = make_crop(os.path.join(QDIR, "a有.png"), 600, 512, 0.9, "crop_has_600_s090.png")
    r, res = run_tool(q4, ["--scales", "auto"], "t4")
    h = top_hit(res)
    check("T4.1 top = a有.png", h and h["file"].endswith("a有.png"), bn(h))
    check("T4.2 位移≈600", h and abs(h["shift"] - 600) <= 6, f"shift={h['shift'] if h else None}")
    check("T4.3 尺度≈1/0.9", h and abs(h["scale"] - 1.111) <= 0.03,
          f"scale={h['scale'] if h else None}")
    check("T4.4 覆盖率≥0.80", h and h["cov"] >= 0.80, f"cov={h['cov'] if h else None}")

    print("=" * 72)
    print("T5 无标记裁片：a无.png [1200,1712)（应安全退出）")
    q5 = make_crop(os.path.join(QDIR, "a无.png"), 1200, 512, 1.0, "crop_wu_1200.png")
    r, res = run_tool(q5, [], "t5")
    check("T5.1 exit=2", r.returncode == 2, f"rc={r.returncode}")
    check("T5.2 提示无标记", "未提取到" in r.stdout, "")

    print("=" * 72)
    print("T6 单标记歧义：a有.png [1480,1680)（宽 200，仅 1 个标记）")
    q6 = make_crop(os.path.join(QDIR, "a有.png"), 1480, 200, 1.0, "crop_has_1480_w200.png")
    r, res = run_tool(q6, [], "t6")
    found = bool(res) and any(
        h["file"].endswith("a有.png") and abs(h["shift"] - 1480) <= 2 and h["cov"] >= 0.99
        for h in res["hits"])
    check("T6.1 真位移(1480±2)在候选中", found,
          f"hits={[(bn(h), round(h['shift'])) for h in (res or {}).get('hits', [])]}")
    if res and res["hits"]:
        best = res["hits"][0]
        print(f"       （排序第一: {bn(best)} shift={best['shift']:.0f} "
              f"ncc={best['ncc'] if best['ncc'] is None else round(best['ncc'], 3)}）")

    print("=" * 72)
    print("T7 相似图案区分：a无.png [386,898)（同时有 a无2 的近似簇作干扰）")
    q7 = make_crop(os.path.join(QDIR, "a无.png"), 386, 512, 1.0, "crop_wu_386.png")
    r, res = run_tool(q7, [], "t7")
    h = top_hit(res)
    check("T7.1 top = a无.png", h and h["file"].endswith("a无.png"), bn(h))
    check("T7.2 位移≈386", h and abs(h["shift"] - 386) <= 3, f"shift={h['shift'] if h else None}")
    check("T7.3 覆盖率≥0.85", h and h["cov"] >= 0.85, f"cov={h['cov'] if h else None}")
    check("T7.4 NCC≥0.90", h and h["ncc"] is not None and h["ncc"] >= 0.90,
          f"ncc={h['ncc'] if h else None}")

    npass = sum(1 for _, ok in RESULTS if ok)
    print("=" * 72)
    print(f"结果：{npass}/{len(RESULTS)} 通过")
    if npass != len(RESULTS):
        print("失败项：" + "; ".join(t for t, ok in RESULTS if not ok))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
