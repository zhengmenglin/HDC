#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
find_pattern.py — B 显图「图案定位/检索」工具 v0.2（2026-10-10）

用途：
    给定一张"目标图"（B 显截图 / 仪器导出片段，支持目录/通配符批量），
    在一批 B 显图里找出——它出现在哪个文件的哪一段
    （像素区间 + 米区间 + 命中标记清单 + 复核图）。

原理（详见 docs/数据筛选工具方案_20261010.md）：
    本系列 B 显图有固定"视觉语法"：固定调色板 + 固定网格框架 + 固定 px/m。
    可比的是「数据语义标记」（缺陷紫条 / 螺孔红点 / 绿标…）；刻度文字、
    里程文字、字体抗锯齿属于「渲染差异」，必须忽略。流水线：
        L1 标记提取 → L2 位移投票 → L3 残差图 NCC 复核

v0.2 相对 v0.1：
    · 语料目录递归扫描（自动跳过本工具自己的输出目录）
    · --query 支持 目录/通配符 ⇒ 批量检索，额外产出 batch_summary.csv
    · 每个查询产出 candidates.csv（utf-8-sig，Excel 直接打开）
    · --scales auto：粗扫 0.80~1.25，再对"差一口气"的文件做局部细化
    · --calib "px/m,x0" 手动标定兜底（自动标定失败时）

依赖：numpy / opencv-python / Pillow（真机 Python312 已具备）

用法示例：
    python find_pattern.py --query 目标.png --corpus D:/HDC/data/2223 --top 5
    python find_pattern.py --query "D:/HDC/data/核伤*.png" --corpus D:/HDC/data --scales auto
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VERSION = "0.2"

# ---------------- 家族常量（2026-10-10 实测取色） ----------------
BG = (255, 255, 192)  # 画布底色：墨迹图基准与裁片填充
PALETTE = {
    "purple": [(192, 0, 240), (240, 64, 240)],   # 核伤等主标记
    "brick":  [(192, 64, 0), (224, 144, 48), (240, 176, 80)],  # 螺孔/伤损点
    "green":  [(0, 192, 0), (0, 240, 0)],        # 绿标（接头等）
    "blue":   [(0, 0, 240)],                     # 蓝标
    "cyan":   [(0, 192, 240)],                   # 青标（实测 0646 语料与多张查询都有，
                                                 # 此前缺失导致 7 张查询被判"无标记"）
}
CLASS_TOL = 60   # 与调色板的曼哈顿距离容差（容纳抗锯齿过渡色）
MIN_AREA = 15    # 小于该面积的色块不可靠：实心标记均 >=20px，而稀疏点阵装饰
                 # （如 a有 里 y≈283 的点状线，9px）在缩放/压缩后颜色和形状会
                 # 严重漂移，留下"残端假标记"——实测把覆盖率从 4/4 拖到 4/6
EDGE_STRIP = 20  # 右缘窄条宽度（用于识别 A..I 字母列）
# 同类标记配对时的 宽比/高比 容差。单根标记查询的关键闸门：实测 0646 录里
# "宽柱(w≈50-80)"与"细条(w≈8-15)"混杂，光靠面积比拦不住（宽柱面积不一定大）
W_RATIO = (0.4, 2.5)
H_RATIO = (0.55, 1.8)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ---------------- 基础 IO ----------------
def load_rgb(path):
    """读为 RGB 数组。RGBA 先合成到白底，避免透明像素干扰取色。"""
    im = Image.open(path).convert("RGBA")
    bgw = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bgw, im).convert("RGB"))


# ---------------- L1 标记提取 ----------------
def extract_marks(a):
    """颜色分类 → 连通域 → 融合抗锯齿碎片 → 标记表。

    融合的原因：同一根紫条常被抗锯齿切成"帽+杆"两段，不融合会误判形状。
    """
    H, W, _ = a.shape
    f = a.astype(np.int32)
    marks = []
    for cls, shades in PALETTE.items():
        dmin = np.full((H, W), 10 ** 9, np.int32)
        for sh in shades:
            d = np.abs(f - np.array(sh, np.int32)).sum(axis=2)
            dmin = np.minimum(dmin, d)
        mask = (dmin < CLASS_TOL).astype(np.uint8)
        n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        comps = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < MIN_AREA:
                continue
            comps.append([x, y, x + w, y + h, area])
        comps.sort(key=lambda c: (c[0], c[1]))
        merged = []
        for c in comps:
            if merged:
                m = merged[-1]
                x_gap = max(c[0], m[0]) - min(c[2], m[2])
                y_gap = max(0, max(c[1], m[1]) - min(c[3], m[3]))
                if x_gap <= 2 and y_gap <= 6:
                    m[0] = min(m[0], c[0]); m[1] = min(m[1], c[1])
                    m[2] = max(m[2], c[2]); m[3] = max(m[3], c[3])
                    m[4] += c[4]
                    continue
            merged.append(list(c))
        for x0, y0, x1, y1, area in merged:
            if y1 <= 112:
                # 顶线（y=113）上方的彩色小点是半米刻度装饰，非数据
                continue
            if cls == "brick" and y0 >= 345 and y1 <= 400:
                # "倒车长度"红字的抗锯齿会被砖红分类误吃，该行带一律不认
                continue
            marks.append({
                "cls": cls, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "area": area,
                "cx": (x0 + x1) / 2.0,
            })
    # 右缘字母列（A..I）：成簇（>=3 个）才判为字母列并剔除；
    # 查询图可能是裁片，其右缘就是图案本体，不能误杀（实测踩过此坑）
    edge = [m for m in marks if m["x0"] > W - EDGE_STRIP]
    if len(edge) >= 3:
        marks = [m for m in marks if m["x0"] <= W - EDGE_STRIP]
    return marks


# ---------------- 刻度标定 ----------------
def calibrate(a):
    """返回 (px_per_m, x0_px)；自动标定失败时给 (None, None)。

    本系列约定：0 米刻度位于画布水平中心，刻度间距即 px/m。
    """
    H, W, _ = a.shape
    f = a.astype(np.int32)
    dark = (f[:, :, 0] < 140) & (f[:, :, 1] < 140) & (f[:, :, 2] < 140)
    rows = np.where(dark.sum(axis=1) > W * 0.5)[0]
    if not len(rows):
        return None, None
    t0 = int(rows[0])
    band = dark[t0 + 2:t0 + 16, :].sum(axis=0)
    cols = np.where(band >= 4)[0]
    cents = []
    for i in cols:
        if cents and i - cents[-1][-1] <= 2:
            cents[-1].append(i)
        else:
            cents.append([i])
    centers = [float(np.mean(c)) for c in cents]
    ds = [b - a2 for a2, b in zip(centers, centers[1:]) if 240 < b - a2 < 330]
    if len(ds) >= 2:
        s = float(np.median(ds))
    elif len(centers) >= 2:
        k = round((centers[-1] - centers[0]) / 285.7)  # 家族先验间距，仅兜底
        s = (centers[-1] - centers[0]) / k if k > 0 else None
    else:
        s = None
    if not s:
        return None, None
    x_mid = W / 2.0
    # 一致性筛选：只保留落在 x_mid + k*s 网格上的刻度（边框竖线会被剔除）
    inl = []
    for c in centers:
        k = round((c - x_mid) / s)
        if abs(c - (x_mid + k * s)) <= 4:
            inl.append((c, k))
    if len(inl) >= 4 and (inl[-1][1] - inl[0][1]) >= 3:
        s_fit = (inl[-1][0] - inl[0][0]) / (inl[-1][1] - inl[0][1])
        x0 = float(np.mean([c - s_fit * k for c, k in inl]))
        return s_fit, x0
    return s, None


# ---------------- L2 位移投票 ----------------
def y_overlap_ratio(qm, cm):
    inter = max(0, min(qm["y1"], cm["y1"]) - max(qm["y0"], cm["y0"]))
    hmin = max(1, min(qm["y1"] - qm["y0"], cm["y1"] - cm["y0"]))
    return inter / hmin


def pair_shape_ok(qm, cm):
    """配对形状闸门：宽/高比例都要接近（防止细条配宽柱）。"""
    qw = max(1, qm["x1"] - qm["x0"]); cw = max(1, cm["x1"] - cm["x0"])
    qh = max(1, qm["y1"] - qm["y0"]); ch = max(1, cm["y1"] - cm["y0"])
    return (W_RATIO[0] <= cw / qw <= W_RATIO[1]
            and H_RATIO[0] <= ch / qh <= H_RATIO[1])


def pair_shape_dist(qm, cm):
    """配对形状距离（越小越像），用于候选排序——单标记查询靠它挑 NCC 对象。"""
    qw = max(1, qm["x1"] - qm["x0"]); cw = max(1, cm["x1"] - cm["x0"])
    qh = max(1, qm["y1"] - qm["y0"]); ch = max(1, cm["y1"] - cm["y0"])
    return abs(ch / qh - 1) + abs(cw / qw - 1)


def scaled_mark(qm, s):
    if s == 1.0:
        return qm
    m = dict(qm)
    m["cx"] = qm["cx"] * s
    m["x0"] = qm["x0"] * s; m["x1"] = qm["x1"] * s
    m["y0"] = qm["y0"] * s; m["y1"] = qm["y1"] * s
    m["area"] = qm["area"] * s * s
    return m


def vote_shifts(Q, C, s):
    """标记两两配对（同类 + 纵向重叠 + 尺度相近）→ 位移直方图。"""
    hist = defaultdict(float)
    qp = [scaled_mark(q, s) for q in Q]
    for qm in qp:
        for cm in C:
            if cm["cls"] != qm["cls"]:
                continue
            if y_overlap_ratio(qm, cm) < 0.5:
                continue
            r = cm["area"] / max(1.0, qm["area"])
            if not (0.33 <= r <= 3.0):
                continue
            if not pair_shape_ok(qm, cm):
                continue
            hist[int(round(cm["cx"] - qm["cx"]))] += 1.0
    return hist


def find_peaks(hist, min_sep=24, max_n=6):
    if not hist:
        return []
    xs = np.arange(min(hist), max(hist) + 1)
    v = np.array([hist.get(int(x), 0.0) for x in xs], np.float64)
    k = np.array([1, 2, 3, 2, 1], np.float64); k /= k.sum()
    if len(v) >= len(k):
        sm = np.convolve(v, k, mode="same")
    else:
        # 短直方图：np.convolve(same) 会返回核长度导致 xs/sm 错位，
        # 先补零卷积再裁回原长度（实测踩过：单标记时会 IndexError 崩）
        pad = (len(k) - 1) // 2
        sm = np.convolve(np.pad(v, pad), k, mode="same")[pad:pad + len(v)]
    order = np.argsort(-sm)
    peaks = []
    for i in order:
        x = xs[i]
        if sm[i] <= 0:
            break
        if all(abs(x - p) >= min_sep for p, _ in peaks):
            peaks.append((int(x), float(sm[i])))
        if len(peaks) >= max_n:
            break
    return peaks


def coverage_at(Q, C, peak, s, tol=2.5):
    """在给定位移上逐标记找对应物，返回支持率与支持/落空清单。"""
    qp = [scaled_mark(q, s) for q in Q]
    sup, miss = [], []
    for qm in qp:
        best = None
        for cm in C:
            if cm["cls"] != qm["cls"]:
                continue
            if abs((cm["cx"] - qm["cx"]) - peak) > tol:
                continue
            if y_overlap_ratio(qm, cm) < 0.5:
                continue
            r = cm["area"] / max(1.0, qm["area"])
            if not (0.33 <= r <= 3.0):
                continue
            if not pair_shape_ok(qm, cm):
                continue
            d = abs((cm["cx"] - qm["cx"]) - peak)
            if best is None or d < best[0]:
                best = (d, cm)
        (sup if best else miss).append((qm, best[1] if best else None))
    return len(sup) / max(1, len(qp)), sup, miss


# ---------------- L3 残差图 NCC ----------------
def dmap(a, protect=None):
    """墨迹强度 × 饱和度权重，并清零"结构性/文字"行与右缘字母列。

    清零的 (0..123) 含刻度标签；(350..412) 含红基线与"倒车长度"红字 ——
    这些文字在查询图与语料图之间**口径不同**（查询图被重新标注过刻度），
    绝不能参与匹配。
    protect：extract_marks 的产物（可传缩放后的框）。带内的**合法标记**必须
    保留 —— 否则"轨底"类伤损的标记（恰好画在基线带上）会被整条误杀，残差图
    变全零（实测 2026-10-14：数据4/数据6 两张"轨底横向裂纹"因此无法像素比对，
    且全零模板会让 matchTemplate 输出常量 1.0，伪装成"完美匹配"）。
    """
    f = a.astype(np.float32)
    ink = np.sqrt(((f - np.array(BG, np.float32)) ** 2).sum(axis=2))
    mx = f.max(axis=2); mn = f.min(axis=2)
    satw = np.clip((mx - mn) / 80.0, 0.0, 1.0)
    D = ink * satw
    band = np.zeros(a.shape[:2], bool)
    band[0:123, :] = True
    band[350:412, :] = True
    if protect:
        for m in protect:
            y0 = int(max(0, m["y0"] - 2)); y1 = int(min(a.shape[0], m["y1"] + 2))
            x0 = int(max(0, m["x0"] - 2)); x1 = int(min(a.shape[1], m["x1"] + 2))
            if y1 > y0 and x1 > x0:
                band[y0:y1, x0:x1] = False
    D[band] = 0
    if a.shape[1] >= 1500:
        # 右缘字母列只在"全画布"渲染里存在；裁片查询图的右缘是真实图案，不能清
        D[:, -EDGE_STRIP:] = 0
    return D


def ncc_map(dq, dc):
    """一次 matchTemplate 生成整条位移的 NCC 曲线（避免每个候选峰重算）。"""
    Ht, Wt = dq.shape
    Hc, Wc = dc.shape
    if Ht > Hc or Wt > Wc:
        return None
    # 退化保护：全零/近常量残差图会让 matchTemplate 输出常量 1.0（实测），
    # 伪装成"像素完美匹配"被误收 —— 直接判为"无像素证据"。
    # （ncc_at_pos 侧有等价的 den<1e-9 守卫，两者语义保持一致）
    if float(dq.max()) <= 0.5 or float(dc.max()) <= 0.5:
        return None
    return cv2.matchTemplate(dc, dq, cv2.TM_CCOEFF_NORMED)


def ncc_lookup(res, shift):
    if res is None:
        return None
    xi = int(round(shift))
    if xi < 0 or xi >= res.shape[1]:
        return None
    return float(res[:, xi].max())


def ncc_at_pos(dq, dc, shift):
    """在指定位移处直接算归一化互相关（零均值），允许查询图部分越界。

    · 相比 matchTemplate 整幅扫快了约 10 倍（只算一个位置），批量检索的关键；
    · 越界部分裁掉再比——本数据集相邻 Step 有 888px 重叠，查询图常落在
      某步的右缘（shift 超出整图可放范围）或左缘（shift 为负）。
    """
    Hc, Wc = dc.shape
    if dq.shape[0] != Hc:
        return None
    x0 = int(round(shift))
    Wt = dq.shape[1]
    k0 = max(0, -x0)
    k1 = Wt - max(0, (x0 + Wt) - Wc)
    if k1 - k0 < 64:          # 有效重叠太窄，不给 NCC
        return None
    xs = x0 + k0
    if xs < 0 or xs + (k1 - k0) > Wc:
        return None
    a = dq[:, k0:k1].astype(np.float64).ravel()
    b = dc[:, xs:xs + (k1 - k0)].astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1e-9:
        return None
    return float((a * b).sum() / den)


# ---------------- 单图扫描（固定尺度集合） ----------------
def scan_scales(q_rgb, Q, c_rgb, C, dc, scales, debug=False):
    """对一张语料图跑一组尺度：投票 → 候选峰 → 覆盖率 → NCC。"""
    hits = []
    for s in scales:
        if s == 1.0:
            dq = dmap(q_rgb, Q)
        else:
            q_rs = cv2.resize(q_rgb,
                              (int(round(q_rgb.shape[1] * s)), int(round(q_rgb.shape[0] * s))),
                              interpolation=cv2.INTER_AREA)
            dq = dmap(q_rs, [{"y0": m["y0"] * s, "y1": m["y1"] * s,
                              "x0": m["x0"] * s, "x1": m["x1"] * s} for m in Q])
        hist = vote_shifts(Q, C, s)
        peaks = find_peaks(hist)
        if debug and peaks:
            print(f"    [scale {s:.3f}] 位移峰: " +
                  ", ".join(f"{p}({sc:.1f})" for p, sc in peaks[:6]))
        res = ncc_map(dq, dc)  # 每个尺度只算一次，峰们共用
        for p, _sc in peaks:
            cov, sup, miss = coverage_at(Q, C, p, s)
            if cov <= 0:
                continue
            v = ncc_lookup(res, p)
            if v is None:
                v = ncc_at_pos(dq, dc, p)  # 越界位移（近图缘）单独算
            hits.append({
                "shift": float(p), "scale": s, "cov": cov,
                "n_sup": len(sup), "total": len(sup) + len(miss),
                "ncc": v,
                "miss": [(m[0]["cls"], round(m[0]["cx"] * s, 1)) for m in miss],
            })
    return hits


def rank_hits(hits):
    """按 (覆盖率, NCC) 排序 + 近邻去重，保留前 5 个候选。"""
    hits.sort(key=lambda h: (-h["cov"], -(h["ncc"] or -1)))
    out = []
    for h in hits:
        if any(abs(h["shift"] - o["shift"]) <= 2 and abs(h["scale"] - o["scale"]) <= 0.006
               for o in out):
            continue
        out.append(h)
    return out[:5]


def finalize_hits(hits, q_rgb, pxm, x0):
    for h in hits:
        Wq = q_rgb.shape[1] * h["scale"]
        P = h["shift"]
        h["px_range"] = [round(P, 1), round(P + Wq, 1)]
        h["m_range"] = ([round((P - x0) / pxm, 3), round((P + Wq - x0) / pxm, 3)]
                        if (pxm and x0 is not None) else None)


# ---------------- 复核图 ----------------
def draw_overlay(q_rgb, Q, c_rgb, hit, out_png, full_png):
    """左=查询图（换算到语料尺度），右=语料命中段，白框逐标记对应。

    注意：缩放查询图必须先把查询重采样到语料尺度再并排，
    否则两侧高度不一致（实测：直接拼会在 np.concatenate 处崩）。
    """
    H, Wc, _ = c_rgb.shape
    s = hit["scale"]
    P = int(round(hit["shift"]))
    Wq = int(round(q_rgb.shape[1] * s))  # 查询图换算到语料尺度后的覆盖宽度
    x0, x1 = max(0, P), min(Wc, P + Wq)
    crop = c_rgb[:, x0:x1].copy()
    if abs(s - 1.0) < 1e-9:
        q_disp = q_rgb.copy()
    else:
        q_disp = cv2.resize(q_rgb,
                            (int(round(q_rgb.shape[1] * s)), int(round(q_rgb.shape[0] * s))),
                            interpolation=cv2.INTER_AREA)
    if q_disp.shape[0] < H:
        q_disp = np.concatenate([q_disp, np.full((H - q_disp.shape[0], q_disp.shape[1], 3),
                                                 255, np.uint8)], axis=0)
    elif q_disp.shape[0] > H:
        q_disp = q_disp[:H].copy()
    for panel, off in ((q_disp, -P), (crop, -x0)):
        for m in Q:
            ax0 = int(m["x0"] * s) + P + off
            ay0 = int(m["y0"] * s)
            ax1 = int(m["x1"] * s) + P + off
            ay1 = int(m["y1"] * s)
            cv2.rectangle(panel, (ax0, ay0), (ax1, ay1), (0, 0, 0), 4)
            cv2.rectangle(panel, (ax0, ay0), (ax1, ay1), (255, 255, 255), 2)
    full = c_rgb.copy()
    cv2.rectangle(full, (P, 0), (min(Wc - 1, P + Wq - 1), H - 1), (0, 0, 0), 3)
    gap = np.full((H, 10, 3), 0, np.uint8)
    w = max(q_disp.shape[1], crop.shape[1])
    panels = []
    for panel in (q_disp, crop):
        if panel.shape[1] < w:
            panel = np.concatenate([panel, np.full((panel.shape[0], w - panel.shape[1], 3),
                                                   255, np.uint8)], axis=1)
        panels.append(panel)
    side = np.concatenate([panels[0], gap, panels[1]], axis=1)
    Image.fromarray(side).save(out_png)
    Image.fromarray(full).save(full_png)


# ---------------- 语料/查询收集 ----------------
def collect_corpus(corpus, excludes):
    """语料收集：目录则递归；通配符则展开。excludes 里的目录/文件被跳过。"""
    ex_dirs = [os.path.abspath(e) for e in excludes if os.path.isdir(e)]
    ex_files = {os.path.abspath(e) for e in excludes if os.path.isfile(e)}

    def ok(p):
        p = os.path.abspath(p)
        if p in ex_files or not p.lower().endswith(IMG_EXTS):
            return False
        return not any(p.startswith(d + os.sep) for d in ex_dirs)

    if os.path.isdir(corpus):
        files = []
        for root, dirs, names in os.walk(corpus):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            files += [os.path.join(root, n) for n in names if ok(os.path.join(root, n))]
    else:
        files = [p for p in glob.glob(corpus, recursive=True) if ok(p)]
    return sorted(set(files))


def collect_queries(q):
    """查询收集：文件 / 目录（全部图片）/ 通配符。"""
    if os.path.isdir(q):
        out = []
        for root, dirs, names in os.walk(q):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            out += [os.path.join(root, n) for n in names if n.lower().endswith(IMG_EXTS)]
        return sorted(out)
    if any(ch in q for ch in "*?["):
        return sorted(p for p in glob.glob(q, recursive=True)
                      if p.lower().endswith(IMG_EXTS))
    return [q]


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description=f"B 显图图案定位/检索工具 v{VERSION}")
    ap.add_argument("--query", required=True, help="目标图（文件 / 目录 / 通配符）")
    ap.add_argument("--corpus", required=True, help="语料目录（递归）或通配符")
    ap.add_argument("--out", default=None, help="输出根目录")
    ap.add_argument("--top", type=int, default=5, help="保存复核图的命中数")
    ap.add_argument("--min-cov", type=float, default=0.75, help="判定命中的最低标记支持率")
    ap.add_argument("--min-ncc", type=float, default=0.4,
                    help="判定命中的最低残差 NCC（与覆盖率构成双证据，实测假命中≤0.20）")
    ap.add_argument("--scales", default="1.0", help='"1.0" / "0.9,1.0,1.1" / "auto"')
    ap.add_argument("--calib", default=None, help='手动标定 "px_per_m,x0_px"（自动失败时兜底）')
    ap.add_argument("--debug", action="store_true", help="打印标记表与位移峰明细")
    args = ap.parse_args()

    calib = None
    if args.calib:
        parts = [float(x) for x in args.calib.split(",")]
        calib = (parts[0], parts[1])

    outroot = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "_out", "find_pattern")
    os.makedirs(outroot, exist_ok=True)

    if args.scales == "auto":
        scales1 = [round(x, 3) for x in np.arange(0.80, 1.2501, 0.02)]
        auto_refine = True
    else:
        scales1 = [float(x) for x in args.scales.split(",")]
        auto_refine = False

    queries = collect_queries(args.query)
    if not queries:
        print(f"[err] 未找到查询图: {args.query}")
        return 2
    print(f"find_pattern v{VERSION} | 查询 {len(queries)} 张 | 语料 {args.corpus}"
          + (" | scales auto" if auto_refine else f" | scales {args.scales}"))

    batch_rows, used_dirs, n_empty = [], set(), 0
    for qp in queries:
        q_abs = os.path.abspath(qp)
        q_rgb = load_rgb(q_abs)
        Q = extract_marks(q_rgb)
        qpxm, qx0 = calibrate(q_rgb)
        stem = os.path.splitext(os.path.basename(q_abs))[0]
        print(f"\n== 查询: {os.path.basename(q_abs)}  ({q_rgb.shape[1]}x{q_rgb.shape[0]}"
              f", {len(Q)} 个标记) ==")
        if args.debug:
            print(f"    px/m={qpxm} x0={qx0}")
            for m in Q:
                print(f"    {m['cls']:6s} cx={m['cx']:7.1f} x=[{m['x0']},{m['x1']}) "
                      f"y=[{m['y0']},{m['y1']}] area={m['area']}")
        if not Q:
            print("  [skip] 查询图中未提取到任何彩色标记")
            n_empty += 1
            batch_rows.append({"query": q_abs, "accepted": False, "top_file": "无标记"})
            continue

        files = collect_corpus(args.corpus, [q_abs, outroot])
        print(f"  [corpus] {len(files)} 个文件")

        odir = os.path.join(outroot, stem)
        if odir in used_dirs:
            i = 2
            while f"{odir}_{i}" in used_dirs:
                i += 1
            odir = f"{odir}_{i}"
        used_dirs.add(odir)
        os.makedirs(odir, exist_ok=True)

        all_hits, cache = [], {}
        for f in files:
            c_rgb = load_rgb(f)
            C = extract_marks(c_rgb)
            pxm, x0 = calibrate(c_rgb)
            if calib:
                pxm, x0 = calib
            dc = dmap(c_rgb, C)
            hits = rank_hits(scan_scales(q_rgb, Q, c_rgb, C, dc, scales1, args.debug))
            if auto_refine and hits and 0.30 <= hits[0]["cov"] < args.min_cov:
                b = hits[0]
                ref = [round(b["scale"] + d, 4)
                       for d in np.arange(-0.018, 0.0181, 0.006)]
                ref = [s for s in ref if s not in set(scales1) and 0.3 < s < 2.5]
                if ref:
                    if args.debug:
                        print(f"    [refine] {os.path.basename(f)} "
                              f"scale≈{b['scale']:.3f} 附近细化 {len(ref)} 档")
                    hits = rank_hits(hits + scan_scales(q_rgb, Q, c_rgb, C, dc, ref))
            finalize_hits(hits, q_rgb, pxm, x0)
            cache[f] = (c_rgb, C)
            all_hits += [dict(h, file=f) for h in hits]

        all_hits.sort(key=lambda h: (-h["cov"], -(h["ncc"] or -1)))
        # 双证据：覆盖率（标记级）+ 残差 NCC（像素级）。
        # NCC 没算出来的候选一律不放行（宁缺勿假）——实测录里同形标记极多，
        # "None 放行"会让没复核过的候选混进结果
        accepted = [h for h in all_hits
                    if h["cov"] >= args.min_cov
                    and h["ncc"] is not None and h["ncc"] >= args.min_ncc]

        print("  == 候选（按支持率排序） ==")
        print(f"  {'文件':<22s} {'位移(px)':>9s} {'尺度':>6s} {'覆盖率':>6s} "
              f"{'NCC':>6s}  覆盖区间(px)             米区间")
        for h in all_hits[:12]:
            nm = os.path.basename(h["file"])
            ncc = f"{h['ncc']:.3f}" if h["ncc"] is not None else "  -  "
            rng = f"[{h['px_range'][0]:.0f},{h['px_range'][1]:.0f})"
            mr = (f"{h['m_range'][0]:.2f}~{h['m_range'][1]:.2f}m" if h["m_range"] else "-")
            print(f"  {nm:<22s} {h['shift']:>9.1f} {h['scale']:>6.3f} {h['cov']:>6.2f} "
                  f"{ncc:>6s}  {rng:<24s} {mr}")

        for i, h in enumerate(accepted[:args.top], 1):
            c_rgb, C = cache[h["file"]]
            fstem = os.path.splitext(os.path.basename(h["file"]))[0]
            png = os.path.join(odir, f"rank{i}_{fstem}_x{int(round(h['shift']))}.png")
            fullp = os.path.join(odir, f"rank{i}_{fstem}_x{int(round(h['shift']))}_full.png")
            draw_overlay(q_rgb, Q, c_rgb, h, png, fullp)
            h["overlay"] = png

        if accepted:
            t = accepted[0]
            mr = f" = {t['m_range'][0]:.2f}~{t['m_range'][1]:.2f}m" if t["m_range"] else ""
            print(f"  [TOP] {os.path.basename(t['file'])}  shift={t['shift']:.1f}px  "
                  f"覆盖 {t['px_range'][0]:.0f}~{t['px_range'][1]:.0f}px{mr}  "
                  f"支持 {t['n_sup']}/{t['total']}")
        else:
            print(f"  [未命中] 全部候选低于覆盖率阈值 {args.min_cov}"
                  + ("" if auto_refine else "（可试 --scales auto）"))

        with open(os.path.join(odir, "result.json"), "w", encoding="utf-8") as fp:
            json.dump({"tool": f"find_pattern v{VERSION}", "query": q_abs,
                       "corpus": args.corpus, "n_marks": len(Q),
                       "scales": scales1, "hits": all_hits,
                       "accepted": [h["file"] for h in accepted]},
                      fp, ensure_ascii=False, indent=2)
        cols = ["file", "shift", "scale", "cov", "n_sup", "total", "ncc",
                "px_range", "m_range", "miss"]
        write_csv(os.path.join(odir, "candidates.csv"), all_hits, cols)

        t = accepted[0] if accepted else (all_hits[0] if all_hits else None)
        batch_rows.append({
            "query": q_abs,
            "accepted": bool(accepted),
            "top_file": os.path.basename(t["file"]) if t else "",
            "shift_px": t["shift"] if t else "",
            "scale": t["scale"] if t else "",
            "cov": t["cov"] if t else "",
            "ncc": (t["ncc"] if t and t["ncc"] is not None else ""),
            "px_range": str(t["px_range"]) if t else "",
            "m_range": (str(t["m_range"]) if t and t["m_range"] else ""),
        })

    bcols = ["query", "accepted", "top_file", "shift_px", "scale", "cov", "ncc",
             "px_range", "m_range"]
    write_csv(os.path.join(outroot, "batch_summary.csv"), batch_rows, bcols)
    print(f"\n[out] {outroot}  （batch_summary.csv / 各查询子目录）")
    return 2 if n_empty == len(queries) else 0


if __name__ == "__main__":
    sys.exit(main())
