# -*- coding: utf-8 -*-
"""gen_explain_figs.py — 生成通俗讲解图（真实数据）

fig_explain_marks.png  —— "把图翻译成清单"：原图 vs 只留被采纳的彩色标记
fig_explain_votes.png  —— "对账投票"：真实票数直方图（a有=尖峰；a无/a无2=散票）
输出到 tools/_out/explain/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

import find_pattern as fp

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "_out", "explain")
os.makedirs(OUT, exist_ok=True)

QDIR = r"D:/HDC/data/2223"
QREAL = (r"D:/HDC/data/核伤251003_0646_22232_20251110_084801/"
         r"核伤251003_0646_22232_20251110_084801.png")


def font(sz):
    for cand in (r"C:/Windows/Fonts/msyh.ttc", r"C:/Windows/Fonts/msyhbd.ttc",
                 r"C:/Windows/Fonts/simhei.ttf"):
        if os.path.exists(cand):
            return ImageFont.truetype(cand, sz)
    return ImageFont.load_default()


def mark_mask(a):
    """与工具同源的彩色标记像素掩码（同调色板 + 同容差）。"""
    f = a.astype(np.int32)
    H, W, _ = a.shape
    dmin = np.full((H, W), 10 ** 9, np.int32)
    for shades in fp.PALETTE.values():
        for sh in shades:
            dmin = np.minimum(dmin, np.abs(f - np.array(sh, np.int32)).sum(axis=2))
    return dmin < fp.CLASS_TOL


def only_marks(a, marks):
    """只显示"被算法采纳的标记"像素：背景涂白，其余为原色。

    注意用最终标记的包围盒过滤，而不是整张颜色掩码——否则文字抗锯齿的
    零星过渡色会混进来（那是被面积/行带规则否决掉的噪声）。
    """
    m = mark_mask(a)
    kept = np.zeros_like(m)
    for mk in marks:
        kept[mk["y0"]:mk["y1"], mk["x0"]:mk["x1"]] |= m[mk["y0"]:mk["y1"], mk["x0"]:mk["x1"]]
    out = np.full_like(a, 255)
    out[kept] = a[kept]
    return out


def draw_boxes(panel, marks):
    for m in marks:
        x0, y0 = max(0, m["x0"] - 3), max(0, m["y0"] - 3)
        x1, y1 = min(panel.shape[1] - 1, m["x1"] + 3), min(panel.shape[0] - 1, m["y1"] + 3)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (0, 0, 0), 4)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (255, 255, 255), 2)


def fig_marks():
    q = fp.load_rgb(QREAL)
    Q = fp.extract_marks(q)
    c = fp.load_rgb(os.path.join(QDIR, "a有.png"))
    x0 = 1311
    crop = np.ascontiguousarray(c[:, x0:x0 + 512])
    Cc = fp.extract_marks(crop)

    p1 = q.copy(); draw_boxes(p1, Q)
    p2 = only_marks(q, Q); draw_boxes(p2, Q)
    p3 = crop.copy(); draw_boxes(p3, Cc)
    p4 = only_marks(crop, Cc); draw_boxes(p4, Cc)

    PAD, GAP = 24, 24
    panel_w = 512
    cap_h = 44
    W = PAD * 2 + panel_w * 2 + GAP
    H = PAD + (panel_w + cap_h) * 2 + 8 + PAD
    canvas = Image.new("RGB", (W, H), (250, 250, 248))
    dr = ImageDraw.Draw(canvas)
    caps = [
        ("① 查询图原图（仪器导出的核伤片段）", (PAD, 24)),
        ("② 只认彩色标记 = 翻译成 4 行清单", (PAD + panel_w + GAP, 24)),
        ("③ a有.png 的对应段（1.28 ~ 3.07 米）", (PAD, PAD + panel_w + cap_h + 8)),
        ("④ 同一份清单在这里逐行对上", (PAD + panel_w + GAP, PAD + panel_w + cap_h + 8)),
    ]
    for panel, (text, (cx, cy)) in zip([p1, p2, p3, p4], caps):
        canvas.paste(Image.fromarray(panel), (cx, cy))
        dr.text((cx + 2, cy + panel_w + 6), text, font=font(21), fill=(45, 45, 45))
    out = os.path.join(OUT, "fig_explain_marks.png")
    canvas.save(out)
    print("saved", out, canvas.size)
    print("查询图清单:")
    for m in Q:
        print(f"   {m['cls']:6s} x={m['x0']}~{m['x1']} y={m['y0']}~{m['y1']}")
    print("a有 命中段清单:")
    for m in Cc:
        print(f"   {m['cls']:6s} x={m['x0']+x0}~{m['x1']+x0} y={m['y0']}~{m['y1']}")


def fig_votes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    q = fp.load_rgb(QREAL)
    Q = fp.extract_marks(q)
    files = [("a有.png", True), ("a无.png", False), ("a无2.png", False)]
    data = []
    for fn, is_hit in files:
        c = fp.load_rgb(os.path.join(QDIR, fn))
        hist = fp.vote_shifts(Q, fp.extract_marks(c), 1.0)
        xs = np.array(sorted(hist))
        ys = np.array([hist[x] for x in xs])
        data.append((fn, is_hit, xs, ys))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    stats = {}
    for ax, (fn, is_hit, xs, ys) in zip(axes, data):
        colors = ["#E24B4A" if (is_hit and 1305 <= x <= 1318) else "#B5D4F4" for x in xs]
        ax.bar(xs, ys, width=1.6, color=colors)
        ax.set_ylim(0, 4.6)
        ax.set_ylabel("票数")
        ax.set_title(f"{fn}   （{'命中' if is_hit else '未命中'}）",
                     color=("#0F6E56" if is_hit else "#5F5E5A"), fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        if is_hit:
            cl = int(ys[(xs >= 1305) & (xs <= 1318)].sum())
            ax.annotate(f"{cl} 张票挤在一起（±2px 之内）\n→ 所有标记都指向同一个位置",
                        xy=(1311, 2.1), xytext=(300, 3.6),
                        arrowprops=dict(arrowstyle="->", color="#A32D2D"),
                        fontsize=12, color="#A32D2D")
            stats[fn] = {"cluster": cl, "max": int(ys.max())}
        else:
            n2 = int((ys >= 2).sum())
            where = f"{n2} 个互不相同的位置" if n2 else "各个互不相同的位置"
            ax.annotate(f"最高一摞只有 {int(ys.max())} 票，\n散落在 {where} → 凑不成对",
                        xy=(float(xs[ys.argmax()]), float(ys.max()) + 0.1),
                        xytext=(300, 3.4),
                        arrowprops=dict(arrowstyle="->", color="#5F5E5A"),
                        fontsize=12, color="#5F5E5A")
            stats[fn] = {"max": int(ys.max()), "n2": n2}
    xmin = min(int(d[2].min()) for d in data) - 60
    xmax = max(int(d[2].max()) for d in data) + 60
    axes[-1].set_xlim(xmin, xmax)
    axes[-1].set_xlabel("位移（像素）：每个标记对量出来的“差多远”")
    fig.suptitle("同一张清单去三张图“对账”的投票结果（真实票数）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT, "fig_explain_votes.png")
    fig.savefig(out, dpi=110)
    print("saved", out)
    print("票数统计:", stats)


if __name__ == "__main__":
    fig_marks()
    fig_votes()
