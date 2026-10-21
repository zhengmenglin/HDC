#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_viewer.py — 命中检查器（把查询图与命中图并排打开，快速人工检查）

数据源：<out>/检索结果.csv（scale_search.py 的产出，默认 D:/HDC/output/scale_full）
三种视图（按键 1/2/3 切换）：
  1 复核图     左=查询图 | 右=命中裁片，白框=逐标记对应（scale_search 已生成，默认）
  2 对齐对比   左=查询原图 | 右=命中图对应区间原图（1:1 未标注，供像素级核对）
  3 全画布定位 左=查询原图 | 右=整张命中 Step 图，红框标出命中区间

操作：
  ←/→ 翻页；1/2/3 切视图；滚轮缩放；中键拖拽平移；Enter=通过并下一个；
  发现问题→备注框写一句→「保存批注并下一个」；还有「跳到下一个未批注」。
检查记录增量写入 <out>/检查批注.csv；位置/过滤存 <out>/_检查进度.json（可随查随停）。

用法：
  python tools/check_viewer.py [--out D:/HDC/output/scale_full]
（或双击桌面「启动命中检查器.pyw」）
"""
import argparse
import csv
import json
import os
import sys
import time

import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageDraw, ImageTk

DEFAULT_OUT = r"D:/HDC/output/scale_merged"   # 合并总表（两批检索的最终结果）
DATA_PNG = r"D:/HDC/data/png"          # 查询图目录
DATA_CORPUS = r"D:/HDC/data/shengyu"   # 图库目录
ZOOMS = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5]
GAP = 10                               # 两图之间的留白（像素）

FILTERS = {
    "命中+疑似（默认）": lambda r: r["是否命中"] in ("是", "疑似"),
    "仅命中": lambda r: r["是否命中"] == "是",
    "仅疑似": lambda r: r["是否命中"] == "疑似",
    "仅未命中": lambda r: r["是否命中"] == "否",
    "全部": lambda r: True,
}


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class ViewerApp:
    def __init__(self, root, out_dir, test_mode=False):
        self.root = root
        self.out = out_dir
        self.test = test_mode
        self.dp = DATA_PNG
        self.dc = DATA_CORPUS

        with open(os.path.join(out_dir, "检索结果.csv"), encoding="utf-8-sig") as fh:
            self.rows_all = list(csv.DictReader(fh))
        self.notes = self._load_notes()
        prog = self._load_progress()

        self.rows = []
        self.idx = 0
        self.mode = 1          # 1 复核图 / 2 对齐 / 3 全画布
        self.zoom = None       # None=自动适应窗口
        self._ck = None        # 缓冲区 key
        self._img = None
        self._tkimg = None

        self._build_ui()
        self.filter_var.set(prog.get("filter", "命中+疑似（默认）"))
        self.apply_filter(initial_idx=prog.get("idx", 0))

    # ---------------- 数据 ----------------
    def _load_notes(self):
        p = os.path.join(self.out, "检查批注.csv")
        d = {}
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    d[r["查询图"]] = r
        return d

    def _save_notes(self):
        p = os.path.join(self.out, "检查批注.csv")
        cols = ["查询图", "判定", "结论", "说明", "时间"]
        with open(p, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in self.rows_all:          # 固定按结果表顺序，便于比对
                n = self.notes.get(r["查询图"])
                if n:
                    w.writerow({k: n.get(k, "") for k in cols})

    def _load_progress(self):
        p = os.path.join(self.out, "_检查进度.json")
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_progress(self):
        if self.test:
            return
        p = os.path.join(self.out, "_检查进度.json")
        try:
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"filter": self.filter_var.get(), "idx": self.idx}, fh)
        except Exception:
            pass

    def row_paths(self, r):
        """→ (查询图路径, 复核图路径|None, 命中图路径|None, shift|None)"""
        qp = os.path.join(self.dp, r["查询图"])
        rel = (r.get("复核图") or "").strip()
        ov = os.path.join(self.out, rel) if rel else None
        cf = None
        hit, img = (r.get("命中文件夹") or "").strip(), (r.get("命中图片") or "").strip()
        if hit and img:
            cf = os.path.join(self.dc, hit.replace("/", os.sep), img)
        shift = None
        s = (r.get("位移(px)") or "").strip()
        if s:
            try:
                shift = int(round(float(s)))
            except ValueError:
                shift = None
        return qp, ov, cf, shift

    # ---------------- 界面 ----------------
    def _build_ui(self):
        root = self.root
        root.title("命中检查器")
        root.geometry("1760x880")
        root.minsize(1100, 640)
        try:
            root.option_add("*Font", ("Microsoft YaHei UI", 10))
        except Exception:
            pass

        top = ttk.Frame(root, padding=(8, 6))
        top.pack(fill="x")
        ttk.Label(top, text="过滤：").pack(side="left")
        self.filter_var = tk.StringVar()
        cb = ttk.Combobox(top, textvariable=self.filter_var, values=list(FILTERS),
                          state="readonly", width=16)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())
        self.counter_var = tk.StringVar()
        ttk.Label(top, textvariable=self.counter_var).pack(side="left", padx=(14, 0))
        for txt, m in (("1 复核图", 1), ("2 对齐对比", 2), ("3 全画布", 3)):
            ttk.Button(top, text=txt, width=10,
                       command=lambda mm=m: self.set_mode(mm)).pack(side="left", padx=(12, 0))
        ttk.Button(top, text="适应窗口", command=self.fit_window).pack(side="left", padx=(12, 0))
        ttk.Button(top, text="打开复核图", command=self.open_overlay).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="打开输出目录", command=lambda: os.startfile(self.out)).pack(side="left", padx=(6, 0))

        info = ttk.Frame(root, padding=(8, 0))
        info.pack(fill="x")
        self.l1_var = tk.StringVar()
        self.l2_var = tk.StringVar()
        self.l3_var = tk.StringVar()
        ttk.Label(info, textvariable=self.l1_var,
                  font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        ttk.Label(info, textvariable=self.l2_var).pack(anchor="w")
        ttk.Label(info, textvariable=self.l3_var, foreground="#8a6d00").pack(anchor="w")
        self.hint_var = tk.StringVar()
        ttk.Label(info, textvariable=self.hint_var, foreground="#666").pack(anchor="w")

        mid = ttk.Frame(root, padding=(8, 4))
        mid.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(mid, background="#f4f4f4", highlightthickness=0)
        hb = ttk.Scrollbar(mid, orient="horizontal", command=self.canvas.xview)
        vb = ttk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hb.set, yscrollcommand=vb.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vb.grid(row=0, column=1, sticky="ns")
        hb.grid(row=1, column=0, sticky="ew")
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda e: self.render() if self.zoom is None else None)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        bot = ttk.Frame(root, padding=(8, 6))
        bot.pack(fill="x")
        ttk.Button(bot, text="◀ 上一张", command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(bot, text="下一张 ▶", command=lambda: self.step(1)).pack(side="left", padx=(6, 0))
        ttk.Button(bot, text="跳转到下一个未批注", command=self.next_unmarked).pack(side="left", padx=(12, 0))
        self.note_var = tk.StringVar()
        ent = ttk.Entry(bot, textvariable=self.note_var)
        ent.pack(side="left", fill="x", expand=True, padx=(14, 6))
        ent.bind("<Return>", lambda e: (self.mark("批注"), "break")[1])
        ttk.Button(bot, text="保存批注并下一个",
                   command=lambda: self.mark("批注")).pack(side="left")
        ttk.Button(bot, text="✓ 通过并下一个（Enter）",
                   command=lambda: self.mark("通过")).pack(side="left", padx=(6, 0))
        self.status_var = tk.StringVar()
        ttk.Label(bot, textvariable=self.status_var, foreground="#2a7").pack(side="left", padx=(10, 0))

        # 键盘导航要做"正在输入备注"守卫，否则往输入框打字会误触发翻页/切视图
        def _guard(fn):
            return lambda e: None if self._typing() else fn()

        root.bind("<Left>", _guard(lambda: self.step(-1)))
        root.bind("<Right>", _guard(lambda: self.step(1)))
        root.bind("<Home>", _guard(lambda: self.jump(0)))
        root.bind("<End>", _guard(lambda: self.jump(len(self.rows) - 1)))
        root.bind("<Return>", _guard(lambda: self.mark("通过")))
        for k, m in (("1", 1), ("2", 2), ("3", 3)):
            root.bind(k, _guard(lambda mm=m: self.set_mode(mm)))
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._update_title()

    def _update_title(self):
        self.root.title("命中检查器 — %s  （共 %d 行，已批注 %d）"
                        % (os.path.basename(self.out), len(self.rows_all), len(self.notes)))

    def _typing(self):
        """焦点在输入框时，键盘快捷键要让路（否则打字会误触发翻页/切视图）。"""
        return isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry))

    # ---------------- 过滤与导航 ----------------
    def apply_filter(self, initial_idx=0):
        pred = FILTERS.get(self.filter_var.get()) or FILTERS["命中+疑似（默认）"]
        cur = self.rows[self.idx]["查询图"] if self.rows else None
        self.rows = [r for r in self.rows_all if pred(r)]
        if not self.rows:
            self.idx = 0
        else:
            if initial_idx:
                self.idx = max(0, min(len(self.rows) - 1, initial_idx))
            else:
                self.idx = 0
                if cur:
                    for i, r in enumerate(self.rows):
                        if r["查询图"] == cur:
                            self.idx = i
                            break
        self.show()

    def jump(self, i):
        if self.rows:
            self.idx = max(0, min(len(self.rows) - 1, i))
            self.show()

    def step(self, d):
        if not self.rows:
            return
        self.idx = (self.idx + d) % len(self.rows)
        self.show()
        self._save_progress()

    def next_unmarked(self):
        n = len(self.rows)
        for k in range(1, n + 1):
            i = (self.idx + k) % n
            if self.rows[i]["查询图"] not in self.notes:
                self.idx = i
                self.show()
                self._save_progress()
                return
        messagebox.showinfo("完成", "当前过滤下全部行都已批注。")

    def set_mode(self, m):
        if self.mode == m:
            return
        self.mode = m
        self.show()

    def fit_window(self):
        self.zoom = None
        self.render()

    # ---------------- 渲染 ----------------
    def show(self):
        if not self.rows:
            self.canvas.delete("all")
            self.canvas.create_text(24, 24, anchor="nw", text="（该过滤下没有记录）", fill="#999")
            self.l1_var.set("")
            self.l2_var.set("")
            self.l3_var.set("")
            self.hint_var.set("")
            self.counter_var.set("0/0")
            return
        r = self.rows[self.idx]
        extra = "（二次判定找回）" if "二次判定" in (r.get("备注") or "") else ""
        self.l1_var.set("[%d/%d]  %s      判定：%s%s"
                        % (self.idx + 1, len(self.rows), r["查询图"], r["是否命中"], extra))
        parts = []
        for k, lab in (("覆盖率", "覆盖率"), ("NCC", "NCC"), ("位移(px)", "位移")):
            v = (r.get(k) or "").strip()
            if v:
                parts.append("%s %s" % (lab, v) + (" px" if k == "位移(px)" else ""))
        if (r.get("命中图片") or "").strip():
            parts.append("→ " + r["命中图片"].strip())
        if (r.get("命中文件夹") or "").strip():
            parts.append("组 " + r["命中文件夹"].strip())
        if (r.get("米区间(相对该图0点)") or "").strip():
            parts.append(r["米区间(相对该图0点)"].strip())
        if (r.get("层级") or "").strip():
            parts.append("层级 " + r["层级"].strip())
        self.l2_var.set("   ".join(parts))
        note = (r.get("备注") or "").strip()
        self.l3_var.set(("备注：" + note) if note else "")
        mynote = self.notes.get(r["查询图"])
        if mynote:
            self.counter_var.set("已查：%s（%s）" % (mynote.get("结论", ""), mynote.get("时间", "")[-8:]))
            self.note_var.set(mynote.get("说明", ""))
        else:
            self.counter_var.set("")
            self.note_var.set("")
        self.cache_key = None
        self._img = None
        self.render()

    def compose(self):
        r = self.rows[self.idx]
        qp, ov, cf, shift = self.row_paths(r)
        if self.mode == 1:
            if ov and os.path.exists(ov):
                return (Image.open(ov).convert("RGB"),
                        "复核图：左=查询图 | 右=命中裁片（白框=逐标记对应，由检索工具生成）")
            if os.path.exists(qp):
                return (Image.open(qp).convert("RGB"),
                        "无复核图（未命中行）：仅显示查询图")
            return None, "找不到图片文件"
        if not (cf and os.path.exists(cf) and shift is not None):
            if os.path.exists(qp):
                return (Image.open(qp).convert("RGB"),
                        "该行无命中信息（未命中），仅显示查询图")
            return None, "找不到图片文件"
        q = Image.open(qp).convert("RGB")
        c = Image.open(cf).convert("RGB")
        W, H = c.size
        x0, x1 = shift, shift + q.width
        if self.mode == 2:
            # 对齐裁片：为保证与查询图列坐标一一对应，两端不足处补白
            padl = max(0, -x0)
            cl, cr = max(0, x0), min(W, x1)
            piece = c.crop((cl, 0, cr, H)) if cr > cl else Image.new("RGB", (0, H))
            panel = Image.new("RGB", (padl + piece.width + max(0, x1 - W), H), (252, 252, 252))
            panel.paste(piece, (padl, 0))
            out = Image.new("RGB", (q.width + GAP + panel.width,
                                    max(q.height, panel.height)), (255, 255, 255))
            out.paste(q, (0, 0))
            out.paste(panel, (q.width + GAP, 0))
            return out, "对齐对比：左=查询原图 | 右=命中图对应区间原图（1:1，供像素级核对）"
        # mode 3 全画布：整张命中图 + 红框
        full = c.copy()
        d = ImageDraw.Draw(full)
        bx0, bx1 = max(0, x0), min(W - 1, x1 - 1)
        d.rectangle([bx0, 0, bx1, H - 1], outline=(255, 0, 0), width=4)
        out = Image.new("RGB", (q.width + GAP + W, max(q.height, H)), (255, 255, 255))
        out.paste(q, (0, 0))
        out.paste(full, (q.width + GAP, 0))
        return out, "全画布定位：左=查询原图 | 右=整张命中 Step 图（红框=命中区间）"

    def render(self):
        key = (self.idx, self.mode)
        if self._img is None or self._ck != key:
            try:
                self._img, hint = self.compose()
            except Exception as e:
                self._img, hint = None, "读图失败：%s: %s" % (type(e).__name__, e)
            self._ck = key
            self.hint_var.set((hint or "")
                              + "　·　←/→ 翻页 · 1/2/3 切视图 · 滚轮缩放 · Enter 通过")
        img = self._img
        c = self.canvas
        c.delete("all")
        if img is None:
            c.create_text(24, 24, anchor="nw", text="（无图可显示）", fill="#999")
            c.config(scrollregion=(0, 0, c.winfo_width(), c.winfo_height()))
            return
        vw = max(50, c.winfo_width() - 6)
        vh = max(50, c.winfo_height() - 6)
        if self.zoom is None:
            scale = max(0.1, min(3.0, min(vw / img.width, vh / img.height)))
        else:
            scale = self.zoom
        if abs(scale - 1.0) < 1e-9:
            disp = img
        else:
            rs = Image.LANCZOS if scale < 1 else Image.BICUBIC
            disp = img.resize((max(1, int(round(img.width * scale))),
                               max(1, int(round(img.height * scale)))), rs)
        self._tkimg = ImageTk.PhotoImage(disp)      # 必须持引用，否则被 GC
        c.create_image(0, 0, anchor="nw", image=self._tkimg)
        c.config(scrollregion=(0, 0, disp.width, disp.height))

    def on_wheel(self, e):
        if self._img is None:
            return
        cur = self.zoom
        if cur is None:
            c = self.canvas
            cur = max(0.1, min(3.0, min((c.winfo_width() - 6) / self._img.width,
                                        (c.winfo_height() - 6) / self._img.height)))
        i = min(range(len(ZOOMS)), key=lambda k: abs(ZOOMS[k] - cur))
        i = max(0, min(len(ZOOMS) - 1, i + (1 if e.delta > 0 else -1)))
        self.zoom = ZOOMS[i]
        self.render()

    # ---------------- 批注 ----------------
    def mark(self, verdict):
        if not self.rows:
            return
        r = self.rows[self.idx]
        note = self.note_var.get().strip()
        self.notes[r["查询图"]] = {"查询图": r["查询图"], "判定": r["是否命中"],
                                   "结论": verdict, "说明": note, "时间": _ts()}
        if not self.test:
            self._save_notes()
        self.note_var.set("")
        self.status_var.set("已记录：%s %s" % (r["查询图"][:20], verdict))
        self._update_title()
        self.step(1)

    def open_overlay(self):
        if not self.rows:
            return
        _, ov, _, _ = self.row_paths(self.rows[self.idx])
        if ov and os.path.exists(ov):
            os.startfile(ov)
        else:
            messagebox.showinfo("提示", "该行没有复核图（未命中行）。")

    def on_close(self):
        self._save_progress()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description="命中检查器")
    ap.add_argument("--out", default=DEFAULT_OUT, help="检索输出目录（含 检索结果.csv）")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(args.out, "检索结果.csv")):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("找不到数据", "该目录下没有 检索结果.csv：\n%s" % args.out)
        root.destroy()
        return 1
    root = tk.Tk()
    ViewerApp(root, args.out)
    root.mainloop()
    return 0


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = main()
    finally:
        # 与其它 GUI 工具一致：关窗后硬退出，任何环境下不留残余进程
        os._exit(_rc)
