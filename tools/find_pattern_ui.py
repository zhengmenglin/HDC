#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
find_pattern_ui.py — 图案检索工具（图形界面）

两种运行形态（同一份代码）：
  · 开发态：与 find_pattern.py / find_pattern_batch.py 同目录运行，直接导入模块；
  · 单文件态：tools/build_single_ui.py 把上述两个模块内联进本文件，生成
    「图案检索工具.py」——发给同事只需这一个文件。

用法：双击运行，或 `python find_pattern_ui.py`。
"""
import hashlib
import json
import os
import queue
import sys
import threading
import time

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 单文件态（由 build_single_ui.py 打包）：run_batch / BatchParams 已内联在本模块
_SELF = sys.modules[__name__]
if not hasattr(_SELF, "run_batch"):
    # 开发态：从同目录模块导入
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from find_pattern_batch import BatchParams, run_batch  # noqa: E402

APP_TITLE = "图案检索工具 — B显图片定位（超声探伤数据筛选）"
CFG_PATH = os.path.join(os.path.expanduser("~"), ".find_pattern_ui.json")


def _enable_dpi():
    """Windows 高分屏下让界面不发虚（失败不影响功能）。"""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class PatternFinderApp:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.last_out = ""
        self._build_ui()
        self._load_cfg()
        self._welcome()
        root.after(100, self._poll)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- 界面搭建 ----------------
    def _build_ui(self):
        r = self.root
        r.title(APP_TITLE)
        r.geometry("920x660")
        r.minsize(820, 560)

        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Hint.TLabel", foreground="#666666")
        style.configure("Run.TButton", font=("Microsoft YaHei UI", 10, "bold"))

        main = ttk.Frame(r, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        r.rowconfigure(0, weight=1)
        r.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(8, weight=1)  # 日志区占用剩余高度

        ttk.Label(main, text="① 选两个目录　② 点『开始检索』，结果自动保存到输出目录",
                  style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(main,
                  text="查询图片目录 = 要找的图（如 data\\轨头伤损）；语料目录 = 被搜索的 B 显图库（如 data\\0646）",
                  style="Hint.TLabel").grid(row=1, column=0, columnspan=3,
                                            sticky="w", pady=(2, 8))

        self.var_q = tk.StringVar()
        self.var_c = tk.StringVar()
        self.var_o = tk.StringVar()
        for i, (text, var) in enumerate((("查询图片目录", self.var_q),
                                         ("语料图片目录", self.var_c),
                                         ("输出目录", self.var_o))):
            row = 2 + i
            ttk.Label(main, text=text).grid(row=row, column=0, sticky="e",
                                            padx=(0, 6), pady=3)
            ttk.Entry(main, textvariable=var).grid(row=row, column=1,
                                                   sticky="ew", pady=3)
            ttk.Button(main, text="浏览…", width=8,
                       command=lambda v=var, t=text: self._pick_dir(v, t)
                       ).grid(row=row, column=2, padx=(6, 0), pady=3)

        bar = ttk.Frame(main)
        bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self.btn_run = ttk.Button(bar, text="开始检索", style="Run.TButton",
                                  command=self._start)
        self.btn_run.pack(side="left", ipadx=18)
        self.btn_stop = ttk.Button(bar, text="停止", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.btn_open = ttk.Button(bar, text="打开结果文件夹",
                                   command=self._open_out, state="disabled")
        self.btn_open.pack(side="left", padx=6)
        self.adv_show = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="高级设置", variable=self.adv_show,
                        command=self._toggle_adv).pack(side="right")

        self.adv = ttk.LabelFrame(main, text="高级设置（一般不用动；修改后作用于本次检索）",
                                  padding=8)
        self.adv.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        self.adv.grid_remove()
        self.var_mincov = tk.StringVar(value="0.75")
        self.var_minncc = tk.StringVar(value="0.40")
        self.var_strong = tk.StringVar(value="0.70")
        self.var_top = tk.StringVar(value="2")
        self.var_limit = tk.StringVar(value="0")
        specs = (("判定覆盖率≥", self.var_mincov),
                 ("判定 NCC≥", self.var_minncc),
                 ("强命中 NCC≥", self.var_strong),
                 ("复核图/张", self.var_top),
                 ("限制张数", self.var_limit))
        for i, (text, var) in enumerate(specs):
            rr, cc = divmod(i, 3)
            cell = ttk.Frame(self.adv)
            cell.grid(row=rr, column=cc, sticky="w", padx=6, pady=2)
            ttk.Label(cell, text=text).pack(side="left")
            ttk.Entry(cell, textvariable=var, width=6).pack(side="left", padx=4)
        ttk.Label(self.adv, text="（限制张数=0 表示全部；试跑建议设 5）",
                  style="Hint.TLabel").grid(row=1, column=2, sticky="w", padx=6)

        self.bar = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.bar.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 6))

        logf = ttk.Frame(main)
        logf.grid(row=8, column=0, columnspan=3, sticky="nsew")
        logf.rowconfigure(0, weight=1)
        logf.columnconfigure(0, weight=1)
        self.log = tk.Text(logf, height=16, wrap="word", state="disabled",
                           bg="#FFFFFF", fg="#222222", relief="solid",
                           borderwidth=1, font=("Microsoft YaHei UI", 9))
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.log.tag_configure("ok", foreground="#0A7D33")
        self.log.tag_configure("warn", foreground="#B45309")
        self.log.tag_configure("err", foreground="#C02626")
        self.log.tag_configure("dim", foreground="#666666")

        self.status = ttk.Label(main, text="就绪", style="Hint.TLabel")
        self.status.grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _welcome(self):
        self._log_line("欢迎使用图案检索工具")
        self._log_line("用法：① 选『查询图片目录』（要找的图） ② 选『语料图片目录』（要搜索的 B 显图库） "
                       "③ 点『开始检索』")
        self._log_line("首次检索会先建立语料索引（较慢），之后自动复用缓存；结果表与复核图保存到输出目录。")

    # ---------------- 小工具 ----------------
    def _log_line(self, msg, tag=None):
        if tag is None:
            if msg.startswith("[err]") or msg.startswith("[stop]"):
                tag = "err"
            elif ": 命中 " in msg:
                tag = "ok"
            elif ": 疑似 " in msg:
                tag = "warn"
            elif "未命中" in msg or "[index]" in msg:
                tag = "dim"
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    def _pick_dir(self, var, title):
        init = var.get().strip() or os.path.expanduser("~")
        if os.path.isfile(init):
            init = os.path.dirname(init)
        p = filedialog.askdirectory(title=f"选择{title}", initialdir=init)
        if p:
            var.set(os.path.normpath(p))

    def _toggle_adv(self):
        if self.adv_show.get():
            self.adv.grid()
        else:
            self.adv.grid_remove()

    def _default_out(self):
        desk = os.path.join(os.path.expanduser("~"), "Desktop")
        base = desk if os.path.isdir(desk) else os.path.expanduser("~")
        return os.path.join(base, "图案检索结果")

    def _reset_buttons(self):
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")

    def _load_cfg(self):
        try:
            with open(CFG_PATH, encoding="utf-8") as f:
                c = json.load(f)
            if c.get("queries"):
                self.var_q.set(c["queries"])
            if c.get("corpus"):
                self.var_c.set(c["corpus"])
            self.var_o.set(c.get("out") or self._default_out())
        except Exception:
            self.var_o.set(self._default_out())

    def _save_cfg(self):
        try:
            with open(CFG_PATH, "w", encoding="utf-8") as f:
                json.dump({"queries": self.var_q.get().strip(),
                           "corpus": self.var_c.get().strip(),
                           "out": self.var_o.get().strip()},
                          f, ensure_ascii=False)
        except Exception:
            pass

    # ---------------- 运行 ----------------
    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        qd = self.var_q.get().strip()
        cd = self.var_c.get().strip()
        od = self.var_o.get().strip() or self._default_out()
        for name, p in (("查询图片目录", qd), ("语料图片目录", cd)):
            if not p or not (os.path.isdir(p) or os.path.isfile(p)):
                messagebox.showwarning("路径无效", f"请先选择有效的{name}。")
                return
        try:
            os.makedirs(od, exist_ok=True)
        except Exception as e:
            messagebox.showerror("输出目录不可用", str(e))
            return

        try:
            prm = BatchParams(min_cov=float(self.var_mincov.get() or 0.75),
                              min_ncc=float(self.var_minncc.get() or 0.40),
                              strong_ncc=float(self.var_strong.get() or 0.70),
                              top=int(float(self.var_top.get() or 2)))
            lim = int(float(self.var_limit.get() or 0))
            limit = lim if lim > 0 else None
        except ValueError:
            messagebox.showwarning("参数错误", "高级设置有非数字项，本次将使用默认值。")
            prm, limit = BatchParams(), None

        self.var_o.set(od)
        self._save_cfg()
        run_dir = os.path.join(od, "检索结果_" + time.strftime("%Y%m%d_%H%M%S"))
        h = hashlib.md5(os.path.abspath(cd).lower().encode("utf-8")).hexdigest()[:8]
        cache = os.path.join(od, f"_语料索引_{h}.json")

        self.stop_event.clear()
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_open.config(state="disabled")
        self.bar.config(value=0, maximum=100)
        self._log_line("")
        self._log_line(f"===== 开始检索 {time.strftime('%Y-%m-%d %H:%M:%S')} =====", "dim")
        self._log_line(f"查询: {qd}")
        self._log_line(f"语料: {cd}")
        self._log_line(f"输出: {run_dir}")
        self._log_line(f"参数: 覆盖率≥{prm.min_cov}  NCC≥{prm.min_ncc}  强命中≥{prm.strong_ncc}"
                       f"  复核图{prm.top}张" + (f"  限{limit}张" if limit else ""), "dim")
        self.status.config(text="启动中…")

        self.worker = threading.Thread(
            target=self._worker, args=(qd, cd, run_dir, cache, prm, limit),
            daemon=True)
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.btn_stop.config(state="disabled")
            self.status.config(text="正在中止…（处理完当前图片后停止）")

    def _worker(self, qd, cd, run_dir, cache, prm, limit):
        try:
            stats = run_batch(qd, cd, run_dir, prm=prm, tag="", limit=limit,
                              index_json=cache, progress=self._on_prog,
                              should_stop=self.stop_event.is_set)
            self.q.put(("finish", stats))
        except Exception as e:
            import traceback
            self.q.put(("error", f"{type(e).__name__}: {e}", traceback.format_exc()))

    def _on_prog(self, stage, done, total, msg):
        self.q.put(("p", stage, done, total, msg))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                tag = item[0]
                if tag == "p":
                    _, stage, done, total, msg = item
                    if stage == "index" and total:
                        self.bar.config(maximum=total, value=done)
                        self.status.config(text=f"建立语料索引 {done}/{total}"
                                                f"（首次较慢，之后有缓存）")
                    elif stage in ("search", "done") and total:
                        self.bar.config(maximum=total, value=done)
                        if stage == "search":
                            self.status.config(text=f"检索 {done}/{total}")
                    if msg:
                        self._log_line(msg)
                elif tag == "finish":
                    self._on_finish(item[1])
                elif tag == "error":
                    self._log_line(f"[err] {item[1]}", "err")
                    self._log_line(item[2], "dim")
                    self.status.config(text="出错")
                    self._reset_buttons()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _on_finish(self, stats):
        self._reset_buttons()
        if stats.get("error"):
            self.status.config(text="未执行")
            self._log_line(f"[err] {stats['error']}", "err")
            messagebox.showerror("无法运行",
                                 f"检索未执行：{stats['error']}\n\n请检查两个目录是否选对。")
            return
        self.bar.config(value=self.bar["maximum"])
        if stats.get("stopped"):
            self.status.config(text="已中止（已完成部分已保存）")
        else:
            self.status.config(text="完成")
        xlsx = stats.get("xlsx") or ""
        if xlsx:
            self.last_out = os.path.dirname(xlsx)
            self.btn_open.config(state="normal")
        summary = (f"检索完成{'（中途停止）' if stats.get('stopped') else ''}：\n"
                   f"  强命中 {stats.get('strong', 0)} 张\n"
                   f"  疑似 {stats.get('suspect', 0)} 张（建议人工复核，见复核图）\n"
                   f"  未命中 {stats.get('miss', 0)} 张\n"
                   f"  无标记 {stats.get('empty', 0)} 张")
        if xlsx:
            summary += f"\n\n结果表：\n{xlsx}"
        if messagebox.askyesno("完成", summary + "\n\n现在打开结果文件夹吗？"):
            self._open_out()

    def _open_out(self):
        d = self.last_out
        if d and os.path.isdir(d):
            try:
                os.startfile(d)
            except Exception as e:
                messagebox.showerror("打开失败", str(e))
        else:
            messagebox.showinfo("提示", "还没有可打开的结果目录。")

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("正在检索", "检索尚未完成，确定退出吗？"):
                return
            self.stop_event.set()
        self._save_cfg()
        self.root.destroy()


def main():
    _enable_dpi()
    root = tk.Tk()
    PatternFinderApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        # 关窗后立即硬退出，保证进程秒退、零残留（实测：部分宿主环境/包装器
        # 会把 Tk 进程的退出清理拖住甚至退而不出）。配置等状态在关窗时已落盘，
        # 不依赖任何解释器清理或 atexit 钩子。
        os._exit(_rc)
