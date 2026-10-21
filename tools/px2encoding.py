#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
px2encoding.py — 影像坐标 ↔ 超声数据坐标 换算工具（M2 数据准备链核心）

换算链（项目数据流 L1 的收官一段）：
    x_px(某 Step 图内) ──[步进不变式]──> K                影像绝对坐标（跨步不变量）
    K ──[-D]──> E                                         数据 E 轴（每录制一常数 D）
    E ──[-BeamOffset[ch]]──> encoding                      通道对齐编码值
    encoding ──[-encoding_min]──> CSV 行号                 *_hdc.csv / *_label.csv

步进不变式（两种模式，由录制行进方向决定）：
    mode M：K = x - N   （内容随 N 增大右移，如 200922_17387 上行）
    mode P：K = x + N   （内容随 N 增大左移，如 0646 系列）
mode 由「锚点查询图在相邻 Step 的孪生命中」判定（Δx 与 ΔN 同号=M，异号=P）——
单步窗口内两种模式完全等价，只有跨步才能区分，这也是标定必须用锚点的原因。

标定流程（calibrate）：
  1. 读检索结果，找该录制的锚点命中（已定位的"片段"查询图 → Step-N @ x）
  2. 孪生检验确定 mode
  3. 取标注里各 ch6 主轨迹中心 E，与"锚点 K 邻域内"的图侧绿标记做
     差值直方图 → D（邻域限制是为了排除其它接头的同构重复）
  4. 残差报告 + 写 标定表.json

用法：
    python tools/px2encoding.py calibrate --recording 200922_17387
    python tools/px2encoding.py convert --recording 200922_17387 --step 465000 --x 1374
    python tools/px2encoding.py label --recording 200922_17387 \
        --out D:/HDC/output/dataset_prep/pilot_200922_17387
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PREP = r"D:/HDC/output/dataset_prep"
CALIB = os.path.join(PREP, "标定表.json")
MERGED_CSV = r"D:/HDC/output/scale_merged/检索结果.csv"
PXM = 285.67                      # 影像比例：px / m

DATA_SOURCES = {
    "200922_17387": dict(
        fragment=r"D:/HDC/data/1/2313/200922_17387片段.xxc.json",
        anno=r"D:/HDC/data/1/2313/200922_17387片段.xxc标注.json",
        image_dir=r"D:/HDC/data/第三批/200922_17387",
        cache=r"D:/HDC/output/scale_batch3/_cache/第三批__200922_17387.json"),
    "210201_18277": dict(
        fragment=r"D:/HDC/data/1/2313/210201_18277片段.xxc.json",
        anno=r"D:/HDC/data/1/2313/210201_18277片段.xxc标注.json",
        image_dir=r"D:/HDC/data/第三批/210201_18277",
        cache=None),
    "210201_18360": dict(
        fragment=r"D:/HDC/data/1/2313/210201_18360片段.xxc.json",
        anno=r"D:/HDC/data/1/2313/210201_18360片段.xxc标注.json",
        image_dir=r"D:/HDC/data/第三批/210201_18360",
        cache=None),
}


def load_calib():
    if os.path.exists(CALIB):
        with open(CALIB, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_calib(d):
    os.makedirs(PREP, exist_ok=True)
    with open(CALIB, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=2)


def K_of(x, n, mode):
    return x - n if mode == "M" else x + n


def x_at(k, n, mode):
    return k + n if mode == "M" else k - n


def load_anchors(recording):
    """从合并检索结果里找该录制的锚点命中 (查询图, Step-N, shift)。"""
    import csv
    out = []
    with open(MERGED_CSV, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["是否命中"] == "否":
                continue
            if not r["查询图"].startswith(recording):
                continue
            m = re.match(r"(.+)_Step-(\d+)\.png$", (r.get("命中图片") or "").strip())
            if not m:
                continue
            out.append({"query": r["查询图"], "step": int(m.group(2)),
                        "shift": float(r["位移(px)"]), "ncc": r.get("NCC")})
    return out


def detect_mode(anchor, src):
    """孪生检验：把锚点查询图放到相邻 Step 里找，看 Δx 与 ΔN 同号还是异号。"""
    import find_pattern as fp
    qp = os.path.join(r"D:/HDC/data/png", anchor["query"])
    if not os.path.exists(qp):
        return None, "查询图文件缺失"
    qd = fp.dmap(fp.load_rgb(qp))
    N0, x0 = anchor["step"], anchor["shift"]
    for dN in (-1000, 1000, -2000, 2000):
        fs = glob.glob(os.path.join(src["image_dir"],
                                    "*_Step-%d.png" % (N0 + dN)))
        if not fs:
            continue
        res = fp.ncc_map(qd, fp.dmap(fp.load_rgb(fs[0])))
        if res is None:
            continue
        import numpy as np
        idx = np.unravel_index(np.argmax(res), res.shape)
        v = float(res[idx])
        if v < 0.7:
            continue
        dx = int(idx[1]) - x0
        if abs(dx - dN) <= 40:
            return "M", "Step-%d 孪生 NCC=%.3f Δx=%+d（同号⇒K=x-N）" % (N0 + dN, v, dx)
        if abs(dx + dN) <= 40:
            return "P", "Step-%d 孪生 NCC=%.3f Δx=%+d（异号⇒K=x+N）" % (N0 + dN, v, dx)
    return None, "相邻步未找到孪生命中"


def load_marks_x(src):
    out = []
    if src.get("cache") and os.path.exists(src["cache"]):
        idx = json.load(open(src["cache"], encoding="utf-8"))
        for f in idx["files"]:
            n = int(re.search(r"Step-(\d+)", f["path"]).group(1))
            for m in f["marks"]:
                out.append((n + m["cx"], n, m["cx"], m["cls"]))
        return out
    import find_pattern as fp
    for f in sorted(glob.glob(os.path.join(src["image_dir"], "*.png"))):
        n = int(re.search(r"Step-(\d+)", f).group(1))
        for m in fp.extract_marks(fp.load_rgb(f)):
            out.append((n + m["cx"], n, m["cx"], m["cls"]))
    return out


def annotation_ch6_centers(anno_path):
    an = json.load(open(anno_path, encoding="utf-8"))
    centers = []
    for ann in an.get("annotations", []):
        pts = sorted(ann["points"], key=lambda p: p["e"])
        runs, cur = [], [pts[0]]
        for p in pts[1:]:
            if p["e"] - cur[-1]["e"] <= 2:
                cur.append(p)
            else:
                runs.append(cur)
                cur = [p]
        runs.append(cur)
        for r in runs:
            if r[0]["channel"] == 6:
                centers.append(sum(p["e"] for p in r) / len(r))
    return centers


def beam_offsets(fragment_json):
    with open(fragment_json, encoding="utf-8") as fh:
        d = json.load(fh)
    bo = d[0].get("BeamOffset", [])
    return {int(bo[i]): bo[i + 1] for i in range(0, len(bo) - 1, 2)}


# ---------------- 子命令 ----------------
def cmd_calibrate(args):
    src = DATA_SOURCES[args.recording]
    anchors = load_anchors(args.recording)
    if not anchors:
        print("该录制没有已定位的锚点查询（无法自动标定）。")
        return 1
    anchor = anchors[0]
    print("锚点: %s → Step-%d @%.0f (NCC=%s)"
          % (anchor["query"], anchor["step"], anchor["shift"], anchor["ncc"]))

    mode, why = detect_mode(anchor, src)
    if mode is None:
        print("⚠ 模式判定失败（%s），默认按 M（K=x-N）继续，请人工核对。" % why)
        mode = "M"
    else:
        print("模式判定: %s —— %s" % (mode, why))

    centers = annotation_ch6_centers(src["anno"])
    print("标注 ch6 轨迹中心 E:", [round(c, 1) for c in centers])

    # 锚点 K 邻域内的图侧标记（排除远端同构接头）
    # 只用绿标记：实测绿标记=ch6 轨迹的检测落点（6/6 残差 ≤2px）；
    # 蓝标记相对绿标记有 ~18px 系统偏移，混用会让密集区的就近匹配串位
    K_anchor = K_of(anchor["shift"], anchor["step"], mode)
    marks = load_marks_x(src)
    near = []
    for nx, n, cx, cls in marks:
        if cls != "green":
            continue
        k = K_of(cx, n, mode)
        if abs(k - K_anchor) <= 2500:
            near.append(k)
    near = sorted(near)
    # 去重（孪生 ±8）
    uniq = []
    for k in near:
        if uniq and k - uniq[-1] <= 8:
            continue
        uniq.append(k)
    print("邻域标记（去重后）: %d 个，K ∈ [%.0f, %.0f]"
          % (len(uniq), uniq[0] if uniq else 0, uniq[-1] if uniq else 0))

    # D 直方图：D = K_mark - E_center
    hist = {}
    for E in centers:
        for k in uniq:
            D = int(round(k - E))
            hist[D] = hist.get(D, 0) + 1
    top = sorted(hist.items(), key=lambda kv: -kv[1])[:12]
    best = None
    cands = []
    # 收集**全部**候选（同构接头多的录制里真解可能排不进 top12！实测
    # 210201_18277 有 14+ 个 6/6 解；注意真拟合残差跨多个整数 bin，
    # 单 bin 计数不高，阈值取低再靠 ±6 细化 + 锚点交叉验证裁决）
    D0s = [d0 for d0, cnt in hist.items() if cnt >= 2]
    D0s.sort(key=lambda d0: -hist[d0])
    for D0 in D0s[:600]:
        for D in range(D0 - 6, D0 + 7):
            res = []
            for E in centers:
                d = min((abs(k - (E + D)) for k in uniq), default=1e9)
                res.append(d)
            n_ok = sum(1 for d in res if d <= 6)
            med = sorted(res)[len(res) // 2]
            cands.append((n_ok, med, D))

    # 多候选裁决（该录制同构接头众多，6/6 可能有十几个解）：
    #  ① 若锚点存在：仅保留"片段窗口覆盖锚点 K"的候选
    #  ② 用锚点裁片自身的绿标记做交叉验证（残差最小者胜）——裁片是从片段里截的，
    #     其内部的 6 孔图案必须与片段 ch6 轨迹一一对上
    cands.sort(key=lambda c: (-c[0], c[1]))
    sel = None
    cross_note = ""
    if uniq:
        pos_cands = [c for c in cands if c[0] >= 5
                     and (c[2] + 2) <= K_anchor <= (c[2] + 1731)]
        if not pos_cands:
            pos_cands = [c for c in cands if c[0] >= 5][:6]
            cross_note = "（无候选覆盖锚点，退化为全录拟合）"
        # 交叉验证：锚点裁片绿标记
        qp = os.path.join(r"D:/HDC/data/png", anchor["query"])
        crop_greens = []
        if os.path.exists(qp):
            import find_pattern as fp
            for m in fp.extract_marks(fp.load_rgb(qp)):
                if m["cls"] == "green":
                    crop_greens.append(m["cx"])
        scored = []
        for n_ok, med, D in pos_cands:
            if crop_greens:
                # 片段 ch6 中心在裁片内的预测 x（裁片左缘 K = K_anchor）
                preds = [(E + D) - K_anchor for E in centers]
                vis = [p for p in preds if 0 <= p <= 512]
                if vis:
                    res2 = [min(abs(g - p) for g in crop_greens) for p in vis]
                    ok2 = sum(1 for d in res2 if d <= 8)
                    med2 = sorted(res2)[len(res2) // 2]
                    scored.append((ok2, -med2, n_ok, -med, D, len(vis)))
                    continue
            scored.append((0, 0, n_ok, -med, D, 0))
        scored.sort(reverse=True)
        ok2, neg2, n_ok, neg_med, D, n_vis = scored[0]
        if crop_greens and n_vis:
            cross_note = ("锚点裁片交叉验证：%d/%d 个可见孔位落上绿标记（中位 %.1fpx）"
                          % (ok2, n_vis, -neg2))
        best = (n_ok, neg_med, D)
    else:
        best = (cands[0][0], -cands[0][1], cands[0][2])
    n_ok, neg_med, D = best
    med = -neg_med
    if cross_note:
        print("\n" + cross_note)
    print("\n== 标定结果 ==")
    print("mode=%s  D=%d  （%d/%d 条 ch6 轨迹匹配，中位残差 %.1f px）"
          % (mode, D, n_ok, len(centers), med))
    for E in centers:
        d = min((abs(k - (E + D)) for k in uniq), default=1e9)
        print("   E=%7.1f → K_pred=%.1f  最近标记距离=%.1f" % (E, E + D, d))
    # 锚点一致性
    in_win = (D + 2) <= K_anchor <= (D + 1731)
    print("片段 K 窗口 [%d, %d] 覆盖锚点 K=%.0f : %s"
          % (D + 2, D + 1731, K_anchor, "是" if in_win else "否（异常！）"))

    calib = load_calib()
    calib[args.recording] = {
        "mode": mode, "D": D, "residual_px": med, "n_match": n_ok,
        "anchor": {"query": anchor["query"], "step": anchor["step"],
                   "shift": anchor["shift"]},
        "method": "孪生模式判定 + 锚点邻域 ch6 轨迹×绿标记直方图 + 裁片交叉验证",
        "fragment_json": src["fragment"], "image_dir": src["image_dir"],
        "note": "K=x-N(M)/x+N(P)；E=K-D；x=K±N；X_abs_m=K/285.67（仅作定位参考）",
    }
    save_calib(calib)
    print("[saved]", CALIB)
    return 0


def cmd_convert(args):
    calib = load_calib()
    if args.recording not in calib:
        print("该录制尚无标定，请先运行 calibrate。")
        return 1
    c = calib[args.recording]
    mode, D = c["mode"], c["D"]
    offs = beam_offsets(DATA_SOURCES[args.recording]["fragment"])

    x0, x1 = args.x, args.x + args.width
    K0 = K_of(x0, args.step, mode)
    K1 = K_of(x1, args.step, mode)
    E0, E1 = K0 - D, K1 - D
    print("影像: Step-%d x=[%d, %d)  →  K=[%d, %d)（mode %s）" % (
        args.step, x0, x1, K0, K1, mode))
    print("数据: E = [%d, %d)（D=%d）" % (E0, E1, D))
    print("\n通道编码区间（encoding = E - BeamOffset[ch]）:")
    for ch in range(9):
        o = offs.get(ch, 0)
        print("   ch%d: BeamOffset=%3d  encoding=[%d, %d)" % (ch, o, E0 - o, E1 - o))
    print("\n（片段 E 范围 [2,1731] 落到影像 x：Step-N 下 x = E+D±N）")
    if args.json_out:
        out = {"recording": args.recording, "mode": mode, "D": D,
               "step": args.step, "x": [x0, x1], "K": [K0, K1], "E": [E0, E1],
               "encoding": {("ch%d" % ch): [E0 - offs.get(ch, 0),
                                            E1 - offs.get(ch, 0)]
                            for ch in range(9)}}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print("[out]", args.json_out)
    return 0


def cmd_label(args):
    src = DATA_SOURCES[args.recording]
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    j = os.path.join(outdir, os.path.basename(src["fragment"]))
    a = os.path.join(outdir, os.path.basename(src["anno"]))
    shutil.copy2(src["fragment"], j)
    shutil.copy2(src["anno"], a)

    sys.path.insert(0, r"D:/HDC")        # 转换器在项目根目录
    from ultrasonic_data_converter import UltrasonicDataConverter
    conv = UltrasonicDataConverter()
    conv.generate_all_csvs(j, a)

    base = os.path.splitext(os.path.basename(j))[0]
    lbl_path = os.path.join(outdir, base + "_label.csv")
    offs = beam_offsets(j)
    import csv as _csv
    rows = list(_csv.DictReader(open(lbl_path, encoding="utf-8")))
    enc_min = int(rows[0]["encoding"])
    an = json.load(open(a, encoding="utf-8"))
    n_pts = n_ok = n_bad = 0
    used_labels = {}
    for ann in an["annotations"]:
        for p in ann["points"]:
            n_pts += 1
            enc = int(round(p["e"] - offs.get(p["channel"], 0)))
            idx = enc - enc_min
            if 0 <= idx < len(rows) and int(rows[idx]["label_%d" % p["channel"]]) > 0:
                n_ok += 1
                v = int(rows[idx]["label_%d" % p["channel"]])
                used_labels[v] = used_labels.get(v, 0) + 1
            else:
                n_bad += 1
    print("\n== 标签生成与验证 ==")
    print("输出目录:", outdir)
    print("CSV 行数: %d（encoding %d ~ %d）" % (len(rows), enc_min,
                                            int(rows[-1]["encoding"])))
    print("标注点 %d 个：落位成功 %d / 失败 %d" % (n_pts, n_ok, n_bad))
    print("标签 id 分布:", used_labels)

    calib = load_calib()
    if args.recording in calib:
        c = calib[args.recording]
        mode, D = c["mode"], c["D"]
        print("\n标注 → 影像定位（各条 ch6 轨迹起点，按锚点步演示）:")
        anc = c["anchor"]
        for ann in an["annotations"]:
            es = sorted(p["e"] for p in ann["points"] if p["channel"] == 6)
            if es:
                k = es[0] + D
                xs = x_at(k, anc["step"], mode)
                print("   %-8s ch6 E=%d..%d → K=%d..%d（Step-%d 下 x=%d..%d）"
                      % (ann["subtype"]["name"], es[0], es[-1],
                         es[0] + D, es[-1] + D, anc["step"],
                         xs, x_at(es[-1] + D, anc["step"], mode)))
    return 0


def main():
    ap = argparse.ArgumentParser(description="影像↔数据坐标换算（M2）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("calibrate")
    p1.add_argument("--recording", required=True)
    p1.set_defaults(func=cmd_calibrate)

    p2 = sub.add_parser("convert")
    p2.add_argument("--recording", required=True)
    p2.add_argument("--step", type=int, required=True)
    p2.add_argument("--x", type=int, required=True)
    p2.add_argument("--width", type=int, default=512)
    p2.add_argument("--json-out", default="")
    p2.set_defaults(func=cmd_convert)

    p3 = sub.add_parser("label")
    p3.add_argument("--recording", required=True)
    p3.add_argument("--out", required=True)
    p3.set_defaults(func=cmd_label)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
