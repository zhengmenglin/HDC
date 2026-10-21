#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scale_search.py — 大规模图库分层检索（阶段 A · 真机首版）

方案见 docs/60万图库快速检索方案_20261013.md：
  · 元数据分桶：T1 同录制键（文件夹名作为 token 出现在查询名）→ T2 同日期+机号
    → T3 同日期 → T4 无信息（本版只记录待办，不做全库扫描）
  · 组内检索复用 find_pattern 的成熟引擎（标记投票 + 分批 NCC 复核，宁缺勿假）
  · 索引按"录制文件夹"建一次、磁盘缓存、多进程并行
  · 抽样模式（--sample）：正式全量前的人工抽检小样

用法：
  # 抽样测试
  python tools/scale_search.py --queries D:/HDC/data/png --corpus D:/HDC/data/shengyu \
      --out D:/HDC/output/scale_test --sample 6 --sample-t4 1 --workers 8
  # 全量（后续）
  python tools/scale_search.py ... --out D:/HDC/output/scale_full
  # 增量复检（新图库到货后只跑未命中）
  python tools/scale_search.py --queries D:/HDC/data/png --corpus D:/HDC/data/第三批 \
      --out D:/HDC/output/scale_batch3 --only-missed D:/HDC/output/scale_full/检索结果.csv

输出：
  检索结果.xlsx / .csv      每张查询一行（含层级/命中文件夹/复核图路径）
  标注迁移表.csv             命中段的 (x0,x1) 区间 —— 喂后续"px→encoding→训练样本"管道
  复核图/<查询图>/rank1_*.png
  run_log.txt
"""
import argparse
import concurrent.futures as cf
import csv
import glob
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_pattern as fp                                    # noqa: E402
from find_pattern_batch import BatchParams, search_one       # noqa: E402

EXPORT = re.compile(r"20\d{6}[_\-]\d{6}")
DR = re.compile(r"(?<!\d)((?:1[0-9]|2[0-6])(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01]))(?!\d)")
NR = re.compile(r"(?<!\d)(\d{2,6})(?!\d)")

COL_HEAD = ["查询图", "是否命中", "层级", "命中文件夹", "命中图片", "位移(px)", "覆盖率",
            "NCC", "区间(px)", "米区间(相对该图0点)", "复核图", "备注"]
COL_MIG = ["查询图", "层级", "命中文件夹", "命中图片", "位移(px)", "x0", "x1",
           "米区间", "覆盖率", "NCC"]


# ---------------- 图库收集与特征 ----------------
def collect_recordings(root):
    """收集录制组。支持两种布局（自动识别）：
    ① 批次/录制/files（shengyu：root/1/241204_.../Step-*.png）
    ② 平铺：root/录制/files（第三批等新图库：root/数据14/数据14_Step-*.png）
    """
    recs = []
    for b in sorted(os.listdir(root)):
        bd = os.path.join(root, b)
        if not os.path.isdir(bd):
            continue
        if glob.glob(os.path.join(bd, "*.png")):
            # 平铺布局：子目录本身就是录制文件夹
            recs.append({"batch": os.path.basename(os.path.abspath(root)), "name": b,
                         "path": bd, "files": sorted(glob.glob(os.path.join(bd, "*.png")))})
            continue
        for n in sorted(os.listdir(bd)):
            p = os.path.join(bd, n)
            if not os.path.isdir(p):
                continue
            files = sorted(glob.glob(os.path.join(p, "*.png")))
            if files:
                recs.append({"batch": b, "name": n, "path": p, "files": files})
    return recs


def build_features(recs):
    for r in recs:
        r["fd"] = set(DR.findall(r["name"]))
        r["fn"] = set(NR.findall(r["name"]))
        r["pat"] = re.compile(r"(?<!\d)" + re.escape(r["name"]) + r"(?!\d)")
    return recs


def query_plan(qpath, recs):
    stem = os.path.splitext(os.path.basename(qpath))[0]
    s = EXPORT.sub(" ", stem)
    qd = set(DR.findall(s))
    qn = set(NR.findall(s))
    t1 = [r for r in recs if r["pat"].search(stem)]
    t2, t3 = [], []
    if not t1:
        for r in recs:
            if r["fd"] & qd:
                if (r["fn"] & qn) - (r["fd"] | qd):
                    t2.append(r)
                else:
                    t3.append(r)
    return {"t1": t1, "t2": t2, "t3": t3}


# ---------------- 索引（多进程 + 磁盘缓存） ----------------
def _index_chunk(task):
    """子进程任务：对一批文件做 标记提取 + 标定。"""
    key, files = task
    metas = []
    for f in files:
        try:
            a = fp.load_rgb(f)
            pxm, x0 = fp.calibrate(a)
            metas.append({"path": os.path.abspath(f), "marks": fp.extract_marks(a),
                          "pxm": pxm, "x0": x0})
        except Exception as e:
            # 单张读图失败不拖垮整组（60 万规模必有个别坏图）；
            # 缺失文件在 flush 时按 path 过滤自然略过（宁缺勿假）
            print("[index] 跳过无法读取的文件 %s: %s: %s"
                  % (f, type(e).__name__, e), file=sys.stderr, flush=True)
    return key, metas


def cache_path(cache_dir, rec):
    key = re.sub(r'[\\/:*?"<>|]', "_", rec["batch"] + "__" + rec["name"])
    return os.path.join(cache_dir, key + ".json")


def index_cached(rec, cache_dir):
    cp = cache_path(cache_dir, rec)
    if not os.path.exists(cp):
        return None
    try:
        with open(cp, encoding="utf-8") as fh:
            idx = json.load(fh)
        if ([os.path.abspath(x["path"]) for x in idx.get("files", [])]
                == [os.path.abspath(x) for x in rec["files"]]
                and idx.get("palette") == json.dumps(fp.PALETTE)):
            return idx["files"]
    except Exception:
        pass
    return None


def prebuild(recs, cache_dir, workers, log):
    todo = [r for r in recs if index_cached(r, cache_dir) is None]
    if not todo:
        log("[index] 全部命中缓存（%d 组）" % len(recs))
        return
    n_files = sum(len(r["files"]) for r in todo)
    log("[index] 需建 %d 组 / %d 张（%d 进程并行）" % (len(todo), n_files, workers))

    # 逐组落盘：某组全部块完成即写缓存（ex.map 按任务序产出结果，组内块在
    # tasks 中连续，故"该组最后一块产出"== 整组齐活）。任务被中止时已建组
    # 全部保留，重启经 index_cached 直接复用 —— 实测一跑 4797 张要 ~8 分钟，
    # 丢一次太贵（2026-10-13 第一次抽样就被中断，白跑了 5 分钟）
    rec_by_cp = {cache_path(cache_dir, r): r for r in todo}
    chunk_total = {cp: (len(r["files"]) + 39) // 40 for cp, r in rec_by_cp.items()}
    tasks = []
    for r in todo:
        cp = cache_path(cache_dir, r)
        for i in range(0, len(r["files"]), 40):
            tasks.append((cp, r["files"][i:i + 40]))

    parts, done_chunks = {}, {}

    def flush(cp):
        r = rec_by_cp[cp]
        order = {os.path.abspath(m["path"]): m for m in parts.pop(cp, [])}
        metas = [order[os.path.abspath(f)] for f in r["files"]
                 if os.path.abspath(f) in order]
        tmp = cp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"palette": json.dumps(fp.PALETTE), "files": metas},
                      fh, ensure_ascii=False)
        os.replace(tmp, cp)      # 原子替换：即使此刻被杀，也不会留下半截缓存

    t0 = time.time()
    done = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for key, metas in ex.map(_index_chunk, tasks):
            parts.setdefault(key, []).extend(metas)
            done_chunks[key] = done_chunks.get(key, 0) + 1
            if done_chunks[key] >= chunk_total[key]:
                flush(key)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                el = time.time() - t0
                log("  [index] %d/%d 块  %.0fs  预计剩余 %.0fs"
                    % (done, len(tasks), el, el / done * (len(tasks) - done)))
    for cp in list(chunk_total):      # 兜底（正常流程不会触发）
        if cp in parts:
            flush(cp)
    log("[index] 完成 %d 组，用时 %.1fs" % (len(todo), time.time() - t0))


# ---------------- XLSX 输出 ----------------
def write_xlsx2(path, rows):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "检索结果"
    head_font = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="E8F0E0")
    fills = {"是": PatternFill("solid", fgColor="E1F5EE"),
             "疑似": PatternFill("solid", fgColor="FAEEDA"),
             "否": PatternFill("solid", fgColor="F1EFE8")}
    ws.append(COL_HEAD)
    for c in range(1, len(COL_HEAD) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = head_font
        cell.fill = head_fill
    for r in rows:
        ws.append([r.get(c, "") for c in COL_HEAD])
    for i in range(2, ws.max_row + 1):
        fill = fills.get(ws.cell(row=i, column=2).value)
        if fill:
            for c in range(1, len(COL_HEAD) + 1):
                ws.cell(row=i, column=c).fill = fill
    for i, w in enumerate([34, 8, 6, 18, 30, 9, 7, 7, 14, 17, 42, 30], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    wb.save(path)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="大规模图库分层检索（阶段 A）")
    ap.add_argument("--queries", default=r"D:/HDC/data/png")
    ap.add_argument("--corpus", default=r"D:/HDC/data/shengyu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="抽样 N 张（优先覆盖不同录制组）")
    ap.add_argument("--sample-t4", type=int, default=0, help="额外抽样 N 张无元数据查询")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-missed", default="",
                    help="增量复检：只跑该 检索结果.csv 中『是否命中=否』的查询（新图库到货后使用）")
    ap.add_argument("--max-t1", type=int, default=5)
    ap.add_argument("--max-t2", type=int, default=4)
    ap.add_argument("--max-t3", type=int, default=3)
    ap.add_argument("--max-tier-files", type=int, default=6000, help="T2/T3 单组规模上限")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache_dir = os.path.join(args.out, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    logf = open(os.path.join(args.out, "run_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    prm = BatchParams()
    recs = build_features(collect_recordings(args.corpus))
    log("图库录制组: %d | 查询目录: %s" % (len(recs), args.queries))

    queries = sorted(glob.glob(os.path.join(args.queries, "*.png")))
    if args.only_missed:
        keep = set()
        with open(args.only_missed, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if (r.get("是否命中") or "").strip() == "否":
                    keep.add(r["查询图"])
        before = len(queries)
        queries = [q for q in queries if os.path.basename(q) in keep]
        log("增量复检：查询 %d 张 → 未命中 %d 张（来源 %s）"
            % (before, len(queries), args.only_missed))
    plans = {q: query_plan(q, recs) for q in queries}
    n_t1 = sum(1 for q in queries if plans[q]["t1"])
    log("查询共 %d 张：T1 可直锁 %d，无直锁 %d" % (len(queries), n_t1, len(queries) - n_t1))

    if args.sample:
        rng = random.Random(args.seed)
        pool1 = [q for q in queries if plans[q]["t1"]]
        groups = {}
        for q in pool1:
            fr = plans[q]["t1"][0]
            groups.setdefault(fr["batch"] + "/" + fr["name"], []).append(q)
        keys = sorted(groups)
        rng.shuffle(keys)
        picked = []
        i = 0
        while len(picked) < args.sample and any(groups[k] for k in keys):
            k = keys[i % len(keys)]
            if groups[k]:
                picked.append(groups[k].pop())
            i += 1
        pool4 = [q for q in queries if not (plans[q]["t1"] or plans[q]["t2"] or plans[q]["t3"])]
        rng.shuffle(pool4)
        picked += pool4[: args.sample_t4]
        sel = picked
    else:
        sel = queries

    log("本次处理 %d 张：" % len(sel))
    for q in sel:
        p = plans[q]
        t1s = "、".join(r["batch"] + "/" + r["name"] for r in p["t1"][:3]) or "-"
        log("   %s   [T1: %s]" % (os.path.basename(q)[:56], t1s))

    need = {}
    for q in sel:
        p = plans[q]
        for tier, cap in (("t1", args.max_t1), ("t2", args.max_t2), ("t3", args.max_t3)):
            used = 0
            for r in p[tier]:
                if used >= cap:
                    break
                if tier != "t1" and len(r["files"]) > args.max_tier_files:
                    continue
                need[id(r)] = r
                used += 1
    log("需索引装载: %d 组" % len(need))
    prebuild(list(need.values()), cache_dir, args.workers, log)

    rows, mig_rows = [], []
    n_yes = n_sus = n_no = 0
    t_all = time.time()
    for qi, q in enumerate(sel, 1):
        stem = os.path.splitext(os.path.basename(q))[0]
        try:
            q_rgb = fp.load_rgb(q)
            Q = fp.extract_marks(q_rgb)
            if not Q:
                n_no += 1
                rows.append({"查询图": os.path.basename(q), "是否命中": "否",
                             "层级": "-", "备注": "图中未提取到彩色标记"})
                log("[%d/%d] %s: 无标记" % (qi, len(sel), stem[:44]))
                continue

            p = plans[q]
            best = None
            best2 = None          # 二次判定候选（像素强匹配、覆盖率未达标）
            rej = None            # 被拒最高候选（未命中备注用）
            stop = False
            for tier, cap in (("t1", args.max_t1), ("t2", args.max_t2), ("t3", args.max_t3)):
                if stop:
                    break
                used = 0
                for r in p[tier]:
                    if used >= cap:
                        break
                    if tier != "t1" and len(r["files"]) > args.max_tier_files:
                        continue
                    idx = index_cached(r, cache_dir)
                    if not idx:
                        continue
                    used += 1
                    dedup, accepted, _n = search_one(q, q_rgb, Q, idx, prm)
                    # 二次判定通道：焊缝/鱼鳞/锈蚀等密集标记场景，两次渲染的标记
                    # 合并/拆分不同 ⇒ 覆盖率被低估（实测 9 例落在 0.53~0.71），但
                    # 残差 NCC 仍达 0.80+。此时以像素证据为准兜回；阈值刻意高于
                    # 强命中线（只靠单证据），9 例已全部肉眼复核为真命中(2026-10-13)
                    for h in dedup:
                        if h["ncc"] is None:
                            continue
                        if (h["ncc"] >= max(prm.strong_ncc, 0.75)
                                and h["cov"] >= prm.cand_cov
                                and (best2 is None or h["ncc"] > best2["h"]["ncc"])):
                            best2 = {"tier": tier, "rec": r, "h": h}
                        if rej is None or h["ncc"] > rej["h"]["ncc"]:
                            rej = {"tier": tier, "rec": r, "h": h}
                    if accepted:
                        h = accepted[0]
                        if best is None or (h["ncc"] or -1) > (best["h"]["ncc"] or -1):
                            best = {"tier": tier, "rec": r, "h": h, "acc": accepted}
                        if h["ncc"] is not None and h["ncc"] >= prm.strong_ncc:
                            stop = True
                            break

            second = best is None and best2 is not None
            if second:
                best = {"tier": best2["tier"], "rec": best2["rec"], "h": best2["h"],
                        "acc": []}

            if best:
                h, r = best["h"], best["rec"]
                tier = best["tier"]
                strong = h["ncc"] is not None and h["ncc"] >= prm.strong_ncc
                n_yes += int(strong)
                n_sus += int(not strong)
                odir = os.path.join(args.out, "复核图", stem)
                os.makedirs(odir, exist_ok=True)
                c_rgb = fp.load_rgb(h["file"])
                png = os.path.join(odir, "rank1_%s_x%d.png" % (
                    os.path.splitext(os.path.basename(h["file"]))[0],
                    int(round(h["shift"]))))
                fp.draw_overlay(q_rgb, Q, c_rgb,
                                {"shift": h["shift"], "scale": 1.0},
                                png, png.replace(".png", "_full.png"))
                overlay_rel = os.path.relpath(png, args.out)
                note = "" if tier == "t1" else "经降级检索（%s）" % tier.upper()
                if second:
                    note = (note + "；" if note else "") + (
                        "二次判定：覆盖率 %.2f 未达标但像素强匹配（NCC=%.3f，"
                        "密集标记合并所致）" % (h["cov"], h["ncc"]))
                if not strong:
                    note = (note + "；" if note else "") + "命中强度较弱，建议人工复核"
                t2s = "、".join(dict.fromkeys(
                    os.path.basename(a["file"]) for a in best["acc"][1:4]
                    if a["file"] != h["file"] and a["ncc"] is not None
                    and a["ncc"] >= 0.6))
                row = {"查询图": os.path.basename(q),
                       "是否命中": "是" if strong else "疑似",
                       "层级": tier.upper(),
                       "命中文件夹": r["batch"] + "/" + r["name"],
                       "命中图片": os.path.basename(h["file"]),
                       "位移(px)": round(h["shift"], 1),
                       "覆盖率": round(h["cov"], 3),
                       "NCC": None if h["ncc"] is None else round(h["ncc"], 3),
                       "区间(px)": "[%s,%s)" % (h["px_range"][0], h["px_range"][1]),
                       "米区间(相对该图0点)": ("%.2f~%.2fm" % tuple(h["m_range"])
                                              if h["m_range"] else ""),
                       "复核图": overlay_rel,
                       "备注": note}
                rows.append(row)
                mig_rows.append({"查询图": os.path.basename(q), "层级": tier.upper(),
                                 "命中文件夹": r["batch"] + "/" + r["name"],
                                 "命中图片": os.path.basename(h["file"]),
                                 "位移(px)": round(h["shift"], 1),
                                 "x0": h["px_range"][0], "x1": h["px_range"][1],
                                 "米区间": row["米区间(相对该图0点)"],
                                 "覆盖率": round(h["cov"], 3),
                                 "NCC": None if h["ncc"] is None else round(h["ncc"], 3)})
                log("[%d/%d] %s: %s %s %s shift=%.0f cov=%.2f ncc=%.3f"
                    % (qi, len(sel), stem[:44], row["是否命中"], tier.upper(),
                       os.path.basename(h["file"])[:26], h["shift"], h["cov"],
                       h["ncc"] or -1))
            else:
                n_no += 1
                has_plan = bool(p["t1"] or p["t2"] or p["t3"])
                if not has_plan:
                    note = "无匹配录制（查询名未含库中录制键）→ 待全库扫描（阶段 C）"
                elif rej is not None:
                    note = ("组内未达标；最接近候选 %s cov=%.2f ncc=%.3f"
                            % (os.path.basename(rej["h"]["file"]), rej["h"]["cov"],
                               rej["h"]["ncc"]))
                else:
                    note = "组内未达标（无候选）"
                rows.append({"查询图": os.path.basename(q), "是否命中": "否",
                             "层级": "T4" if not has_plan else "-", "备注": note})
                log("[%d/%d] %s: 未命中（%s）" % (qi, len(sel), stem[:44], note))
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append({"查询图": os.path.basename(q), "是否命中": "否",
                         "层级": "-", "备注": "异常: %s: %s" % (type(e).__name__, e)})
            log("[%d/%d] %s: 异常 %s: %s" % (qi, len(sel), stem[:44], type(e).__name__, e))

    xlsx = os.path.join(args.out, "检索结果.xlsx")
    write_xlsx2(xlsx, rows)
    csvp = os.path.join(args.out, "检索结果.csv")
    fp.write_csv(csvp, rows, COL_HEAD)
    migp = os.path.join(args.out, "标注迁移表.csv")
    fp.write_csv(migp, mig_rows, COL_MIG)
    log("")
    n_2nd = sum(1 for r in rows if "二次判定" in (r.get("备注") or ""))
    log("== 完成: 命中(强) %d（含二次判定 %d）/ 疑似 %d / 未命中 %d  （共 %d，用时 %.1fs）"
        % (n_yes, n_2nd, n_sus, n_no, len(sel), time.time() - t_all))
    log("[out] %s" % xlsx)
    log("[out] %s" % csvp)
    log("[out] %s" % migp)
    logf.close()


if __name__ == "__main__":
    main()
