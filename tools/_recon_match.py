# -*- coding: utf-8 -*-
"""
scratch recon: 几何测量 + 多尺度 NCC 可行性实验（非最终工具）
目标：确认 512x512 查询图与 1888x512 语料图之间的几何关系与可匹配性
"""
import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
import cv2

QDIR = r'D:/HDC/data/2223'
QRY = r'D:/HDC/data/核伤251003_0646_22232_20251110_084801/核伤251003_0646_22232_20251110_084801.png'
OUT = r'D:/HDC/tools/_out'
os.makedirs(OUT, exist_ok=True)


def load_rgb(p):
    im = Image.open(p).convert('RGBA')
    bgw = Image.new('RGBA', im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bgw, im).convert('RGB')
    return np.asarray(im).astype(np.int32)


def ink(a, bg):
    d = (a - bg).astype(np.float32)
    return np.sqrt((d * d).sum(axis=2)).astype(np.float32)


def geom(name, a):
    H, W, _ = a.shape
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    bg = a[2, 2]
    dark = (R < 100) & (G < 100) & (B < 100)
    red = (R > 170) & (G < 100) & (B < 100)
    drow = dark.sum(1)
    rrow = red.sum(1)
    print(f'[{name}] {W}x{H} bg={tuple(int(v) for v in bg)}')
    dline = np.where(drow > W * 0.5)[0]
    print('  dark rows>W*0.5:', dline[:12])
    print('  red rows>W*0.5 :', np.where(rrow > W * 0.5)[0][:12])
    if len(dline):
        t0 = dline[0]
        band = dark[t0:t0 + 16, :].sum(0)
        cols = np.where(band >= 6)[0]
        cent = []
        for i in cols:
            if cent and i - cent[-1][-1] <= 2:
                cent[-1].append(i)
            else:
                cent.append([i])
        cents = [int(np.mean(c)) for c in cent]
        print('  tick cols:', cents[:40])
        if len(cents) >= 3:
            d = np.diff(cents)
            print('  tick spacing px median=%.1f  (0.5m)  -> px/m=%.1f' % (np.median(d), np.median(d) * 2))


for p in sorted(glob.glob(QDIR + '/*.png')) + [QRY]:
    geom(os.path.basename(p), load_rgb(p))

# ---------- multi-scale NCC ----------
q = load_rgb(QRY)
qbg = q[2, 2]
qk = ink(q, qbg)
print('\n== multi-scale NCC (template = query full) ==')
Hq, Wq = qk.shape
for p in sorted(glob.glob(QDIR + '/*.png')):
    c = load_rgb(p)
    cbg = c[2, 2]
    ck = ink(c, cbg)
    Hc, Wc = ck.shape
    best = None
    for s in np.arange(0.40, 1.001, 0.02):
        tw, th = int(round(Wq * s)), int(round(Hq * s))
        if tw > Wc or th > Hc:
            continue
        t = cv2.resize(qk, (tw, th), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(ck, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(r)
        if best is None or mx > best[0]:
            best = (float(mx), float(s), ml)
            bestr = r
    print(f'  {os.path.basename(p)}: score={best[0]:.4f} scale={best[1]:.2f} loc={best[2]}')

# ---------- band templates (给 y 更多自由度) ----------
print('\n== band templates: rows [0:430] / [80:512] ==')
bands = {'top430': qk[0:430, :], 'bot432': qk[80:512, :]}
for p in sorted(glob.glob(QDIR + '/*.png')):
    c = load_rgb(p)
    cbg = c[2, 2]
    ck = ink(c, cbg)
    Hc, Wc = ck.shape
    line = []
    for bn, tb in bands.items():
        Hq2, Wq2 = tb.shape
        best = None
        for s in np.arange(0.40, 1.001, 0.02):
            tw, th = int(round(Wq2 * s)), int(round(Hq2 * s))
            if tw > Wc or th > Hc:
                continue
            t = cv2.resize(tb, (tw, th), interpolation=cv2.INTER_AREA)
            r = cv2.matchTemplate(ck, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(r)
            if best is None or mx > best[0]:
                best = (float(mx), float(s), ml)
        line.append(f'{bn}:{best[0]:.3f}@{best[1]:.2f}@{best[2]}')
    print(' ', os.path.basename(p), '|', ' | '.join(line))
