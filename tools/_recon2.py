# -*- coding: utf-8 -*-
"""recon2: 调色板 + 刻度检测 + 彩色标记簇检测"""
import sys, os, glob
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
import cv2

QDIR = r'D:/HDC/data/2223'
QRY = r'D:/HDC/data/核伤251003_0646_22232_20251110_084801/核伤251003_0646_22232_20251110_084801.png'


def load_rgb(p):
    im = Image.open(p).convert('RGBA')
    bgw = Image.new('RGBA', im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bgw, im).convert('RGB')
    return np.asarray(im)


def palette(name, a):
    qq = (a // 16 * 16).reshape(-1, 3)
    cnt = Counter(map(tuple, qq[::11]))
    print(f'--- palette top18 [{name}]')
    for col, n in cnt.most_common(18):
        print(f'    {col} x{n}')


def ticks(name, a):
    R, G, B = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    dark = (R < 130) & (G < 130) & (B < 130)
    W = a.shape[1]
    rows = np.where(dark.sum(1) > W * 0.5)[0]
    if not len(rows):
        print(f'--- ticks [{name}] no top line'); return
    t0 = rows[0]
    band = dark[t0 + 2:t0 + 18, :].sum(0)
    cols = np.where(band >= 4)[0]
    clus = []
    for i in cols:
        if clus and i - clus[-1][-1] <= 2:
            clus[-1].append(i)
        else:
            clus.append([i])
    print(f'--- ticks [{name}] top_black_row={t0}')
    for c in clus:
        center = int(np.mean(c))
        vlen = int(dark[t0 - 1:t0 + 20, center].sum())
        wid = len(c)
        print(f'    x={center:5d} w={wid} vlen={vlen}')


def clusters(name, a):
    R, G, B = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    masks = {
        'purple': (R > 170) & (B > 170) & (G < 140),
        'green':  (G > 150) & (R < 140) & (B < 140) & (G - np.maximum(R, B) > 60),
        'orange': (R > 140) & (G > 40) & (G < 150) & (B < 90) & (R - B > 90) & (R - G > 50),
        'blue':   (B > 140) & (R < 110) & (G < 110),
    }
    print(f'--- clusters [{name}]')
    for cn, m in masks.items():
        m8 = (m.astype(np.uint8)) * 255
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m8, 8)
        items = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 2:
                continue
            items.append((x, y, w, h, area, cent[i]))
        items.sort()
        print(f'  [{cn}] {len(items)} blobs')
        for x, y, w, h, area, c in items:
            print(f'    px=({x:4d},{y:3d}) w={w:3d} h={h:3d} area={area:4d} cx={c[0]:.0f}')


for p in sorted(glob.glob(QDIR + '/*.png')) + [QRY]:
    nm = os.path.basename(p)
    a = load_rgb(p)
    palette(nm, a)
    ticks(nm, a)
    clusters(nm, a)
