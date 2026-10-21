# -*- coding: utf-8 -*-
"""可视化核对: 刻度区 vs 命中位置"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
import cv2

QDIR = r'D:/HDC/data/2223'
QRY = r'D:/HDC/data/核伤251003_0646_22232_20251110_084801/核伤251003_0646_22232_20251110_084801.png'
OUT = r'D:/HDC/tools/_out'


def load_rgb(p):
    im = Image.open(p).convert('RGBA')
    bgw = Image.new('RGBA', im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bgw, im).convert('RGB')
    return np.asarray(im)


def save(a, name):
    Image.fromarray(a).save(os.path.join(OUT, name))
    print('saved', name, a.shape)


ah = load_rgb(QDIR + '/a有.png')
q = load_rgb(QRY)

# 1) 顶部刻度区（左半/右半）
save(ah[60:140, 0:944], 'strip_a_has_left.png')
save(ah[60:140, 944:1888], 'strip_a_has_right.png')
save(q[60:140, :], 'strip_query.png')
# 2) 命中位置 vs 查询图 并排
for nm, x in [('a有.png', 1311), ('a无2.png', 1279), ('a无.png', 873)]:
    c = load_rgb(QDIR + '/' + nm)
    reg = c[:, x:x + 512]
    gap = np.full((512, 8, 3), 0, np.uint8)
    side = np.concatenate([q, gap, reg], axis=1)
    save(side, f'sbs_{nm.replace(".png","")}_{x}.png')
