# -*- coding: utf-8 -*-
"""从 QDRL 论文 Fig. 2 的渲染图反推历史窗口曲线的数据点与峰值位置。

只读脚本：读取 QDRL_page14.png，输出
  - fig2_peak_measurement.json  机器可读的测量结果
  - fig2_peaks_annotated.png    标注了检测点与峰值的复查图

依赖：Pillow、numpy。不访问网络，不读取数据集，不训练。

方法
----
1. Fig. 2 使用 matplotlib tab10 配色，MRR 为蓝色 (31,119,180)。
2. 实心圆 marker 处的蓝色像素垂直跨度明显大于 1.5px 的连线，据此定位数据点。
3. 每个子图右上角有 legend；legend 色块总是全图最高，按「y 中心不低于子图
   顶部 + LEGEND_DROP 像素」剔除。
4. **不做绝对像素定标**。改用相邻间距的比例：图中最宽的间距是最后两个点，
   即 m = 30 -> 60（30 个单位），以此为单位反推全部网格值。锚定依据是论文
   §5.6 正文写明的 m = 1 / m = 3 / m >= 30 三个网格点，以及 x 轴上限恰为 60。
5. 四个子图各自独立反推；四组结果收敛到同一组整数网格，构成交叉验证。
6. 峰值 = MRR 曲线中 y 像素最小（位置最高）的数据点。同时记录次高点的像素
   差，用于判断峰值是否显著。

已知限制
--------
- 本脚本只测量图上画出的点，不能给出未画出的窗口的数值。
- 像素级判读的横向精度约 ±0.5 个窗口单位，纵向差值按像素报告，不换算成 MRR。
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).absolute().parent
SRC = HERE / 'QDRL_page14.png'
OUT_JSON = HERE / 'fig2_peak_measurement.json'
OUT_PNG = HERE / 'fig2_peaks_annotated.png'

BLUE = (31, 119, 180)      # tab10 blue -> MRR
LEGEND_DROP = 130          # px below subplot top: legend lives above this
SPAN_MIN = 5               # px: marker columns are thicker than the 2px line
WIDEST_GAP_UNITS = 30      # m = 30 -> 60

# 四个子图的裁剪框 (x0, y0, x1, y1)，覆盖绘图区并留出刻度标签余量
BOXES = {
    'ICEWS14':    (140, 880, 780, 1345),
    'ICEWS05-15': (775, 880, 1400, 1345),
    'ICEWS18':    (140, 1370, 780, 1855),
    'YAGO':       (775, 1370, 1400, 1855),
}


def blue_mask(box, tol=60):
    x0, y0, x1, y1 = box
    sub = np.asarray(Image.open(SRC).convert('RGB')).astype(int)[y0:y1, x0:x1]
    return (np.abs(sub - np.array(BLUE)).sum(axis=2) < tol)


def cluster(values, gap=2):
    groups, cur = [], [values[0]]
    for v in values[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return groups


def detect_markers(box):
    x0, y0, _, _ = box
    mask = blue_mask(box)
    rows = []
    for c in np.nonzero(mask.any(axis=0))[0]:
        ys = np.nonzero(mask[:, c])[0]
        rows.append((c, ys.max() - ys.min() + 1, ys.mean()))
    thick = [r for r in rows if r[1] >= SPAN_MIN]
    points = []
    for group in cluster([r[0] for r in thick]):
        c = int(round(np.mean(group)))
        ys = np.nonzero(mask[:, c])[0]
        y_center = float(ys.mean())
        if y_center < LEGEND_DROP:          # legend swatch, not a data point
            continue
        points.append(dict(x_px=c + x0, y_px=round(y_center + y0, 1)))
    return sorted(points, key=lambda p: p['x_px'])


def main():
    image = Image.open(SRC).convert('RGB')
    report = {'source': SRC.name, 'source_size': list(image.size),
              'curve': 'MRR (tab10 blue)', 'datasets': {}}

    for name, box in BOXES.items():
        pts = detect_markers(box)
        rec = {'markers': len(pts), 'points': pts}

        if len(pts) >= 2:
            gaps = [b['x_px'] - a['x_px'] for a, b in zip(pts, pts[1:])]
            rec['gaps_px'] = gaps
            unit = max(gaps) / float(WIDEST_GAP_UNITS)
            rec['px_per_unit'] = round(unit, 3)
            grid, cum = [1.0], 1.0
            for g in gaps:
                cum += g / unit
                grid.append(round(cum, 2))
            rec['grid'] = grid
            rec['grid_rounded'] = [int(round(v)) for v in grid]
            rec['grid_max_deviation'] = round(
                max(abs(v - round(v)) for v in grid), 3)

            best = min(pts, key=lambda p: p['y_px'])
            rec['peak_index'] = pts.index(best)
            rec['peak_grid'] = grid[pts.index(best)]
            rec['peak_x_px'] = best['x_px']
            rec['peak_y_px'] = best['y_px']
            ordered = sorted(p['y_px'] for p in pts)
            rec['runner_up_y_px'] = round(ordered[1], 1)
            rec['peak_margin_px'] = round(ordered[1] - ordered[0], 1)
            rec['y_spread_px'] = round(ordered[-1] - ordered[0], 1)

        report['datasets'][name] = rec

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8')

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for rec in report['datasets'].values():
        for i, p in enumerate(rec['points']):
            peak = i == rec.get('peak_index')
            r = 9 if peak else 6
            draw.ellipse([p['x_px'] - r, p['y_px'] - r, p['x_px'] + r, p['y_px'] + r],
                         outline=(220, 20, 20) if peak else (0, 150, 0), width=2)
            if peak:
                draw.text((p['x_px'] + 10, p['y_px'] - 14),
                          'peak m=%d' % rec['peak_grid'], fill=(220, 20, 20))
    canvas.save(OUT_PNG)
    print('wrote', OUT_JSON.name, 'and', OUT_PNG.name)
    for name, rec in report['datasets'].items():
        print('%-11s grid=%s peak=m%s margin=%spx' %
              (name, rec.get('grid_rounded'), rec.get('peak_grid'),
               rec.get('peak_margin_px')))


if __name__ == '__main__':
    main()
