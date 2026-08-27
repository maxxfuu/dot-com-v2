import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("sectorutil")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page4")

X0, X1 = 40.0, 605.0
CW = (X1 - X0) / 32.0

s.text(324, 10, "What The Memory System Sees", size=18, anchor="center")
s.sq(10, 44, 628, 184, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)


def strip(y, h, colors, label_y):
    for i in range(32):
        s.sq(X0 + i * CW, y, CW, h, stroke="#dee2e6",
             bg=colors[i], fill="solid" if colors[i] != NONE else "solid", sw=1)
    for k in range(5):                       # 32 B sector walls
        x = X0 + k * 8 * CW
        s.line([(x, y - 4), (x, y + h + 4)], sw=1.5, roundness=SHARP)
    for k in range(4):
        s.text(X0 + (k + 0.5) * 8 * CW, label_y, "sector %d" % k,
               size=10, color=GRAY, anchor="center")


# one 4 B word taken out of each 32 B sector
naive = [("#c92a2a" if i % 8 == 0 else NONE) for i in range(32)]
strip(78, 30, naive, 64)
for k in range(4):
    s.text(X0 + k * 8 * CW + 1, 112, "T%d" % k, size=11, color=RED)

s.line([(24, 136), (624, 136)], stroke="#dee2e6", ss="dashed", sw=1, roundness=SHARP)

# every byte of every sector used, lane 0 through lane 31
strip(160, 30, ramp("#1b7a32", "#d3f9d8", 32), 146)
s.text(X0 + 1, 204, "T0", size=11, color=GREEN)
s.text((X0 + X1) / 2, 206, "lane 0 -> lane 31", size=10, color=GREEN, anchor="center")
s.text(X1 - 4, 204, "T31", size=11, color=GREEN, anchor="right")

s.save(os.path.join(OUT, "sector-utilization.excalidraw"))
