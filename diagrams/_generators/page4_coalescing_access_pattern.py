import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("coalA")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page4")

s.text(355, 8, "Global Memory Coalescing on Matrix A", size=18, anchor="center")

NC, NR = 28, 26
LINE = "#e9ecef"

# ---------------------------------------------------------------- naive
s.sq(8, 40, 340, 350, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(44, 56, "Naive Implementation", size=14, color=RED)
s.text(54, 78, "Matrix A ( M x K )   row-major", size=11)

MX, MY, MW, MH = 58, 104, 280, 264
s.text(58, 92, "k=0  k=1", size=8, color=GRAY)
grid_box(s, MX, MY, MW, MH, NC, NR, sw=1, line=LINE, ls="solid")
s.sq(MX, MY, MW / NC, MH, stroke="#ffc9c9", bg="#ffe3e3", sw=1)

s.arrow([(46, 106), (46, 364)], stroke=RED, sw=1)
for i, t in enumerate(["T0", "T1", "T2", "·", "·", "·"]):
    s.text(38, 106 + i * 14, t, size=8, color=RED, anchor="right")
s.text(38, 352, "T31", size=8, color=RED, anchor="right")

# ---------------------------------------------------------------- coalesced
s.sq(368, 40, 334, 350, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(404, 56, "Global Memory Coalescing", size=14, color=GREEN)
s.text(414, 74, "Matrix A ( M x K )   row-major", size=11)

CX, CY, CW_, CH = 410, 118, 280, 250
s.text(CX, 94, "T0", size=9, color=GREEN)
s.text(CX + CW_ / 2, 94, "lane 0 -> lane 31", size=9, color=GREEN, anchor="center")
s.text(CX + CW_, 94, "T31", size=9, color=GREEN, anchor="right")
s.arrow([(CX, 110), (CX + CW_ - 4, 110)], stroke=GREEN, sw=1)

grid_box(s, CX, CY, CW_, CH, NC, NR, sw=1, line=LINE, ls="solid")
ROWY = CY + 4 * CH / NR
cell = CW_ / NC
for i, col in enumerate(ramp("#1b7a32", "#ebfbee", NC)):
    s.sq(CX + i * cell, ROWY, cell, CH / NR, stroke=col, bg=col, sw=1)
s.text(CX - 6, ROWY + CH / NR / 2, "row r", size=9, color=GREEN,
       anchor="right", valign="middle")

s.save(os.path.join(OUT, "coalescing-access-pattern.excalidraw"))
