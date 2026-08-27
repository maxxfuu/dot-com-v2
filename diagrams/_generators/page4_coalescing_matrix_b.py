import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("coalB")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page4")

s.text(474, 8, "Global Memory Coalescing on Matrix B", size=18, anchor="center")

NC, NR = 24, 24
LINE = "#e9ecef"
GY, GH = 158.0, 240.0

# ---------------------------------------------------------------- naive
s.sq(8, 44, 460, 394, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(28, 56, "Naive Implementation", size=14, color=RED)
s.text(28, 80, "col = blockIdx.y * blockDim.y + threadIdx.y", size=11)
s.text(52, 100, "Matrix B ( K x N )   row-major", size=11)
s.text(28, 116, "warp lanes   T0 -> T31", size=8, color=RED)

BX, BW = 52.0, 246.0
for i, col in enumerate(ramp("#a02020", "#e88f8f", 32)):
    s.sq(BX + i * BW / 32, 124, BW / 32, 16, stroke=col, bg=col, sw=1)

HITX = BX + 11 * BW / NC          # the single column every lane resolves to
for i in range(21):
    s.line([(BX + 4 + i * (BW - 8) / 20, 141), (HITX + 5, 154)],
           stroke="#ffa8a8", sw=1, roundness=SHARP)

grid_box(s, BX, GY, BW, GH, NC, NR, sw=1, line=LINE, ls="solid")
s.sq(HITX, GY, BW / NC, GH, stroke="#ffc9c9", bg="#ffe3e3", sw=1)
s.sq(HITX, GY, BW / NC, GH / NR, stroke=RED_D, bg=RED_D, sw=1)

s.text(30, 138, "k", size=12, color=RED)
s.arrow([(38, 150), (38, 400)], stroke=RED, sw=1)

s.text(310, 152, "all 32 lanes -> B[k][col]", size=10, color=GRAY_D)
s.text(310, 174, "one address, one 32 B sector,\n4 B used, 31 lanes served\nby broadcast",
       size=10, color=RED)
s.text(306, 224, "each k step the warp moves\none row down the SAME column",
       size=9, color=GRAY)
s.text(28, 418, "32 lanes -> 1 address        broadcast        1 sector (32 B) loaded, 4 B used",
       size=10, color=RED)

# ---------------------------------------------------------------- coalesced
s.sq(480, 44, 468, 394, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(500, 56, "Global Memory Coalescing", size=14, color=GREEN)
s.text(500, 80, "col = blockIdx.x * blockDim.x + threadIdx.x", size=11)
s.text(524, 100, "Matrix B ( K x N )   row-major", size=11)
s.text(500, 116, "warp lanes  T0 -> T31", size=8, color=GREEN)

CX, CW_ = 524.0, 266.0
for i, col in enumerate(ramp("#1b7a32", "#8ce99a", 32)):
    s.sq(CX + i * CW_ / 32, 124, CW_ / 32, 16, stroke=col, bg=col, sw=1)

for i in range(8):
    x = CX + (i + 0.5) * CW_ / 8
    s.arrow([(x, 142), (x, 156)], stroke=GREEN, sw=1)

grid_box(s, CX, GY, CW_, GH, NC, NR, sw=1, line=LINE, ls="solid")
cell = CW_ / NC
for i, col in enumerate(ramp("#1b7a32", "#d3f9d8", NC)):
    s.sq(CX + i * cell, GY, cell, GH / NR, stroke=col, bg=col, sw=1)

s.text(502, 138, "k", size=12, color=GREEN)
s.arrow([(510, 150), (510, 400)], stroke=GREEN, sw=1)

s.text(798, 152, "32 lanes -> B[k][0..31]", size=10, color=GREEN)
s.text(798, 174, "32 adjacent floats, sequential\nin physical memory, packed\ninto 4 x 32 B sectors",
       size=9, color=GREEN)
s.text(798, 224, "each k step the WHOLE warp\ndrops one row down", size=9, color=GRAY)
s.text(500, 418, "32 lanes -> 32 adjacent addresses       128 B contiguous       4 sectors, all used",
       size=10, color=GREEN)

s.save(os.path.join(OUT, "coalescing-matrix-b.excalidraw"))
