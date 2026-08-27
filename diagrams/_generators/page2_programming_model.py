import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("progmodel")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page2")

s.text(10, 8, "CUDA Programming Model: Grid  →  Blocks  →  Threads", size=30)
s.text(28, 60, "hardware mapping: block  →  SM,   thread  →  CUDA core / lane",
       size=13, color=GRAY)


def panel_head(x, title, sub):
    s.text(x, 122, title, size=22)
    s.text(x, 158, sub, size=13, color=GRAY)


def cube_stack(ox, oy, n, pitch, cell, depth, layers, stroke, bg, hi=None,
               hi_stroke=None, hi_bg=None):
    """n x n grid drawn `layers` deep, back layers faded and offset up-right"""
    for L in range(layers - 1, -1, -1):
        op = [100, 45, 22][L] if L < 3 else 20
        dx, dy = L * depth, -L * depth
        for r in range(n):
            for c in range(n):
                is_hi = (L == 0 and hi is not None and (r, c) == hi)
                s.rect(ox + dx + c * pitch, oy + dy + r * pitch, cell, cell,
                       stroke=hi_stroke if is_hi else stroke,
                       bg=hi_bg if is_hi else bg,
                       sw=2 if L == 0 else 1, opacity=op, roundness=ROUND)


# ---------------------------------------------------------------- grid
panel_head(28, "GRID", "the complete collection of blocks")

# little 3-axis marker
s.arrow([(382, 180), (402, 143)], stroke=RED, sw=1)
s.text(404, 138, "z", size=11, color=RED)
s.arrow([(382, 180), (424, 184)], stroke=RED, sw=1)
s.text(428, 176, "x", size=11, color=RED)
s.arrow([(382, 180), (356, 238)], stroke=RED, sw=1)
s.text(348, 230, "y", size=11, color=RED)

cube_stack(78, 267, 3, 57, 52, 28, 3, ORANGE_D, BG_ORANGE,
           hi=(1, 1), hi_stroke=ORANGE, hi_bg=BG_YELLOW)

s.text(42, 494, "gridDim.x × gridDim.y × gridDim.z = 3 × 3 × 3 blocks",
       size=13, color=ORANGE_D)
s.text(42, 521, "each block is located by blockIdx.(x, y, z)", size=13)

# ---------------------------------------------------------------- block
s.text(462, 290, "zoom into\none block", size=12, color=GRAY)
s.arrow([(458, 334), (536, 334)], stroke=RED, sw=2)

panel_head(558, "THREAD BLOCK", "a group of threads that cooperate")
cube_stack(642, 258, 4, 48, 44, 22, 3, ORANGE, BG_YELLOW)

s.text(568, 474, "blockDim.x × blockDim.y × blockDim.z = 4 × 4 × 3 threads",
       size=13, color=ORANGE)
s.text(568, 500, "threads in a block share memory and can __syncthreads()", size=13)

# ---------------------------------------------------------------- thread
s.text(1004, 290, "zoom into\none thread", size=12, color=GRAY)
s.arrow([(1000, 334), (1076, 334)], stroke=RED, sw=2)

panel_head(1088, "THREAD", "the smallest unit of execution")

s.text(1276, 222, "threadIdx", size=18, color=RED, anchor="center")
s.rect(1190, 254, 174, 174, stroke=ORANGE, bg=BG_YELLOW, sw=2, roundness=ROUND)
for k in range(11):
    x = 1202 + k * 15
    pts, y, j = [], 266, 0
    while y < 418:
        pts.append((x + (3.5 if j % 2 else -3.5), y))
        y += 13; j += 1
    s.line(pts, stroke=RED, sw=1, roundness=LROUND)

s.text(1135, 474, "every thread runs the SAME kernel body,", size=15)
s.text(1135, 500, "but over its own piece of the data:", size=15)

s.save(os.path.join(OUT, "programming-model.excalidraw"))
