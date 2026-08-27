import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("naiveacc")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page3")

s.text(404, 8, "Naive Kernel: Thread Block Memory Access Pattern", size=17, anchor="center")

# ------------------------------------------------ left: the thread block
s.sq(8, 36, 314, 400, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(30, 56, "Logical Thread Organization", size=11)

s.text(128, 106, "threadIdx.x", size=14, color=ORANGE)
s.arrow([(96, 126), (288, 126)], stroke=ORANGE, sw=2)
s.text(34, 256, "threadIdx.y", size=14, color=ORANGE,
       anchor="center", valign="middle", angle=-math.pi / 2)
s.arrow([(52, 152), (52, 360)], stroke=ORANGE, sw=2)

GX, GY, GW, GH = 72, 150, 216, 212
grid_box(s, GX, GY, GW, GH, 8, 8)
s.sq(GX, GY, GW, 20, sw=1)                                   # the first row of threads
s.sq(GX, GY, 27, 20, stroke=RED, bg=BG_RED, sw=2)
s.sq(GX + 27, GY, 27, 20, stroke=GREEN, bg=BG_GREEN, sw=2)

s.text(64, 378, "Thread 0: threadIdx (0,0)", size=13, color=RED)
s.text(64, 400, "Thread 1:  threadIdx (1,0)", size=13, color=GREEN)

# ------------------------------------------------ right: the three matrices
s.sq(338, 36, 462, 400, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(390, 58, "A × B = C", size=17, color=GRAY, font=CODE)

# B, with the column this thread walks
s.text(614, 52, "B  (K × N)", size=11)
grid_box(s, 588, 70, 154, 152, 8, 8)
s.sq(588, 70, 14, 152, stroke=BLUE, bg=BG_BLUE, sw=1)

# A, with the two rows the two threads walk
s.text(438, 234, "A  (M × K)", size=11)
grid_box(s, 412, 252, 154, 154, 8, 8)
s.sq(412, 254, 154, 16, stroke=RED, bg=BG_RED, sw=1)
s.sq(412, 276, 154, 16, stroke=GREEN, bg=BG_GREEN, sw=1)

# C, with the one element each thread produces
s.text(614, 234, "C  (M × N)", size=11)
grid_box(s, 588, 252, 154, 154, 8, 8)
s.sq(588, 254, 14, 16, stroke=RED, bg=BG_RED, fill="cross-hatch", sw=1)
s.sq(588, 272, 14, 16, stroke=GREEN, bg=BG_GREEN, fill="cross-hatch", sw=1)

s.arrow([(101, 158), (250, 166), (400, 202), (520, 244), (584, 258)],
        stroke=RED, sw=2, roundness=LROUND)
s.arrow([(101, 176), (260, 228), (420, 288), (520, 284), (584, 278)],
        stroke=GREEN, sw=2, roundness=LROUND)

s.save(os.path.join(OUT, "naive-access-pattern.excalidraw"))
