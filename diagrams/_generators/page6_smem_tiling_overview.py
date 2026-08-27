import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("smemtile")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page6")

CY_ST, CY_BG = "#0c8599", "#99e9f2"     # the B / Bs teal
BL_ST, BL_BG = "#1971c2", "#a5d8ff"     # the A / As blue

s.text(148, 12, "SMEM tiling  -  one 32×32 output tile per thread block", size=24)
s.text(148, 48, "BLOCKSIZE = 32   ·   blockDim = 32×32 = 1024 threads   ·   "
                "1 thread → 1 element of C", size=13, color=GRAY)

# ------------------------------------------------------------------ B
BX, BY, BW, BH = 518, 115, 210, 260
s.text(623, 84, "N = 4096", size=13, color=GRAY, anchor="center")
s.line([(555, 104), (692, 104)], stroke=GRAY, sw=1)
s.text(470, 245, "K = 4096", size=13, color=GRAY, anchor="right", valign="middle")
s.line([(505, 128), (505, 362)], stroke=GRAY, sw=1)
s.sq(BX, BY, BW, BH, sw=2)
s.sq(584, BY, 60, BH, stroke=NONE, bg=CY_BG, fill="cross-hatch", sw=1, opacity=45)
s.sq(584, 196, 60, 50, stroke=CY_ST, bg=CY_BG, fill="hachure", sw=2)
s.text(536, 158, "&B", size=14)
s.arrow([(556, 174), (582, 194)], sw=1)
s.text(614, 172, "32", size=11, anchor="center")
s.text(652, 196, "32", size=11)
s.arrow([(614, 262), (614, 348)], stroke=BLUE, sw=3)
s.text(732, 108, "cCol = blockIdx.x", size=12, color=GRAY_D)
s.text(BX + BW - 16, BY + BH - 26, "B", size=22, anchor="right")

# ------------------------------------------------------------------ A
AX, AY, AW, AH = 145, 420, 310, 200
s.text(300, 386, "K = 4096", size=13, color=GRAY, anchor="center")
s.line([(150, 406), (450, 406)], stroke=GRAY, sw=1)
s.text(100, 512, "M = 4096", size=13, color=GRAY, anchor="right", valign="middle")
s.line([(130, 428), (130, 612)], stroke=GRAY, sw=1)
s.text(60, 384, "cRow = blockIdx.y", size=12, color=GRAY_D)
s.sq(AX, AY, AW, AH, sw=2)
s.sq(AX, 496, AW, 50, stroke=NONE, bg=BL_BG, fill="cross-hatch", sw=1, opacity=45)
s.sq(245, 490, 60, 55, stroke=BL_ST, bg=BL_BG, fill="hachure", sw=2)
s.text(198, 456, "&A", size=14)
s.arrow([(218, 470), (243, 488)], sw=1)
s.text(270, 550, "32", size=11, anchor="center")
s.text(313, 512, "32", size=11)
s.arrow([(320, 522), (418, 522)], stroke=BLUE, sw=3)
s.text(AX + AW - 16, AY + AH - 26, "A", size=22, anchor="right")

# ------------------------------------------------------------------ C
CX, CY, CW, CH = 518, 420, 210, 200
grid_box(s, CX, CY, CW, CH, 3, 3, sw=2, line="#e9ecef", ls="solid")
s.sq(588, 490, 60, 55, stroke=GREEN, bg=BG_GREEN, fill="hachure", sw=2)
s.text(541, 456, "&C", size=14)
s.arrow([(561, 470), (586, 488)], sw=1)
s.text(CX + CW - 16, CY + CH - 26, "C", size=22, anchor="right")

# ------------------------------------------------------------------ SMEM
s.rect(805, 325, 315, 245, stroke=VIOLET, ss="dashed", sw=1, roundness=ROUND)
s.text(828, 340, "shared memory  (per block)", size=13, color=VIOLET)
s.sq(828, 382, 120, 120, stroke=BL_ST, bg=BL_BG, fill="hachure", sw=2)
s.sq(975, 382, 120, 120, stroke=CY_ST, bg=CY_BG, fill="hachure", sw=2)
s.text(888, 508, "As[32][32]", size=13, anchor="center")
s.text(1035, 508, "Bs[32][32]", size=13, anchor="center")
s.text(828, 534, "2 · 32 · 32 · 4 B  =  8 KB  resident", size=11, color=GRAY)

s.arrow([(652, 224), (740, 210), (880, 250), (1000, 340), (1032, 378)],
        stroke=CY_ST, sw=2, roundness=LROUND)
s.text(778, 208, "GMEM → SMEM", size=13, color=CY_ST)

s.arrow([(300, 624), (306, 682), (500, 704), (700, 694), (782, 600),
         (786, 470), (826, 424)], stroke=BLUE, sw=2, roundness=LROUND)
s.text(355, 650, "GMEM → SMEM  (coalesced)", size=13, color=BLUE)

s.save(os.path.join(OUT, "smem-tiling-overview.excalidraw"))
