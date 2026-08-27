import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("reg1dov")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page7")

CY_ST, CY_BG = "#0c8599", "#99e9f2"
BL_ST, BL_BG = "#1971c2", "#a5d8ff"

s.text(100, 12, "1D register tiling  -  one thread, TM outputs", size=24)

# ------------------------------------------------------------------ B
s.text(580, 54, "N = 4096", size=13, color=GRAY, anchor="center")
s.line([(520, 74), (640, 74)], stroke=GRAY, sw=1)
s.text(455, 208, "K = 4096", size=13, color=GRAY, anchor="right", valign="middle")
s.sq(480, 88, 210, 264, sw=2)
s.sq(544, 88, 60, 264, stroke=NONE, bg=CY_BG, fill="cross-hatch", sw=1, opacity=45)
s.sq(544, 172, 60, 32, stroke=CY_ST, bg=CY_BG, fill="hachure", sw=2)
s.text(496, 144, "Bs", size=14)
s.arrow([(516, 156), (542, 172)], sw=1)
s.text(574, 152, "BN=64", size=11, color=BL_ST, anchor="center")
s.text(610, 180, "BK=8", size=11, color=BL_ST)
s.arrow([(574, 226), (574, 306)], stroke=BL_ST, sw=3)
s.text(676, 322, "B", size=22, anchor="right")

# ------------------------------------------------------------------ A
s.text(255, 352, "K = 4096", size=13, color=GRAY, anchor="center")
s.line([(160, 372), (390, 372)], stroke=GRAY, sw=1)
s.text(68, 480, "M = 4096", size=13, color=GRAY, anchor="right", valign="middle")
s.sq(105, 388, 300, 192, sw=2)
s.sq(105, 462, 300, 48, stroke=NONE, bg=BL_BG, fill="cross-hatch", sw=1, opacity=45)
s.sq(232, 462, 30, 48, stroke=BL_ST, bg=BL_BG, fill="hachure", sw=2)
s.text(168, 434, "As", size=14)
s.arrow([(190, 446), (230, 462)], sw=1)
s.text(268, 478, "BM=64", size=11, color=BL_ST)
s.text(224, 522, "BK=8", size=11, color=BL_ST, anchor="center")
s.arrow([(318, 488), (394, 488)], stroke=BL_ST, sw=3)
s.text(392, 550, "A", size=22, anchor="right")

# ------------------------------------------------------------------ C
s.sq(480, 388, 210, 192, sw=2)
s.line([(568, 388), (568, 580)], stroke="#e9ecef", sw=1, roundness=SHARP)
s.line([(608, 388), (608, 580)], stroke="#e9ecef", sw=1, roundness=SHARP)
s.sq(568, 462, 40, 48, stroke=GREEN, bg=BG_GREEN, fill="hachure", sw=2)
s.text(676, 550, "C", size=22, anchor="right")
s.text(488, 604, "one block  →  one 64×64 tile of C", size=12, color=GRAY)

# ------------------------------------------------------------------ zoom
ZX, ZY, ZW, ZH, N = 768.0, 350.0, 232.0, 230.0, 16
s.text(762, 318, "zoom: 16×16 corner of the C tile   ·   1 cell = 1 element of C",
       size=12, color=GRAY)
s.line([(608, 462), (ZX, ZY)], stroke=GRAY, ss="dashed", sw=1, roundness=SHARP)
s.line([(608, 510), (ZX, ZY + ZH)], stroke=GRAY, ss="dashed", sw=1, roundness=SHARP)
grid_box(s, ZX, ZY, ZW, ZH, N, N, stroke=GREEN, sw=2, line="#dee2e6", ls="solid")
cw, ch = ZW / N, ZH / N
s.sq(ZX + 4 * cw, ZY, cw, 8 * ch, stroke=NONE, bg="#37b24d", fill="solid", sw=1)
s.sq(ZX + 5 * cw, ZY, cw, 8 * ch, stroke=NONE, bg="#b2f2bb", fill="solid", sw=1)

s.arrow([(1070, ZY + 4 * ch), (ZX + 6.4 * cw, ZY + 4 * ch)], stroke=GREEN, sw=2)
s.text(1085, ZY + 40, "one thread  →  TM = 8 outputs,\nsame column, consecutive rows",
       size=13, color=GREEN)
s.text(1085, ZY + 92, "next thread  →  the column\nbeside it", size=13, color=GRAY)
s.text(762, 604, "kernel 4 mapping was 1 cell = 1 thread;   now 1 column of 8 cells = 1 thread",
       size=12, color=GRAY)

s.save(os.path.join(OUT, "register-1d-overview.excalidraw"))
