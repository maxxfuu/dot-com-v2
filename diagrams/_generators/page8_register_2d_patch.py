import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("reg2dpatch")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page8")

CY_ST, CY_BG = "#0c8599", "#99e9f2"
BL_ST, BL_BG = "#1971c2", "#a5d8ff"
RD = "#fa5252"

s.text(100, 12, "2D register tiling  -  one thread, a TM × TN patch", size=24)
s.text(34, 48, "BM = BN = 128,  BK = 8,  TM = TN = 8   ·   (128 × 128) / (8 × 8) = 256 threads"
               "   ·   TM + TN loads feed TM × TN FMAs", size=13, color=GRAY)

# ------------------------------------------------------------------ Bs
s.text(140, 114, "Bs   [BK × BN] = 8 × 128", size=13, color=BL_ST)
s.sq(140, 140, 326, 56, stroke=CY_ST, bg=CY_BG, fill="hachure", sw=2)
s.sq(322, 140, 20, 56, stroke=RD, bg="#ffe3e3", fill="hachure", sw=2)

# ------------------------------------------------------------------ As
s.text(58, 172, "As   [BM × BK]\n= 128 × 8", size=13, color=BL_ST, anchor="right")
s.sq(68, 216, 56, 324, stroke=BL_ST, bg=BL_BG, fill="hachure", sw=2)
s.sq(68, 318, 56, 20, stroke=RD, bg="#ffe3e3", fill="hachure", sw=2)

# ------------------------------------------------------------------ C tile
s.sq(140, 216, 326, 324, sw=2)
for gx in (322, 342):
    s.line([(gx, 216), (gx, 540)], stroke="#e9ecef", sw=1, roundness=SHARP)
for gy in (430, 450):
    s.line([(140, gy), (466, gy)], stroke="#e9ecef", sw=1, roundness=SHARP)
s.sq(322, 430, 20, 20, stroke=RD, bg="#ffe3e3", fill="hachure", sw=2)
s.text(150, 546, "C block tile   128 × 128   —   256 patches", size=12, color=GRAY)

s.arrow([(404, 472), (350, 450)], stroke=RD, sw=2)
s.text(486, 464, "one thread  →  an 8 × 8 patch of C", size=12, color=RD)
s.text(486, 484, "its TM = 8 rows of A  and  TN = 8 cols of B", size=12, color=GRAY)

# ------------------------------------------------------------------ inset
s.rect(490, 158, 448, 254, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(714, 174, "The Shape A Thread Owns", size=15, anchor="center")
s.line([(342, 430), (520, 318)], stroke=GRAY, ss="dashed", sw=1, roundness=SHARP)
s.line([(342, 450), (520, 398)], stroke=GRAY, ss="dashed", sw=1, roundness=SHARP)

s.sq(552, 196, 16, 72, stroke=GRAY_D, sw=1)
s.text(614, 206, "kernel 5  -  TM × 1 column", size=12, color=GRAY_D)
s.text(614, 226, "1 + TM = 9 loads  →  8 FMAs   ·   = 0.9 FMA / load", size=11, color=GRAY)

s.sq(520, 318, 80, 80, stroke=RD, bg="#ffe3e3", fill="cross-hatch", sw=2)
s.text(560, 302, "TN = 8", size=10, color=RD, anchor="center")
s.text(508, 358, "TM = 8", size=10, color=RD, anchor="center", valign="middle",
       angle=-math.pi / 2)
s.text(614, 326, "kernel 6  -  TM × TN patch", size=12, color=RD)
s.text(614, 346, "TM + TN = 16 loads  →  64 FMAs   ·   4 FMAs / load", size=11, color=RD)

s.save(os.path.join(OUT, "register-2d-patch.excalidraw"))
