import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("reg1dinner")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page7")

CY_ST, CY_BG = "#0c8599", "#99e9f2"
BL_ST, BL_BG = "#1971c2", "#a5d8ff"
TM = 8

s.rect(10, 10, 840, 478, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(34, 22, "One thread, one step of the BK loop", size=18)
s.text(34, 52, "2 reads → 1 result       becomes       1 + TM reads → TM results", size=14)

# ------------------------------------------------------- the As column
s.text(96, 82, "As column (SMEM)", size=11, color=GRAY)
s.sq(100, 100, 46, 350, stroke=BL_ST, bg=BL_BG, fill="hachure", sw=1)
CY0, CELL = 200.0, 20.0
for i in range(TM):
    s.sq(100, CY0 + i * CELL, 46, CELL, stroke=BL_ST, bg="#8ec5f0", fill="solid", sw=1)
s.text(18, 260, "TM = 8", size=13, color=BL_ST)
s.text(18, 282, "A values", size=11, color=GRAY)
s.text(88, 456, "As[ · ][dot]", size=11, color=GRAY)

# ------------------------------------------------------- the Bs row
s.text(286, 82, "Bs row  (SMEM)", size=11, color=GRAY)
s.sq(286, 100, 300, 40, stroke=CY_ST, bg=CY_BG, fill="hachure", sw=1)
s.sq(430, 100, 26, 40, stroke=CY_ST, bg=CY_ST, fill="solid", sw=1)
s.text(468, 86, "1 SMEM read", size=11, color=CY_ST)
s.arrow([(590, 120), (636, 120)], stroke=CY_ST, sw=2)

s.text(664, 64, "broadcast: 1 value,\nreused TM times", size=12, color=VIOLET)
b = s.rect(640, 100, 152, 42, stroke=VIOLET, sw=2)
s.label(b, "Btmp  (register)", size=14, color=VIOLET)
s.arrow([(716, 148), (716, 192)], stroke=VIOLET, sw=2)

s.text(214, 172, "TM = 8 fused multiply-adds:   acc[i] += As[i] × Btmp", size=14)

# ------------------------------------------------------- the accumulators
AX, AY, AW, AH, PITCH = 644.0, 200.0, 148.0, 19.0, 21.0
for i in range(TM):
    r = s.sq(AX, AY + i * PITCH, AW, AH, stroke=GREEN, bg=BG_GREEN, fill="hachure", sw=1)
    s.label(r, "acc[%d]" % i, size=11, color=GREEN_D)
    s.arrow([(148, CY0 + (i + 0.5) * CELL), (AX - 4, AY + i * PITCH + AH / 2)],
            stroke=BL_ST, sw=1)
s.text(644, AY + TM * PITCH + 8, "TM accumulators, live in registers", size=11, color=GRAY)

s.save(os.path.join(OUT, "register-1d-inner-loop.excalidraw"))
