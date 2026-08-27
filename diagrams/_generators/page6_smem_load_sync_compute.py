import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("loadsync")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page6")

CY_ST, CY_BG = "#0c8599", "#99e9f2"
BL_ST, BL_BG = "#1971c2", "#a5d8ff"

s.text(668, 12, "Inside one block:  load → sync → compute → sync", size=24, anchor="center")

# ================================================================ 1 · load
s.rect(14, 72, 698, 458, sw=2, roundness=ROUND)
s.text(30, 84, "1 · Load   all 1024 threads copy one element each", size=15)


def tile(x, y, w, h, st, bg, fill, cell_st, cell_bg, caption):
    s.sq(x, y, w, h, stroke=st, bg=bg, fill=fill, sw=2)
    s.sq(x + w / 2 - 11, y + h / 2 - 11, 22, 22, stroke=cell_st, bg=cell_bg,
         fill="solid", sw=1)
    s.text(x, y + h + 10, caption, size=11, color=GRAY_D)


tile(44, 140, 160, 150, "#a5d8ff", "#e3f6fb", "cross-hatch", BL_ST, BL_ST, "A tile  in GMEM")
s.arrow([(215, 215), (300, 215)], stroke=BLUE, sw=3)
tile(315, 140, 155, 150, BL_ST, BL_BG, "hachure", BL_ST, BL_ST, "As  in SMEM")

tile(44, 335, 160, 155, "#99e9f2", "#e6fcfd", "cross-hatch", CY_ST, CY_ST, "B tile  in GMEM")
s.arrow([(215, 412), (300, 412)], stroke=CY_ST, sw=3)
tile(315, 335, 155, 155, CY_ST, CY_BG, "hachure", CY_ST, CY_ST, "Bs  in SMEM")

s.text(500, 150, "tx = threadIdx.x;\nty = threadIdx.y;", size=13, font=CODE)
s.text(500, 196, "As[ty][tx] = A[ty*K + tx];\nBs[ty][tx] = B[ty*N + tx];", size=13, font=CODE)
s.text(500, 242, "__syncthreads();", size=13, font=CODE)

s.text(500, 368, "thread (tx,ty) owns\nexactly one cell of As\nand one of Bs",
       size=13, color=GRAY_D)
s.text(500, 438, "sync = \"tile is whole\nbefore anyone reads it\"", size=13, color=GRAY_D)

# ============================================================= 2 · compute
s.rect(748, 72, 574, 458, sw=2, roundness=ROUND)
s.text(766, 84, "2 · Compute   32 MACs per thread, no GMEM traffic", size=15)

BSX, BSY, BSW, BSH = 1040, 140, 160, 180
ASX, ASY, ASW, ASH = 818, 340, 182, 180
CX, CY_, CW, CH = 1040, 340, 180, 180

# faint guides tying the Bs column and the As row to the C element
s.line([(1100, BSY + BSH), (1100, CY_)], stroke="#dee2e6", sw=1, roundness=SHARP)
s.line([(1122, BSY + BSH), (1122, CY_)], stroke="#dee2e6", sw=1, roundness=SHARP)
s.line([(ASX + ASW, 405), (CX, 405)], stroke="#dee2e6", sw=1, roundness=SHARP)
s.line([(ASX + ASW, 427), (CX, 427)], stroke="#dee2e6", sw=1, roundness=SHARP)

s.sq(BSX, BSY, BSW, BSH, stroke=CY_ST, bg=CY_BG, fill="hachure", sw=2)
s.sq(1100, BSY, 22, BSH, stroke=NONE, bg=CY_ST, fill="solid", sw=1, opacity=45)
s.text(1245, 196, "Bs", size=18)
s.text(1245, 224, "col tx", size=12, color=GRAY)

s.sq(ASX, ASY, ASW, ASH, stroke=BL_ST, bg=BL_BG, fill="hachure", sw=2)
s.sq(ASX, 405, ASW, 22, stroke=NONE, bg=BL_ST, fill="solid", sw=1, opacity=45)
s.text(806, 396, "As", size=18, anchor="right")
s.text(806, 424, "row ty", size=12, color=GRAY, anchor="right")

s.sq(CX, CY_, CW, CH, stroke=GREEN, bg=BG_GREEN, fill="hachure", sw=2)
s.sq(1100, 405, 22, 22, stroke=GREEN, bg=GREEN, fill="solid", sw=1)
s.text(1245, 400, "C tile", size=15)
s.text(1150, 462, "sum → C[ty][tx]", size=12, color=GREEN, anchor="center")
s.arrow([(1180, 452), (1128, 424)], stroke=GREEN, sw=2)

s.save(os.path.join(OUT, "smem-load-sync-compute.excalidraw"))
