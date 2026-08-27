import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("gb203die")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page2")

GPC_BG, GPC_ST = "#f1f7fe", "#4dabf7"
TPC_BG, TPC_ST = "#e7f5ff", "#a5d8ff"
SM_BG,  SM_ST  = "#c5e2fb", BLUE

s.text(10, 8, "NVIDIA RTX 5080  -  GB203 GPU Die", size=30, color=GRAY)
s.text(10, 54, "7 GPCs, 42 TPCs, 84 SMs, 10,752 CUDA Cores, 64 MB L2, 8 x 32-bit GDDR7 "
               "Memory Controllers (256-bit)", size=17, color=GRAY)

s.sq(18, 100, 1552, 1050, sw=2, roundness=ROUND)


def bar(x, y, w, h, txt, bg, st, size, color=BLACK, lh=1.25, sw=1):
    r = s.sq(x, y, w, h, stroke=st, bg=bg, sw=sw, roundness=ROUND)
    s.label(r, txt, size=size, color=color, lh=lh)
    return r


bar(38, 126, 742, 32, "PCI Express 5.0 x 16 Host Interface", BG_VIOLET, VIOLET, 15)
bar(800, 126, 752, 32, "GigaThread Engine  -  Work Distribution & Scheduling",
    BG_VIOLET, VIOLET, 15)

GPC_W, GPC_H = 376, 418
GPCS = [(32, 172), (425, 172), (807, 172), (1195, 172),
        (32, 665), (425, 665), (807, 665)]

for g, (gx, gy) in enumerate(GPCS):
    s.sq(gx, gy, GPC_W, GPC_H, stroke=GPC_ST, bg=GPC_BG, sw=2, roundness=ROUND)
    s.text(gx + 14, gy + 10, "GPC %d" % g, size=15)
    bar(gx + 14, gy + 34, GPC_W - 28, 24, "Raster Engine  -  16 ROPs", BG_VIOLET, VIOLET, 11)

    for r in range(3):
        for c in range(2):
            tpc = g * 6 + r * 2 + c
            tx = gx + 14 + c * 176
            ty = gy + 68 + r * 116
            s.sq(tx, ty, 168, 108, stroke=TPC_ST, bg=TPC_BG, sw=1, roundness=ROUND)
            s.text(tx + 84, ty + 4, "TPC %d" % tpc, size=9, color=GRAY_D, anchor="center")
            for i in range(2):
                sm = tpc * 2 + i
                sx = tx + 6 + i * 82
                sy = ty + 20
                s.sq(sx, sy, 76, 76, stroke=SM_ST, bg=SM_BG, sw=1, roundness=ROUND)
                s.text(sx + 38, sy + 6, "SM %d" % sm, size=12, anchor="center")
                for j, t in enumerate(("128 CUDA", "4 Tensor", "1 RT Core")):
                    s.text(sx + 38, sy + 28 + j * 14, t, size=8,
                           color=GRAY_D, anchor="center")

bar(32, 600, 1528, 48, "64 MB (65,536 KB) Unified L2 Cache  —  shared by all 7 GPCs",
    BG_YELLOW, ORANGE, 22, color=ORANGE_D, sw=2)

# media / display block sits where an eighth GPC would
MX, MY, MW, MH = 1188, 665, 372, 410
s.sq(MX, MY, MW, MH, stroke=GREEN, bg="#ebfbee", sw=2, roundness=ROUND)
s.text(MX + 14, MY + 10, "Media, Display & Management", size=15, color=GREEN_D)
for i, t in enumerate(["AI Management Processor (AMP)",
                       "NVENC × 2  (9th Generation)",
                       "NVDEC × 2  (6th Generation)",
                       "Display Engine - DisplayPort 2.1b (UHBR20)",
                       "Optical Flow Accelerator / Copy Engines"]):
    bar(MX + 14, MY + 40 + i * 74, MW - 28, 60, t, BG_GREEN, GREEN, 12)

for i in range(8):
    x = 35 + i * 191
    r = s.sq(x, 1085, 176, 46, stroke=ORANGE_D, bg=BG_ORANGE, sw=1, roundness=ROUND)
    s.text(x + 88, 1093, "32-bit GDDR7", size=13, color=ORANGE_D, anchor="center")
    s.text(x + 88, 1112, "Memory Controller - 2 GB", size=9, color=ORANGE_D, anchor="center")

s.save(os.path.join(OUT, "rtx-5080-die.excalidraw"))
