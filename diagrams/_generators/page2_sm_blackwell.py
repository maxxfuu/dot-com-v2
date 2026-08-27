import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("smblack")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page2")

S = 2.0                      # scene units per pixel of the reference export
def k(v): return v * S

AMBER_BG, AMBER_ST = BG_YELLOW, ORANGE
VIO_BG, VIO_ST = BG_VIOLET, VIOLET
ORA_BG, ORA_ST = BG_ORANGE, ORANGE_D
CORE_BG, CORE_ST = "#a5d8ff", BLUE
TEN_BG, TEN_ST = BG_GREEN, GREEN


def bar(x, y, w, h, txt, bg, st, size, color=BLACK, lh=1.25):
    r = s.sq(k(x), k(y), k(w), k(h), stroke=st, bg=bg, sw=1, roundness=ROUND)
    s.label(r, txt, size=k(size), color=color, lh=lh)
    return r


# outer shell
s.sq(k(6), k(10), k(509), k(902), stroke="#4dabf7", sw=2, roundness=ROUND)
s.text(k(260), k(22), "Streaming Multiprocessor (SM) - Blackwell GB203, GeForce RTX 5080",
       size=k(11), color=VIOLET, anchor="center")

bar(20, 58, 480, 20, "Instruction Cache  (L1)", AMBER_BG, AMBER_ST, 8)

PANELS = [(24, 88), (268, 88), (24, 460), (268, 460)]
PW, PH = 228, 362

for idx, (px_, py_) in enumerate(PANELS):
    s.sq(k(px_), k(py_), k(PW), k(PH), sw=2, roundness=ROUND)
    s.text(k(px_ + PW / 2), k(py_ + 8), "Sub-Partition %d" % (idx + 1),
           size=k(10), anchor="center")

    bar(px_ + 8, py_ + 24, PW - 16, 18, "L0 Instruction Cache", AMBER_BG, AMBER_ST, 8)
    bar(px_ + 8, py_ + 50, PW - 16, 20, "Warp Scheduler   (32 thread/clk)", VIO_BG, VIO_ST, 8)
    bar(px_ + 8, py_ + 76, PW - 16, 20, "Dispatch Unit   (32 thread/clk)", VIO_BG, VIO_ST, 8)
    bar(px_ + 8, py_ + 104, PW - 16, 22, "Register File    (16,384 x 32-bit)", ORA_BG, ORA_ST, 8)

    for r in range(8):
        for c in range(4):
            bar(px_ + 10 + c * 53, py_ + 136 + r * 17.3, 49, 15,
                "FP32/INT32", CORE_BG, CORE_ST, 5)

    bar(px_ + 8, py_ + 284, PW - 16, 40,
        "5th Gen Tensor Core\nFP4 / FP8 / FP16 / BF16", TEN_BG, TEN_ST, 8)

    hw = (PW - 20) / 2
    bar(px_ + 8, py_ + 332, hw, 22, "4 x LD/ST", NONE, BLACK, 8)
    bar(px_ + 12 + hw, py_ + 332, hw, 22, "4 x SFU", NONE, BLACK, 8)

bar(20, 838, 480, 22, "128 KB  L1 Data Cache  /  Shared Memory", AMBER_BG, AMBER_ST, 8)
bar(20, 868, 232, 22, "1 x  4th Gen RT Core   (Ray/Triangle Intersect)", BG_RED, RED, 6)
bar(266, 868, 234, 22, "4 x  Texture Units (TMU)", NONE, GRAY_D, 7)

s.save(os.path.join(OUT, "sm-blackwell.excalidraw"))
