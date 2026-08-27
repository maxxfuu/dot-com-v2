import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("roofsmem")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page6")

XT0, XPITCH = 148.0, 108.7          # one tick = one factor of 4
Y025, YPITCH = 462.0, 79.0          # ditto on the y axis
X0, Y0 = 78.0, 510.0
RIGHT = 790.0

def px(i): return XT0 + XPITCH * math.log(i / 0.25, 4)
def py(v): return Y025 - YPITCH * math.log(v / 0.25, 4)

RX, RY = px(58.6), py(56.3)
SLOPE = YPITCH / XPITCH            # y px per x px along the bandwidth roof
LX = RX - (Y0 - RY) / SLOPE

s.text(60, 12, "Roofline model  -  RTX 5080  ( GB203, Blackwell )", size=24)
s.text(60, 46, "attainable FLOP/s  =  min( peak compute ,  bandwidth × arithmetic intensity )",
       size=13, color=GRAY)

s.line([(LX, Y0), (RX, RY), (RX, Y0), (LX, Y0)],
       stroke=NONE, bg="#a5d8ff", fill="hachure", sw=1, opacity=30, roundness=SHARP)
s.line([(RX, RY), (RIGHT, RY), (RIGHT, Y0), (RX, Y0), (RX, RY)],
       stroke=NONE, bg="#b2f2bb", fill="hachure", sw=1, opacity=30, roundness=SHARP)

s.arrow([(X0, Y0), (X0, 86)], sw=2)
s.arrow([(X0, Y0), (836, Y0)], sw=2)

for v, lab in [(0.25, "0.25"), (1, "1"), (4, "4"), (16, "16"),
               (64, "64"), (256, "256"), (1024, "1024")]:
    x = px(v)
    s.line([(x, Y0), (x, Y0 + 7)], sw=1)
    s.text(x, Y0 + 12, lab, size=13, color=GRAY_D, anchor="center")
for v, lab in [(0.25, "0.25"), (1, "1"), (4, "4"), (16, "16"), (64, "64")]:
    y = py(v)
    s.line([(X0 - 7, y), (X0, y)], sw=1)
    s.text(X0 - 12, y, lab, size=13, color=GRAY_D, anchor="right", valign="middle")

s.text(250, 548, "arithmetic intensity   I   ( FLOP / byte )  -  log scale", size=15)
s.text(28, 300, "attainable performance  ( TFLOP/s )  -  log scale",
       size=13, color=GRAY_D, anchor="center", valign="middle", angle=-math.pi / 2)

s.line([(480, RY), (RX, RY)], stroke=GRAY, ss="dashed", sw=1)
s.line([(RX, RY), (RX, Y0)], stroke=GRAY, ss="dashed", sw=1)
s.line([(LX, Y0), (RX, RY)], stroke=BLUE, sw=2)
s.line([(RX, RY), (RIGHT, RY)], stroke=GREEN, sw=3)
s.ellipse(RX - 8, RY - 8, 16, 16, stroke=BLACK, bg=BLACK, fill="solid", sw=2)

# where the two kernels actually sit
NX, NY = px(0.25), py(0.24)
SX, SY = px(8), py(8 * 0.96)
s.ellipse(NX - 8, NY - 8, 16, 16, stroke=ORANGE_D, bg=NONE, sw=2)
s.arrow([(NX + 6, NY - 5), (SX - 4, SY + 6)], stroke="#fa5252", sw=4)
s.ellipse(SX - 9, SY - 9, 18, 18, stroke="#fa5252", bg="#fa5252", fill="solid", sw=2)

th = math.atan(SLOPE)
SN, CS = math.sin(th), math.cos(th)


def along(x, off):
    """point on the bandwidth roof at x, pushed `off` px perpendicular (+ = below)"""
    return x + off * SN, RY + (RX - x) * SLOPE + off * CS


mx, my = along(285, 21)
s.text(mx, my, "shared-memory tiling", size=13, color="#fa5252",
       anchor="center", valign="middle", angle=-th)
mx, my = along(362, -13)
s.text(mx, my, "->  × 32", size=13, color="#fa5252",
       anchor="center", valign="middle", angle=-th)

s.text(148, 478, "naive   I = 0.25   ( 0.24 TFLOP/s )", size=12, color=GRAY)
s.text(438, 276, "still 7.3 ×  short", size=12, color=GRAY)
s.arrow([(500, 250), (RX - 20, RY + 24)], stroke=GRAY, sw=2)

s.text(600, 92, "peak FP32 = 56.3 TFLOP/s", size=14, color=GREEN)
s.text(600, 114, "( 10,752 cores × 2 FLOP × 2.62 GHz )", size=12, color=GRAY)
s.text(600, 162, "ridge point", size=14)
s.text(600, 184, "I* = P_peak / BW = 58.6 FLOP/byte", size=12, color=GRAY_D)

mx, my = along(248, -33)
s.text(mx, my, "slope = 960 GB/s  (GDDR7, 256-bit @ 30 Gbps)",
       size=12, color=BLUE, anchor="center", valign="middle", angle=-th)

s.text(378, 388, "MEMORY BOUND", size=17, color=BLUE)
s.text(378, 412, "DRAM is the limit,", size=12, color=GRAY_D)
s.text(378, 430, "extra FLOPs are free", size=12, color=GRAY_D)
s.text(612, 388, "COMPUTE BOUND", size=17, color=GREEN)
s.text(612, 412, "the SMs are the limit,", size=12, color=GRAY_D)
s.text(612, 430, "only more FLOP/s helps", size=12, color=GRAY_D)
s.arrow([(766, 466), (766, 212)], stroke=GREEN, sw=2)

s.save(os.path.join(OUT, "roofline-smem-tiling.excalidraw"))
