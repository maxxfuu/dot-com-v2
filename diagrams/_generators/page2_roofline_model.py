import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *
from roofline_common import *

s = Scene("roofline")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page2")

RX, RY = px(RIDGE_I), py(PEAK)      # the ridge point
LX = RX - (Y0 - RY) / SLOPE         # where the bandwidth roof leaves the x axis
RIGHT = 838.0

s.text(88, 12, "Roofline model  -  RTX 5080  (GB203, Blackwell)", size=26)
s.text(88, 48, "attainable FLOP/s  =  min( peak compute ,  bandwidth × arithmetic intensity )",
       size=14, color=GRAY)

# ------------------------------------------------------------ shaded regions
s.line([(LX, Y0), (RX, RY), (RX, Y0), (LX, Y0)],
       stroke=NONE, bg="#a5d8ff", fill="hachure", sw=1, opacity=35, roundness=SHARP)
s.line([(RX, RY), (RIGHT, RY), (RIGHT, Y0), (RX, Y0), (RX, RY)],
       stroke=NONE, bg="#b2f2bb", fill="hachure", sw=1, opacity=35, roundness=SHARP)

# ------------------------------------------------------------ axes
s.arrow([(X0, Y0), (X0, 88)], sw=2)
s.arrow([(X0, Y0), (862, Y0)], sw=2)

for v, lab in XTICKS:
    x = px(v)
    s.line([(x, Y0), (x, Y0 + 7)], sw=1)
    s.text(x, Y0 + 12, lab, size=13, color=GRAY_D, anchor="center")
for v, lab in YTICKS:
    y = py(v)
    s.line([(X0 - 7, y), (X0, y)], sw=1)
    s.text(X0 - 12, y, lab, size=13, color=GRAY_D, anchor="right", valign="middle")

s.text(248, 556, "arithmetic intensity   I   ( FLOP / byte )  -  log scale", size=15)
s.text(30, 306, "attainable performance  ( TFLOP/s )  -  log scale",
       size=13, color=GRAY_D, anchor="center", valign="middle", angle=-math.pi / 2)

# ------------------------------------------------------------ the roof
s.line([(X0, RY), (RX, RY)], stroke=GRAY, ss="dashed", sw=1)
s.line([(RX, RY), (RX, Y0)], stroke=GRAY, ss="dashed", sw=1)

s.line([(LX, Y0), (RX, RY)], stroke=BLUE, sw=3)
s.line([(RX, RY), (RIGHT, RY)], stroke=GREEN, sw=3)
s.ellipse(RX - 8, RY - 8, 16, 16, stroke=BLACK, bg=BLACK, fill="solid", sw=2)

# ------------------------------------------------------------ annotations
s.text(312, 138, "peak FP32 = 56.3 TFLOP/s    (10,752 cores × 2 FLOP × 2.62 GHz)",
       size=14, color=GREEN)
s.text(580, 186, "ridge point", size=14)
s.text(580, 210, "I* = P_peak / BW = 58.6 FLOP/byte", size=13, color=GRAY_D)

# nudge the label off the line, perpendicular to it
th = math.atan(SLOPE)
OFFS = 17
s.text(312 - OFFS * math.sin(th), RY + (RX - 312) * SLOPE - OFFS * math.cos(th),
       "slope = 960 GB/s  (GDDR7, 256-bit @ 30 Gbps)",
       size=13, color=BLUE, anchor="center", valign="middle", angle=-th)

s.text(362, 398, "MEMORY BOUND", size=18, color=BLUE)
s.text(362, 424, "DRAM is the limit, extra", size=13, color=GRAY_D)
s.text(362, 442, "FLOPs are free", size=13, color=GRAY_D)
s.text(240, 468, "raise I : reuse data  (SMEM / register tiling)", size=13, color=BLUE)
s.arrow([(240, 492), (505, 492)], stroke=BLUE, sw=2)

s.text(612, 398, "COMPUTE BOUND", size=18, color=GREEN)
s.text(612, 424, "the SMs are the limit, only", size=13, color=GRAY_D)
s.text(612, 442, "more FLOP/s helps", size=13, color=GRAY_D)
s.text(808, 322, "raise FLOP/s : ILP,", size=13, color=GREEN)
s.text(808, 340, "occupancy, tensor cores", size=13, color=GREEN)
s.arrow([(792, 468), (792, 264)], stroke=GREEN, sw=2)

s.save(os.path.join(OUT, "roofline-model.excalidraw"))
