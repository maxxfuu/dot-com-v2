import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("warpdiv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page2")

s.text(10, 10, "Divergence Serializes The Warp", size=24)

OFF_STROKE, OFF_BG = "#ced4da", NONE
PITCH, W, H, Y = 14.3, 11.5, 16, 88


def lane_row(x0, active_lo, active_hi, stroke, bg):
    for i in range(32):
        on = active_lo <= i < active_hi
        s.rect(x0 + i * PITCH, Y, W, H,
               stroke=stroke if on else OFF_STROKE,
               bg=bg if on else OFF_BG, sw=1, roundness=ROUND)


s.text(10, 60, "step 1 · the if-path runs", size=15, color=GREEN)
lane_row(10, 0, 18, GREEN, BG_GREEN)
s.text(10, 112, "14 lanes switched off", size=13, color=GRAY)

s.arrow([(480, 96), (518, 96)], sw=2)

s.text(532, 60, "step 2 · the else-path runs", size=15, color=BLUE)
lane_row(532, 18, 32, BLUE, BG_BLUE)
s.text(532, 112, "18 lanes switched off", size=13, color=GRAY)

s.text(12, 166, "Both paths are walked, one after the other. The warp pays for both given the divergence.", size=16)
s.text(12, 194, "If all 32 lanes take the same branch such that there is no divergence, then there are no cost.", size=16)

s.save(os.path.join(OUT, "warp-divergence.excalidraw"))
