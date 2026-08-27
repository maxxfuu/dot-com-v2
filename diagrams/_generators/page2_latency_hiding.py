import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("latency")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page2")

s.text(10, 10, "Latency Hiding", size=26)

X0, X1, H = 115, 737, 28
TRACK_BG, TRACK_ST = "#f1f3f5", "#dee2e6"
# boundaries of the four issue slots along the track
B = [115, 238, 355, 478, 610, 737]

WARPS = [
    ("warp 0", BLUE,     BG_BLUE,   0, "stalled on a memory load"),
    ("warp 1", VIOLET,   BG_VIOLET, 1, "stalled on a memory load"),
    ("warp 2", ORANGE_D, BG_ORANGE, 2, "stalled on a memory load"),
    ("warp 3", RED,      BG_RED,    3, "stalled on a me..."),
]

for i, (name, stroke, bg, slot, note) in enumerate(WARPS):
    y = 62 + i * 41
    s.text(12, y + H / 2, name, size=15, color=stroke, valign="middle")
    s.rect(X0, y, X1 - X0, H, stroke=TRACK_ST, bg=TRACK_BG, sw=1, roundness=ROUND)
    a, b = B[slot], B[slot + 1]
    s.rect(a, y, b - a, H, stroke=stroke, bg=bg, sw=2, roundness=ROUND)
    s.text(b + 10, y + H / 2, note, size=13, color=GRAY, valign="middle")
    if i == 0:   # warp 0 comes back at the end of the window
        s.rect(B[4], y, X1 - B[4], H, stroke=stroke, bg=bg, sw=2, roundness=ROUND)

# what the SM actually issues: a solid run made of the four warps in turn
Y = 244
s.text(12, Y + H / 2, "SM issues", size=16, valign="middle")
order = [BLUE, VIOLET, ORANGE_D, RED, BLUE]
bgs = [BG_BLUE, BG_VIOLET, BG_ORANGE, BG_RED, BG_BLUE]
for i in range(5):
    s.rect(B[i], Y, B[i + 1] - B[i], H, stroke=order[i], bg=bgs[i], sw=2, roundness=ROUND)

s.arrow([(X0, 296), (X1 - 2, 296)], stroke=GRAY, sw=1)
s.text(120, 304, "time →", size=13, color=GRAY)

s.save(os.path.join(OUT, "latency-hiding.excalidraw"))
