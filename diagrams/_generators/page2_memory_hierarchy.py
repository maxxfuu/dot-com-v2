import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("memhier")

s.text(190, 10, "Memory Hierarchy", size=28)

s.text(190, 64, "scope", size=13, color=GRAY)
s.text(385, 64, "level", size=13, color=GRAY)
s.text(753, 64, "rough cost", size=13, color=GRAY)

# the "bigger / slower / farther" axis
s.arrow([(128, 105), (128, 385)], stroke=GRAY, sw=1)
s.text(68, 222, "bigger\nslower\nfarther", size=14, color=GRAY,
       anchor="center", valign="middle")

ROWS = [
    ("thread", "Registers",             "~1 cycle",                GREEN,    BG_GREEN,  GREEN),
    ("block",  "Shared Memory / L1",    "tens of cycles",          ORANGE,   BG_YELLOW, ORANGE),
    ("device", "L2 Cache",              "hundreds of cycles",      ORANGE_D, BG_ORANGE, ORANGE_D),
    ("device", "Device Memory (VRAM)",  "many hundreds of cycles", RED,      BG_RED,    RED),
]

X, W, H = 378, 334, 52
for i, (scope, level, cost, stroke, bg, ctext) in enumerate(ROWS):
    cy = 133 + i * 76
    s.text(190, cy, scope, size=16, valign="middle")
    s.arrow([(288, cy), (370, cy)], stroke=GRAY, sw=1)
    s.box(X, cy - H / 2, W, H, level, size=19, stroke=stroke, bg=bg)
    s.text(753, cy, cost, size=16, color=ctext, valign="middle")

s.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "page2", "memory-hierarchy.excalidraw"))
