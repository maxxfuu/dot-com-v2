import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("matcidx")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page3")

X, Y, W = 12, 40, 434
CELL = W / 8.0

s.text(X + W / 2, 10, "Matrix C", size=20, anchor="center")
grid_box(s, X, Y, W, W, 8, 8)

for (r, c, lab, st, bg) in [(0, 0, "t[0][0]", RED, BG_RED),
                            (0, 1, "t[0][1]", ORANGE, BG_YELLOW),
                            (1, 0, "t[1][0]", GREEN, BG_GREEN)]:
    s.sq(X + c * CELL, Y + r * CELL, CELL, CELL, stroke=st, bg=bg, sw=2)
    s.text(X + c * CELL + 5, Y + r * CELL + 20, lab, size=9, color=st)

# stepping one column right, one row down, and along the diagonal
s.arrow([(X + 2.2 * CELL, Y + 0.5 * CELL), (X + 4.0 * CELL, Y + 0.5 * CELL)], sw=2)
s.arrow([(X + 0.5 * CELL, Y + 2.2 * CELL), (X + 0.5 * CELL, Y + 4.3 * CELL)], sw=2)
s.arrow([(X + 1.3 * CELL, Y + 1.15 * CELL), (X + 5.0 * CELL, Y + 5.0 * CELL)], sw=2)

s.save(os.path.join(OUT, "matrix-c-indexing.excalidraw"))
