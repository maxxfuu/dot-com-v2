import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("blockwarps")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page2")

s.text(10, 8, "Block Splits Into Warps", size=26)
s.text(10, 44, "thread block: 1024 threads", size=13, color=GRAY)

# the block, as a dashed container
s.rect(8, 70, 340, 228, stroke=GRAY, ss="dashed", sw=2, roundness=ROUND)

for label, y in [("warp 0", 92), ("warp 1", 132), ("warp 2", 172), ("warp 31", 244)]:
    s.box(30, y, 296, 32, label, size=15, stroke=ORANGE, bg=BG_YELLOW)

# the elision between warp 2 and warp 31
for dy in (0, 9, 18):
    s.ellipse(176, 214 + dy, 4, 4, stroke=GRAY, bg=GRAY, sw=1)

s.arrow([(366, 190), (408, 190)], sw=2)

s.text(432, 158, "one warp = 32 threads/lanes", size=13, color=GRAY)
for i in range(32):
    s.rect(432 + i * 20, 182, 17, 20, stroke=ORANGE, bg=BG_YELLOW, sw=1, roundness=ROUND)

s.save(os.path.join(OUT, "block-splits-into-warps.excalidraw"))
