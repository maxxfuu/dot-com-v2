import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("insideblk")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page2")

HL = 3   # the highlighted thread

s.text(78, 8, "Inside One Thread Block", size=24)
s.text(78, 58, "execution over time →", size=13, color=GRAY)

WAVE_X0, WAVE_X1, HALF, AMP = 88, 480, 29.4, 9
for i in range(8):
    cy = 96 + i * 31.7
    hot = i == HL
    stroke = RED if hot else ORANGE
    bg = BG_RED if hot else BG_YELLOW
    s.text(10, cy, "t%d" % i, size=15, color=BLACK if not hot else RED, valign="middle")
    d = 2 if hot else 0
    s.rect(40 - d, cy - 11 - d, 22 + 2 * d, 22 + 2 * d, stroke=stroke, bg=bg,
           sw=2 if hot else 1, roundness=ROUND)
    # one instruction stream, drawn as a wave that ends in an arrowhead
    pts = []
    k = 0
    x = WAVE_X0
    while x < WAVE_X1:
        pts.append((x, cy + (AMP if k % 2 else -AMP)))
        x += HALF; k += 1
    pts.append((WAVE_X1 + 12, cy))
    s.arrow(pts, stroke=stroke, sw=3 if hot else 1, roundness=LROUND)

s.text(518, 190, "One Thread", size=14, color=RED, valign="middle")
s.arrow([(600, 180), (682, 146), (748, 138), (816, 166)],
        stroke="#ff8787", sw=2, roundness=LROUND)

s.text(672, 86, "all 8 threads run the same instructions;", size=15)
s.text(672, 114, "the highlighted one just carries a different index:", size=15)

for i in range(8):
    hot = i == HL
    s.box(668 + i * 51, 170, 44, 42, str(i), size=18,
          stroke=RED if hot else ORANGE, bg=BG_RED if hot else BG_YELLOW)

s.text(672, 230, "data[i]  -  each thread touches its own element", size=15, color=GRAY_D)
s.text(672, 260, "threads in the same block can also talk to each other", size=15, color=ORANGE_D)
s.text(672, 286, "through shared memory and __syncthreads() barriers.", size=15, color=ORANGE_D)

s.save(os.path.join(OUT, "inside-thread-block.excalidraw"))
