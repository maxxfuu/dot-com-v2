import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exc import *

s = Scene("ineffwarp")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "page3")

s.text(390, 10, "Inefficient Warp Memory Access Pattern", size=17, anchor="center")

# ------------------------------------------------ left: one warp, 32 lanes
s.text(18, 158, "One Warp of the Block = 32 Lanes", size=14)
s.text(30, 200, "threadIdx.x  ·  lane 0 → 31", size=16, color=ORANGE)
s.arrow([(30, 224), (272, 224)], stroke=ORANGE, sw=2)

LX, LY, LW, LH = 10, 258, 262, 24
grid_box(s, LX, LY, LW, LH, 32, 1, sw=2, line="#adb5bd", ls="solid")
s.sq(LX, LY, LW / 32, LH / 2, stroke=RED, bg=BG_RED, sw=1)
s.sq(LX, LY + LH / 2, LW / 32, LH / 2, stroke=GREEN, bg=BG_GREEN, sw=1)
s.text(12, 286, "0", size=11, color=RED)
s.text(252, 286, "31", size=11, color=GRAY)

# ------------------------------------------------ right: the three matrices
s.sq(330, 50, 440, 390, stroke=GRAY, ss="dashed", sw=1, roundness=ROUND)
s.text(360, 66, "A × B = C", size=16, color=GRAY, font=CODE)

s.text(610, 66, "B  (K × N)", size=11)
grid_box(s, 555, 84, 157, 140, 8, 8)
s.sq(555, 84, 14, 140, stroke=BLUE, bg=BG_BLUE, fill="hachure", sw=1)
s.sq(555, 114, 14, 28, stroke=BLUE, bg=BG_BLUE, fill="solid", sw=1)

s.text(444, 234, "A  (M × K)", size=11)
grid_box(s, 382, 252, 152, 160, 8, 8)
s.sq(416, 252, 20, 160, stroke=ORANGE, bg=BG_YELLOW, sw=1)
s.sq(382, 254, 152, 14, stroke=RED, bg=BG_RED, fill="hachure", sw=1)
s.sq(382, 276, 152, 14, stroke=GREEN, bg=BG_GREEN, fill="hachure", sw=1)

s.text(610, 234, "C  (M × N)", size=11)
grid_box(s, 555, 252, 157, 160, 8, 8)
s.sq(555, 252, 14, 160, stroke=ORANGE, bg=BG_YELLOW, sw=1)
s.sq(555, 254, 14, 14, stroke=RED, bg=BG_RED, sw=1)
s.sq(555, 276, 14, 14, stroke=GREEN, bg=BG_GREEN, sw=1)

# ------------------------------------------------ annotations
s.text(370, 92, "Uniform access across all 32 lanes\nin an instruction cycle, so the\n"
                "hardware broadcasts the values\ninto all of the registers",
       size=9, color=BLUE)
s.arrow([(524, 126), (552, 118)], stroke=BLUE, sw=2)

s.text(352, 160, "Strided access across all 32 lanes\nin an instruction cycle, so the\n"
                 "hardware fetches 32 separate\ncache lines, resulting in\n"
                 "uncoalesced memory reads",
       size=9, color=RED)
s.arrow([(424, 234), (424, 250)], stroke=RED, sw=2)

s.arrow([(30, 262), (160, 258), (300, 250), (378, 256)],
        stroke=RED, sw=2, roundness=LROUND)
s.arrow([(30, 278), (160, 300), (300, 296), (378, 282)],
        stroke=GREEN, sw=2, roundness=LROUND)

s.save(os.path.join(OUT, "inefficient-warp-access.excalidraw"))
