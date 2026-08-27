"""Shared geometry for the two roofline charts (page2 and page6)."""
import math

X0, Y0 = 95.0, 520.0          # origin of the axes
XPITCH = 120.7                # one x tick = one factor of 4
YPITCH = 49.0                 # one y tick = one doubling
XREF = 0.25                   # value at x = X0 - 7 ... first tick sits at XT0
XT0 = 88.0                    # x of the 0.25 tick
YT1 = 462.0                   # y of the "1 TFLOP/s" tick

RIDGE_I, PEAK = 58.6, 56.3
SLOPE = 2 * YPITCH / XPITCH   # px of y per px of x along the bandwidth roof


def px(i):
    return XT0 + XPITCH * math.log(i / XREF, 4)


def py(v):
    return YT1 - YPITCH * math.log(v, 2)


XTICKS = [(0.25, "0.25"), (1, "1"), (4, "4"), (16, "16"),
          (64, "64"), (256, "256"), (1024, "1024")]
YTICKS = [(1, "1"), (2, "2"), (4, "4"), (8, "8"), (16, "16"), (32, "32"), (64, "64")]
