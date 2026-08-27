# compare a rendered PNG's ink extent against the scene's own bounding box
import json, sys
from PIL import Image
scene = json.load(open(sys.argv[1]))
img = Image.open(sys.argv[2]).convert("L")
w, h = img.size
import numpy as np
a = np.array(img)
ink = a < 250
cols = np.where(ink.any(axis=0))[0]; rows = np.where(ink.any(axis=1))[0]
print(f"image {w}x{h}  ink x:[{cols.min()},{cols.max()}] y:[{rows.min()},{rows.max()}]")
print(f"right margin {w-1-cols.max()}px  bottom margin {h-1-rows.max()}px  (expect ~24 at scale 2)")
