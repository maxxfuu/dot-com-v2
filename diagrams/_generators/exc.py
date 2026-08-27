"""Tiny builder for .excalidraw scene files.

Coordinates are plain scene units (1 unit = 1 px at zoom 1).
Text elements carry private _anchor* hints that normalize.js resolves
into real x/y/width/height once it has measured the glyphs.
"""
import json, os, zlib

# ---------------------------------------------------------------- palette
BLACK   = "#1e1e1e"
GRAY    = "#868e96"
GRAY_D  = "#495057"
RED     = "#e03131"
RED_D   = "#c92a2a"
GREEN   = "#2f9e44"
GREEN_D = "#2b8a3e"
BLUE    = "#1971c2"
BLUE_D  = "#1864ab"
ORANGE  = "#f08c00"
ORANGE_D= "#e8590c"
VIOLET  = "#6741d9"
TEAL    = "#0c8599"
CYAN    = "#1098ad"

BG_RED    = "#ffc9c9"
BG_GREEN  = "#b2f2bb"
BG_BLUE   = "#a5d8ff"
BG_YELLOW = "#ffec99"
BG_VIOLET = "#d0bfff"
BG_ORANGE = "#ffd8a8"
BG_CYAN   = "#99e9f2"
BG_GRAY   = "#e9ecef"
NONE      = "transparent"

# fontFamily ids
HAND = 5   # Excalifont
CODE = 3   # Cascadia
NUNI = 6   # Nunito
VIRG = 1   # Virgil

ROUND = {"type": 3}   # adaptive corner radius
SHARP = None
LROUND = {"type": 2}  # for linear elements


class Scene:
    def __init__(self, name):
        self.name = name
        self.els = []
        self._n = 0
        self._group_stack = []

    # -- ids / seeds are derived from a counter so reruns are byte-stable
    def _id(self, tag=""):
        self._n += 1
        return "%s-%04d" % (self.name[:12].replace(" ", ""), self._n)

    def _seed(self):
        return zlib.crc32(("%s|%d" % (self.name, self._n)).encode()) % 2_000_000_000 + 1

    # -- grouping ------------------------------------------------------
    class _G:
        def __init__(self, sc, gid):
            self.sc, self.gid = sc, gid
        def __enter__(self):
            self.sc._group_stack.append(self.gid); return self.gid
        def __exit__(self, *a):
            self.sc._group_stack.pop()

    def group(self, gid=None):
        self._n += 1
        return Scene._G(self, gid or "g-%s-%03d" % (self.name[:8], self._n))

    # -- core ----------------------------------------------------------
    def _base(self, typ, x, y, w, h, **kw):
        e = {
            "id": kw.pop("id", None) or self._id(),
            "type": typ,
            "x": round(float(x), 2), "y": round(float(y), 2),
            "width": round(float(w), 2), "height": round(float(h), 2),
            "angle": kw.pop("angle", 0),
            "strokeColor": kw.pop("stroke", BLACK),
            "backgroundColor": kw.pop("bg", NONE),
            "fillStyle": kw.pop("fill", "solid"),
            "strokeWidth": kw.pop("sw", 2),
            "strokeStyle": kw.pop("ss", "solid"),
            "roughness": kw.pop("rough", 1),
            "opacity": kw.pop("opacity", 100),
            "groupIds": list(self._group_stack) + kw.pop("groupIds", []),
            "frameId": None,
            "roundness": kw.pop("roundness", SHARP),
            "seed": self._seed(),
            "version": 1,
            "versionNonce": self._seed(),
            "isDeleted": False,
            "boundElements": kw.pop("boundElements", None),
            "updated": 1,
            "link": None,
            "locked": False,
        }
        e.update(kw)
        self.els.append(e)
        return e

    # -- shapes --------------------------------------------------------
    def rect(self, x, y, w, h, **kw):
        kw.setdefault("roundness", ROUND)
        return self._base("rectangle", x, y, w, h, **kw)

    def sq(self, x, y, w, h, **kw):
        """sharp-cornered rectangle"""
        kw.setdefault("roundness", SHARP)
        return self._base("rectangle", x, y, w, h, **kw)

    def ellipse(self, x, y, w, h, **kw):
        return self._base("ellipse", x, y, w, h, **kw)

    def diamond(self, x, y, w, h, **kw):
        return self._base("diamond", x, y, w, h, **kw)

    def _linear(self, typ, pts, **kw):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, y0 = xs[0], ys[0]
        rel = [[round(px - x0, 2), round(py - y0, 2)] for px, py in pts]
        kw.setdefault("roundness", LROUND)
        e = self._base(typ, x0, y0, max(xs) - min(xs), max(ys) - min(ys), **kw)
        e["points"] = rel
        e["lastCommittedPoint"] = None
        e["startBinding"] = None
        e["endBinding"] = None
        e["startArrowhead"] = kw.get("startArrowhead", None)
        e["endArrowhead"] = kw.get("endArrowhead", "arrow" if typ == "arrow" else None)
        if typ == "arrow":
            e["elbowed"] = False
        return e

    def arrow(self, pts, **kw):
        head = kw.pop("head", "arrow")
        start = kw.pop("start", None)
        e = self._linear("arrow", pts, **kw)
        e["endArrowhead"] = head
        e["startArrowhead"] = start
        return e

    def line(self, pts, **kw):
        kw.pop("head", None)
        e = self._linear("line", pts, **kw)
        e["endArrowhead"] = None
        e["startArrowhead"] = None
        return e

    def freedraw(self, pts, **kw):
        kw.setdefault("roundness", SHARP)
        e = self._linear("freedraw", pts, **kw)
        e.pop("startArrowhead", None); e.pop("endArrowhead", None)
        e["pressures"] = []
        e["simulatePressure"] = True
        return e

    # -- text ----------------------------------------------------------
    def text(self, x, y, s, size=16, color=BLACK, font=HAND,
             anchor="left", valign="top", align=None, lh=1.25, **kw):
        """anchor: how (x,y) relates to the measured box - left|center|right
           valign: top|middle|bottom"""
        e = self._base("text", x, y, 10, size * lh, stroke=color,
                       roundness=SHARP, **kw)
        e.update({
            "text": s, "originalText": s,
            "fontSize": size, "fontFamily": font,
            "textAlign": align or ("center" if anchor == "center" else "left"),
            "verticalAlign": "top",
            "containerId": None,
            "autoResize": True,
            "lineHeight": lh,
            "_anchorX": round(float(x), 2), "_anchorY": round(float(y), 2),
            "_anchor": anchor, "_valign": valign,
        })
        return e

    def label(self, container, s, size=16, color=BLACK, font=HAND, lh=1.25, dy=0):
        """text bound to (and centred in) a container element"""
        e = self._base("text", container["x"], container["y"], 10, size * lh,
                       stroke=color, roundness=SHARP,
                       groupIds=list(container["groupIds"]))
        e.update({
            "text": s, "originalText": s,
            "fontSize": size, "fontFamily": font,
            "textAlign": "center", "verticalAlign": "middle",
            "containerId": container["id"],
            "autoResize": True, "lineHeight": lh,
            "_boundDy": dy,
        })
        be = container.get("boundElements") or []
        be.append({"id": e["id"], "type": "text"})
        container["boundElements"] = be
        return e

    # -- convenience ---------------------------------------------------
    def box(self, x, y, w, h, s=None, size=16, color=BLACK, font=HAND, **kw):
        r = self.rect(x, y, w, h, **kw)
        if s:
            self.label(r, s, size=size, color=color, font=font)
        return r

    def sbox(self, x, y, w, h, s=None, size=16, color=BLACK, font=HAND, **kw):
        r = self.sq(x, y, w, h, **kw)
        if s:
            self.label(r, s, size=size, color=color, font=font)
        return r

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        doc = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self.els,
            "appState": {"gridSize": 20, "gridStep": 5, "gridModeEnabled": False,
                         "viewBackgroundColor": "#ffffff"},
            "files": {},
        }
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        print("wrote %s  (%d elements)" % (path, len(self.els)))
        return path


# ---------------------------------------------------------------- helpers
GRID_LINE = "#ced4da"


def grid_box(s, x, y, w, h, nx, ny, stroke=BLACK, sw=2, line=GRID_LINE,
             ls="dotted", lsw=1, bg=NONE, fill="solid"):
    """A sharp-cornered matrix outline with faint internal gridlines."""
    for i in range(1, nx):
        gx = x + w * i / nx
        s.line([(gx, y), (gx, y + h)], stroke=line, ss=ls, sw=lsw, roundness=SHARP)
    for j in range(1, ny):
        gy = y + h * j / ny
        s.line([(x, gy), (x + w, gy)], stroke=line, ss=ls, sw=lsw, roundness=SHARP)
    return s.sq(x, y, w, h, stroke=stroke, sw=sw, bg=bg, fill=fill)


def lerp_hex(a, b, t):
    a = a.lstrip("#"); b = b.lstrip("#")
    return "#" + "".join("%02x" % round(int(a[i:i+2], 16) + (int(b[i:i+2], 16) - int(a[i:i+2], 16)) * t)
                         for i in (0, 2, 4))


def ramp(a, b, n):
    return [lerp_hex(a, b, i / max(n - 1, 1)) for i in range(n)]
