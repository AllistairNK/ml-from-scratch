"""Generate the array-operation diagrams used in the lessons (SVG, light/dark aware).

    python 00_foundations/diagrams/make_diagrams.py

Each diagram shows input arrays, the operation, and the output, with colours tracking where
each value goes. Re-run after editing; the .svg files are committed so the lessons work on GitHub too.
"""
import os
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
C = 42  # cell size in px

BLUE, ORANGE, GREEN, PURPLE, RED, GRAY = "#4c6ef5", "#f08c00", "#2f9e44", "#ae3ec9", "#e03131", "#868e96"

STYLE = """
.bg { fill: #ffffff; stroke: #e2e0da; }
text { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; fill: #1f1f1d; white-space: pre; }
.mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.muted { fill: #6b6a65; }
.title { font-weight: 600; }
.cellline { stroke: #9a9993; }
.arrow { stroke: #77766f; stroke-width: 1.6; fill: none; }
.ah { fill: #77766f; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1f1f1e; stroke: #353532; }
  text { fill: #ecebe6; }
  .muted { fill: #a3a29a; }
  .cellline { stroke: #6d6c66; }
  .arrow { stroke: #a3a29a; }
  .ah { fill: #a3a29a; }
}
"""


def fmt(v):
    if isinstance(v, float):
        return ("%g" % v).replace("-", "−")
    return str(v).replace("-", "−") if isinstance(v, int) else str(v)


class Fig:
    def __init__(self):
        self.parts = []
        self.max_x = 0
        self.max_y = 0

    def _ext(self, x, y):
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    def text(self, x, y, s, size=13, anchor="middle", cls="", weight=None, color=None):
        style = ' style="fill:{}"'.format(color) if color else ""
        w = ' font-weight="{}"'.format(weight) if weight else ""
        self.parts.append('<text x="{:.1f}" y="{:.1f}" font-size="{}" text-anchor="{}" class="{}"{}{}>{}</text>'.format(
            x, y, size, anchor, cls, w, style, escape(s)))
        width = len(s) * size * (0.62 if "mono" in cls else 0.56)
        right = x + width if anchor == "start" else x + width / 2 if anchor == "middle" else x
        self._ext(right, y + 4)

    def cell(self, x, y, val, color=None, ghost=False, dim=False, w=C, h=C, size=15):
        opacity = 0.3 if dim else 1
        if color:
            fill_op = 0.10 if ghost else 0.24
            stroke = color
        else:
            fill_op, stroke = 0, None
        dash = ' stroke-dasharray="4 3"' if ghost else ""
        stroke_attr = 'stroke="{}"'.format(stroke) if stroke else 'class="cellline"'
        self.parts.append('<rect x="{}" y="{}" width="{}" height="{}" rx="3" fill="{}" fill-opacity="{}" {} stroke-width="1.3"{} opacity="{}"/>'.format(
            x + 1, y + 1, w - 2, h - 2, color or "none", fill_op, stroke_attr, dash, opacity))
        if val is not None and val != "":
            self.parts.append('<text x="{:.1f}" y="{:.1f}" font-size="{}" text-anchor="middle" class="mono" opacity="{}"{}>{}</text>'.format(
                x + w / 2, y + h / 2 + size * 0.35, size, 0.55 if ghost else opacity,
                ' font-style="italic"' if ghost else "", escape(fmt(val))))
        self._ext(x + w, y + h)

    def grid(self, x, y, rows, color=None, ghost=None, dim=None, rlabels=None, clabels=None, w=C, gap_after_col=None, size=15):
        """rows: list of lists. color/ghost/dim: functions (r, c) -> value. Returns the bounding box."""
        gx = lambda c: x + c * w + (12 if gap_after_col is not None and c > gap_after_col else 0)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self.cell(gx(c), y + r * C, val, color(r, c) if color else None, ghost(r, c) if ghost else False,
                          dim(r, c) if dim else False, w=w, size=size)
        if rlabels:
            for r, lab in enumerate(rlabels):
                lab, col = (lab if isinstance(lab, tuple) else (lab, None))
                self.text(x - 8, y + r * C + C / 2 + 4, lab, 12, "end", "muted" if not col else "", color=col)
        if clabels:
            for c, lab in enumerate(clabels):
                self.text(gx(c) + w / 2, y - 7, lab, 12, "middle", "muted")
        right = gx(len(rows[0]) - 1) + w
        return x, y, right, y + len(rows) * C

    def arrow(self, x1, y1, x2, y2, label=None, label_dx=0, label_dy=0, anchor="middle"):
        self.parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" class="arrow" marker-end="url(#ah)"/>'.format(x1, y1, x2, y2))
        self._ext(max(x1, x2), max(y1, y2))
        if label:
            self.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy, label, 12, anchor, "mono muted")

    def symbol(self, x, y, s):
        self.text(x, y + 7, s, 24, "middle", "muted")

    def save(self, name, alt):
        w, h = self.max_x + 20, self.max_y + 18
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="{alt}">\n'
               '<title>{alt}</title>\n<style>{style}</style>\n'
               '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               '<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker></defs>\n'
               '<rect class="bg" x="0.5" y="0.5" width="{bw:.0f}" height="{bh:.0f}" rx="8"/>\n{body}\n</svg>\n').format(
            w=w, h=h, bw=w - 1, bh=h - 1, alt=escape(alt), style=STYLE, body="\n".join(self.parts))
        with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(svg)
        print("wrote", name)


X23 = [[1, 2, 3], [4, 5, 6]]
COLS3 = [BLUE, ORANGE, GREEN]


def shape_axes():
    f = Fig()
    f.text(20, 26, "X = np.array([[1, 2, 3], [4, 5, 6]])", 14, "start", "mono title")
    x0, y0 = 90, 78
    _, _, x1, y1 = f.grid(x0, y0, X23, rlabels=["row 0", "row 1"], clabels=["col 0", "col 1", "col 2"])
    f.arrow(x1 + 26, y0, x1 + 26, y1, None)
    f.text(x1 + 38, (y0 + y1) / 2 - 4, "axis 0", 13, "start", "title")
    f.text(x1 + 38, (y0 + y1) / 2 + 13, "goes down the rows", 12, "start", "muted")
    f.arrow(x0, y1 + 22, x1, y1 + 22)
    f.text((x0 + x1) / 2, y1 + 42, "axis 1 goes across the columns", 12, "middle", "muted")
    f.text(20, y1 + 74, "X.shape = (2, 3):  2 rows (axis 0) and 3 columns (axis 1)", 13, "start")
    f.save("shape_axes.svg", "A 2 by 3 array: axis 0 runs down the rows, axis 1 runs across the columns")


def indexing():
    f = Fig()
    panels = [
        ("X[1]", "row 1", lambda r, c: BLUE if r == 1 else None, lambda r, c: r != 1, [[4, 5, 6]], BLUE, "result shape (3,)"),
        ("X[:, 0]", "every row, column 0", lambda r, c: ORANGE if c == 0 else None, lambda r, c: c != 0, [[1, 4]], ORANGE, "shape (2,): comes out flat"),
        ("X[0, 2]", "row 0, column 2", lambda r, c: GREEN if (r, c) == (0, 2) else None, lambda r, c: (r, c) != (0, 2), [[3]], GREEN, "a single number"),
    ]
    for i, (code, sub, col, dim, res, rc, cap) in enumerate(panels):
        px = 30 + i * 220
        f.text(px, 26, code, 15, "start", "mono title")
        f.text(px, 44, sub, 12, "start", "muted")
        _, _, x1, y1 = f.grid(px, 62, X23, color=col, dim=dim)
        mid = (px + x1) / 2
        f.arrow(mid, y1 + 6, mid, y1 + 36)
        rx = mid - len(res[0]) * C / 2
        f.grid(rx, y1 + 42, res, color=lambda r, c, rc=rc: rc)
        f.text(mid, y1 + 42 + C + 18, cap, 12, "middle", "muted")
    f.save("indexing.svg", "Indexing: X[1] picks a row, X[:, 0] picks a column, X[0, 2] picks one element")


def slicing():
    f = Fig()
    f.text(20, 26, "y[1:4]   start = 1, stop = 4 (the stop is NOT included)", 14, "start", "mono title")
    vals = [[10, 20, 30, 40, 50]]
    x0, y0 = 30, 70
    f.grid(x0, y0, vals, color=lambda r, c: BLUE if 1 <= c < 4 else None, dim=lambda r, c: not 1 <= c < 4,
           clabels=["index 0", "1", "2", "3", "4"])
    f.text(x0 + 1.5 * C, y0 + C + 18, "↑ start", 12, "middle", color=BLUE)
    f.text(x0 + 4.5 * C, y0 + C + 18, "↑ stop (excluded)", 12, "middle", color=RED)
    f.arrow(x0 + 2.5 * C, y0 + C + 28, x0 + 2.5 * C, y0 + C + 58)
    f.grid(x0 + C, y0 + C + 64, [[20, 30, 40]], color=lambda r, c: BLUE)
    f.text(x0 + 4 * C + 12, y0 + C + 64 + C / 2 + 4, "3 items: indexes 1, 2, 3", 12, "start", "muted")
    f.save("slicing.svg", "Slicing y[1:4] keeps the items at index 1, 2 and 3")


def sum_axis():
    f = Fig()
    # Panel A: axis=0
    px, py = 30, 62
    f.text(px, 26, "X.sum(axis=0)", 15, "start", "mono title")
    f.text(px, 44, "squash DOWN each column", 12, "start", "muted")
    _, _, x1, y1 = f.grid(px, py, X23, color=lambda r, c: COLS3[c])
    for c in range(3):
        cx = px + c * C + C / 2
        f.arrow(cx, y1 + 4, cx, y1 + 32)
    f.grid(px, y1 + 38, [[5, 7, 9]], color=lambda r, c: COLS3[c])
    f.text(px, y1 + 38 + C + 20, "(2, 3) → (3,): axis 0 disappears", 12, "start", "muted")
    f.text(px, y1 + 38 + C + 36, "one total per column (per feature)", 12, "start", "muted")
    # Panel B: axis=1
    px = 300
    rows2 = [BLUE, ORANGE]
    f.text(px, 26, "X.sum(axis=1)", 15, "start", "mono title")
    f.text(px, 44, "squash ACROSS each row", 12, "start", "muted")
    _, _, x1, y1 = f.grid(px, py, X23, color=lambda r, c: rows2[r])
    for r, total in enumerate([6, 15]):
        f.arrow(x1 + 4, py + r * C + C / 2, x1 + 30, py + r * C + C / 2)
        f.text(x1 + 36, py + r * C + C / 2 + 5, fmt(total), 15, "start", "mono", color=rows2[r])
    f.text(px, y1 + 26, "collected into a flat array:", 12, "start", "muted")
    f.grid(px, y1 + 38, [[6, 15]], color=lambda r, c: rows2[c])
    f.text(px, y1 + 38 + C + 20, "(2, 3) → (2,): axis 1 disappears", 12, "start", "muted")
    f.text(px, y1 + 38 + C + 36, "one total per row (per sample)", 12, "start", "muted")
    # Panel C: keepdims
    px = 560
    f.text(px, 26, "X.sum(axis=1, keepdims=True)", 15, "start", "mono title")
    f.text(px, 44, "same totals, kept as a column", 12, "start", "muted")
    _, _, x1, y1 = f.grid(px, py, X23, color=lambda r, c: rows2[r])
    for r in range(2):
        f.arrow(x1 + 4, py + r * C + C / 2, x1 + 30, py + r * C + C / 2)
    f.grid(x1 + 36, py, [[6], [15]], color=lambda r, c: rows2[r])
    f.text(px, y1 + 38 + C + 20, "(2, 3) → (2, 1): still 2-D", 12, "start", "muted")
    f.text(px, y1 + 38 + C + 36, "so X / totals divides each row by its own total", 12, "start", "muted")
    f.save("sum_axis.svg", "Summing along axis 0 gives one total per column; along axis 1 one total per row; keepdims keeps a column shape")


def broadcasting():
    f = Fig()
    # Row case
    y0 = 62
    f.text(20, 26, "X + np.array([10, 20, 30])", 15, "start", "mono title")
    f.text(20, 44, "(2, 3) + (3,): the row is copied down to every row", 12, "start", "muted")
    _, _, x1, _ = f.grid(30, y0, X23)
    f.symbol(x1 + 22, y0 + C, "+")
    _, _, x2, _ = f.grid(x1 + 44, y0, [[10, 20, 30], [10, 20, 30]], color=lambda r, c: BLUE, ghost=lambda r, c: r == 1)
    f.text(x1 + 44 + 1.5 * C, y0 + 2 * C + 16, "dashed = virtual copy", 11, "middle", "muted")
    f.symbol(x2 + 22, y0 + C, "=")
    f.grid(x2 + 44, y0, [[11, 22, 33], [14, 25, 36]], color=lambda r, c: BLUE)
    # Column case
    y0 = 236
    f.text(20, y0 - 36, "X + np.array([[100], [200]])", 15, "start", "mono title")
    f.text(20, y0 - 18, "(2, 3) + (2, 1): the column is copied across to every column", 12, "start", "muted")
    _, _, x1, _ = f.grid(30, y0, X23)
    f.symbol(x1 + 22, y0 + C, "+")
    _, _, x2, _ = f.grid(x1 + 44, y0, [[100] * 3, [200] * 3], color=lambda r, c: [GREEN, PURPLE][r], ghost=lambda r, c: c > 0)
    f.symbol(x2 + 22, y0 + C, "=")
    f.grid(x2 + 44, y0, [[101, 102, 103], [204, 205, 206]], color=lambda r, c: [GREEN, PURPLE][r])
    f.text(20, y0 + 2 * C + 34, "Rule: line the shapes up from the right; each pair of sizes must be equal, or one of them must be 1 (the 1 gets copied).",
           12, "start", "muted")
    f.save("broadcasting.svg", "Broadcasting copies a row down, or a column across, to match the bigger array")


def newaxis():
    f = Fig()
    f.text(20, 26, "Adding an axis with None", 15, "start", "title")
    y0 = 120
    f.text(30, y0 - 12, "a", 15, "start", "mono title")
    f.grid(30, y0, [[1, 2, 3]], color=lambda r, c: COLS3[c])
    f.text(30, y0 + C + 18, "shape (3,): 1-D,", 12, "start", "muted")
    f.text(30, y0 + C + 34, "no rows or columns", 12, "start", "muted")
    bx = 30 + 3 * C + 90
    f.arrow(30 + 3 * C + 10, y0 + C / 2, bx - 10, y0 - 40 + C / 2)
    f.arrow(30 + 3 * C + 10, y0 + C / 2, bx - 10, y0 + 60 + C / 2)
    f.text(bx, y0 - 52, "a[None, :]   shape (1, 3): one ROW", 13, "start", "mono title")
    f.grid(bx, y0 - 40, [[1, 2, 3]], color=lambda r, c: COLS3[c])
    f.text(bx, y0 + 50, "a[:, None]   shape (3, 1): one COLUMN", 13, "start", "mono title")
    f.grid(bx, y0 + 60, [[1], [2], [3]], color=lambda r, c: COLS3[r])
    f.text(bx + C + 14, y0 + 60 + 1.5 * C + 4, "None means: put a size-1 axis here", 12, "start", "muted")
    f.save("newaxis.svg", "a[None, :] turns a 1-D array into a row; a[:, None] turns it into a column")


def pairwise():
    f = Fig()
    f.text(20, 26, "pts[:, None] - cs[None, :]      pts = [1, 5],  cs = [0, 4, 10]", 14, "start", "mono title")
    f.text(20, 44, "every point minus every centre, without a loop", 12, "start", "muted")
    y0 = 92
    _, _, x1, y1 = f.grid(40, y0, [[1, 1, 1], [5, 5, 5]], color=lambda r, c: [BLUE, ORANGE][r], ghost=lambda r, c: c > 0)
    f.text(40, y1 + 18, "pts[:, None]: (2, 1)", 12, "start", "mono muted")
    f.text(40, y1 + 34, "copied across → (2, 3)", 12, "start", "muted")
    f.symbol(x1 + 22, y0 + C, "−")
    _, _, x2, _ = f.grid(x1 + 44, y0, [[0, 4, 10], [0, 4, 10]], color=lambda r, c: GREEN, ghost=lambda r, c: r > 0)
    f.text(x1 + 44, y1 + 18, "cs[None, :]: (1, 3)", 12, "start", "mono muted")
    f.text(x1 + 44, y1 + 34, "copied down → (2, 3)", 12, "start", "muted")
    f.symbol(x2 + 22, y0 + C, "=")
    f.grid(x2 + 100, y0, [[1, -3, -9], [5, 1, -5]], color=lambda r, c: [BLUE, ORANGE][r],
           rlabels=[("point 1", BLUE), ("point 5", ORANGE)], clabels=["c=0", "c=4", "c=10"])
    f.text(x2 + 100, y1 + 18, "result (2, 3)", 12, "start", "mono muted")
    f.text(x2 + 100, y1 + 34, "row = a point, column = a centre", 12, "start", "muted")
    f.save("pairwise.svg", "Broadcasting a column against a row gives a table of every point minus every centre")


def mask():
    f = Fig()
    f.text(20, 26, "Boolean mask:  X[y == 0]", 15, "start", "mono title")
    y0 = 56
    f.grid(60, y0, [[0, 1, 0]], color=lambda r, c: GREEN if c != 1 else None, clabels=["", "", ""])
    f.text(52, y0 + C / 2 + 4, "y", 14, "end", "mono title")
    f.arrow(60 + 3 * C + 8, y0 + C / 2, 60 + 3 * C + 76, y0 + C / 2, "y == 0", 0, -8)
    f.grid(60 + 3 * C + 84, y0, [["True", "False", "True"]], color=lambda r, c: GREEN if c != 1 else GRAY, w=58, size=13)
    f.text(60 + 3 * C + 84 + 3 * 58 + 10, y0 + C / 2 + 4, "the mask: one True/False per sample", 12, "start", "muted")
    y1 = y0 + C + 52
    f.text(60, y1 - 14, "X", 14, "start", "mono title")
    keep = lambda r, c: r != 1
    _, _, x1, yb = f.grid(110, y1, [[150, 9], [170, 3], [140, 8]], color=lambda r, c: GREEN if keep(r, c) else None,
                          dim=lambda r, c: not keep(r, c), rlabels=[("True", GREEN), ("False", GRAY), ("True", GREEN)])
    f.arrow(x1 + 10, (y1 + yb) / 2, x1 + 90, (y1 + yb) / 2, "X[mask]", 0, -8)
    f.grid(x1 + 100, y1 + C / 2, [[150, 9], [140, 8]], color=lambda r, c: GREEN)
    f.text(x1 + 100, y1 + C / 2 + 2 * C + 18, "only the rows where the mask is True", 12, "start", "muted")
    f.text(x1 + 100, y1 + C / 2 + 2 * C + 34, "(the apples), shape (2, 2)", 12, "start", "muted")
    f.save("mask.svg", "A boolean mask keeps only the rows where it is True")


def fancy():
    f = Fig()
    f.text(20, 26, "X[[2, 0]]   pick rows by their index, in the order given", 14, "start", "mono title")
    cols = [BLUE, ORANGE, GREEN]
    y0 = 62
    _, _, x1, yb = f.grid(90, y0, [[1, 2], [3, 4], [5, 6]], color=lambda r, c: cols[r], rlabels=["row 0", "row 1", "row 2"])
    f.arrow(x1 + 12, (y0 + yb) / 2, x1 + 96, (y0 + yb) / 2, "[2, 0]", 0, -8)
    f.grid(x1 + 160, y0 + C / 2, [[5, 6], [1, 2]], color=lambda r, c: [GREEN, BLUE][r], rlabels=["X[2]", "X[0]"])
    f.text(x1 + 160, y0 + C / 2 + 2 * C + 18, "row 1 is skipped; order follows the list", 12, "start", "muted")
    f.save("fancy_indexing.svg", "Indexing with a list of row numbers picks those rows in that order")


def pick_per_row():
    f = Fig()
    f.text(20, 26, "probs[np.arange(3), y]    with y = [0, 1, 1]", 14, "start", "mono title")
    f.text(20, 44, "pairs up row i with column y[i]: one value per row", 12, "start", "muted")
    picks = {(0, 0), (1, 1), (2, 1)}
    y0 = 82
    _, _, x1, yb = f.grid(150, y0, [[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]], color=lambda r, c: GREEN if (r, c) in picks else None,
                          dim=lambda r, c: (r, c) not in picks, rlabels=["row 0, y=0", "row 1, y=1", "row 2, y=1"],
                          clabels=["col 0", "col 1"])
    f.arrow(x1 + 12, (y0 + yb) / 2, x1 + 70, (y0 + yb) / 2)
    f.grid(x1 + 80, (y0 + yb) / 2 - C / 2, [[0.7, 0.8, 0.4]], color=lambda r, c: GREEN)
    f.text(x1 + 80, (y0 + yb) / 2 + C / 2 + 18, "each sample's probability", 12, "start", "muted")
    f.text(x1 + 80, (y0 + yb) / 2 + C / 2 + 34, "for its TRUE class (cross-entropy)", 12, "start", "muted")
    f.save("pick_per_row.svg", "probs[np.arange(n), y] picks one value per row, at the column given by y")


def argsort_topk():
    f = Fig()
    f.text(20, 26, "KNN for one test point:  argsort → take k → look up labels", 14, "start", "title")
    x0 = 150
    idx_col = {1: BLUE, 3: ORANGE}
    rows = [
        ("dists", [[5.0, 1.0, 3.0, 2.0]], lambda r, c: idx_col.get(c), ["0", "1", "2", "3"], "distance to each training point"),
        ("np.argsort(dists)", [[1, 3, 2, 0]], lambda r, c: idx_col.get([1, 3, 2, 0][c]), None, "indexes, closest first"),
        ("[:k]   (k = 2)", [[1, 3]], lambda r, c: idx_col.get([1, 3][c]), None, "the k nearest neighbours"),
        ("y_train", [[1, 0, 1, 0]], lambda r, c: idx_col.get(c), ["0", "1", "2", "3"], "labels of all training points"),
        ("y_train[[1, 3]]", [[0, 0]], lambda r, c: [BLUE, ORANGE][c], None, "neighbours' labels → vote: class 0"),
    ]
    y = 64
    for i, (code, vals, col, cl, note) in enumerate(rows):
        if cl:
            y += 14
        f.text(x0 - 14, y + C / 2 + 5, code, 13, "end", "mono title")
        f.grid(x0, y, vals, color=col, clabels=["index " + cl[0]] + cl[1:] if cl else None)
        f.text(x0 + 4 * C + 16, y + C / 2 + 4, note, 12, "start", "muted")
        if i in (0, 1, 3):
            f.arrow(x0 + C, y + C + 4, x0 + C, y + C + 22)
        y += C + 28
    f.save("argsort_topk.svg", "KNN: argsort the distances, keep the first k indexes, and look up their labels")


def transpose():
    f = Fig()
    f.text(20, 26, "X.T   rows become columns", 15, "start", "mono title")
    y0 = 70
    _, _, x1, yb = f.grid(60, y0, X23, color=lambda r, c: COLS3[c], clabels=["col 0", "col 1", "col 2"])
    f.text(60, yb + 18, "X: (2, 3)", 12, "start", "mono muted")
    f.arrow(x1 + 14, (y0 + yb) / 2, x1 + 74, (y0 + yb) / 2, ".T", 0, -8)
    _, _, _, yb2 = f.grid(x1 + 130, y0 - 20, [[1, 4], [2, 5], [3, 6]], color=lambda r, c: COLS3[r], rlabels=["row 0", "row 1", "row 2"])
    f.text(x1 + 84, yb2 + 18, "X.T: (3, 2)   X.T[j, i] == X[i, j]", 12, "start", "mono muted")
    f.save("transpose.svg", "Transposing turns each column into a row")


def dot():
    f = Fig()
    f.text(20, 26, "a @ b   (dot product)    a = [1, 2, 3],  b = [4, 0, -1]", 14, "start", "mono title")
    y0 = 60
    f.grid(80, y0, [[1, 2, 3]], color=lambda r, c: COLS3[c])
    f.text(70, y0 + C / 2 + 5, "a", 14, "end", "mono title")
    f.grid(80, y0 + C + 8, [[4, 0, -1]], color=lambda r, c: COLS3[c])
    f.text(70, y0 + C + 8 + C / 2 + 5, "b", 14, "end", "mono title")
    for c in range(3):
        f.arrow(80 + c * C + C / 2, y0 + 2 * C + 12, 80 + c * C + C / 2, y0 + 2 * C + 38)
    f.grid(80, y0 + 2 * C + 44, [[4, 0, -3]], color=lambda r, c: COLS3[c])
    f.text(70, y0 + 2 * C + 44 + C / 2 + 5, "a×b", 13, "end", "mono muted")
    yb = y0 + 3 * C + 44
    f.arrow(80 + 3 * C + 10, yb - C / 2, 80 + 3 * C + 60, yb - C / 2, "sum", 0, -8)
    f.cell(80 + 3 * C + 68, yb - C, 1, PURPLE)
    f.text(80, yb + 22, "multiply matching positions, then add up:  4 + 0 − 3 = 1", 12, "start", "muted")
    f.save("dot.svg", "Dot product: multiply matching positions, then add the results")


def matmul():
    f = Fig()
    f.text(20, 26, "X @ w    (3, 2) @ (2,) → (3,)", 15, "start", "mono title")
    f.text(20, 44, "each row of X is dotted with w: one result per sample", 12, "start", "muted")
    y0 = 84
    hl = lambda r, c: ORANGE if r == 1 else None
    _, _, x1, yb = f.grid(70, y0, [[1, 2], [3, 4], [5, 6]], color=hl, rlabels=["sample 0", "sample 1", "sample 2"],
                          clabels=["f0", "f1"])
    f.text(70, yb + 18, "X: (3, 2)", 12, "start", "mono muted")
    f.symbol(x1 + 20, y0 + 1.5 * C - 8, "@")
    _, _, x2, _ = f.grid(x1 + 40, y0 + C / 2, [[0.5], [-1.0]], color=lambda r, c: ORANGE, rlabels=None)
    f.text(x1 + 40, y0 + C / 2 + 2 * C + 18, "w: (2,)", 12, "start", "mono muted")
    f.symbol(x2 + 22, y0 + 1.5 * C - 8, "=")
    f.grid(x2 + 44, y0, [[-1.5], [-2.5], [-3.5]], color=hl)
    f.text(x2 + 44, yb + 18, "(3,)", 12, "start", "mono muted")
    f.text(x2 + 44 + C + 14, y0 + 1.5 * C + 4, "3×0.5 + 4×(−1) = −2.5", 12, "start", color=ORANGE)
    f.text(20, yb + 48, "Shape rule: (n, k) @ (k, m) → (n, m). The inner sizes must match, and they disappear.", 12, "start", "muted")
    f.save("matmul.svg", "Matrix times vector: each row of X dotted with w gives one output per row")


def split_heads():
    f = Fig()
    f.text(20, 26, "Multi-head attention: split_heads (batch axis left out)", 15, "start", "title")
    f.text(20, 44, "x: 3 tokens × d_model = 4 features, n_heads = 2, so d_head = 2", 12, "start", "muted")
    x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    head = lambda r, c: BLUE if c < 2 else ORANGE
    y0 = 92
    _, _, x1, yb = f.grid(80, y0, x, color=head, rlabels=["token 0", "token 1", "token 2"], clabels=["0", "1", "2", "3"])
    f.text(80, yb + 18, "x: (T, d_model) = (3, 4)", 12, "start", "mono muted")
    f.arrow(x1 + 12, (y0 + yb) / 2, x1 + 70, (y0 + yb) / 2, "reshape", 0, -8)
    bx = x1 + 80
    _, _, x2, _ = f.grid(bx, y0, x, color=head, gap_after_col=1, clabels=["h0", "", "h1", ""])
    f.text(bx, yb + 18, "(T, h, d_head) = (3, 2, 2)", 12, "start", "mono muted")
    f.text(bx, yb + 34, "same numbers, grouped per head", 12, "start", "muted")
    f.arrow(x2 + 12, (y0 + yb) / 2, x2 + 70, (y0 + yb) / 2, "transpose", 0, -8)
    hx = x2 + 120
    f.text(hx, y0 - 30, "head 0", 13, "start", "title", color=BLUE)
    f.grid(hx, y0 - 22, [[1, 2], [5, 6], [9, 10]], color=lambda r, c: BLUE, rlabels=["t0", "t1", "t2"])
    f.text(hx + 2 * C + 40, y0 - 30, "head 1", 13, "start", "title", color=ORANGE)
    f.grid(hx + 2 * C + 40, y0 - 22, [[3, 4], [7, 8], [11, 12]], color=lambda r, c: ORANGE)
    f.text(hx, y0 - 22 + 3 * C + 22, "(h, T, d_head) = (2, 3, 2)", 12, "start", "mono muted")
    f.text(hx, y0 - 22 + 3 * C + 38, "each head sees every token, but only its features", 12, "start", "muted")
    f.save("split_heads.svg", "split_heads reshapes the features into heads, then moves the head axis first")


if __name__ == "__main__":
    for make in (shape_axes, indexing, slicing, sum_axis, broadcasting, newaxis, pairwise, mask, fancy,
                 pick_per_row, argsort_topk, transpose, dot, matmul, split_heads):
        make()
