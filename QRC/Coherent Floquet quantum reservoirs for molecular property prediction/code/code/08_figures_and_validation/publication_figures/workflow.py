"""Editable two-column schematic of the two molecular input protocols and their readouts."""

from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch


# The APS writing skill's Floquet schematic palette. Color marks physical roles.
INK = "#263B4D"
EDGE = "#A7B6C4"
FIELD = "#EDF2F6"
GATE = "#D8E6EF"
RESET = "#E8E0EB"
LOOP = "#8C728E"
MEASURE = "#DDECEB"
ACCENT = "#E59E66"
TEAL = "#245E61"
PLUM = "#704F70"
BLUE = "#365B79"
SOFT_BLUE = "#F2F6F9"
WHITE = "#FFFFFF"
ENDPOINT_BRANCH = (130.75, 35.4)  # Gutter between the reservoir and the feature vector.


FONT_DIR = Path(__file__).resolve().parent / "fonts"
DIAGRAM_FONTS = {
    weight: FontProperties(fname=str(FONT_DIR / filename), weight=weight)
    for weight, filename in (("normal", "LiberationSans-Regular.ttf"),
                             ("bold", "LiberationSans-Bold.ttf"))
}


class Diagram:
    """Draw in millimeters measured from the top left of the final figure."""

    def __init__(self, width=178, height=99):
        self.fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor=WHITE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
        self.ax.set_axis_off()

    def text(self, x, y, value, *, size=7.8, color=INK, weight="normal", align="center",
             background=None):
        bbox = dict(facecolor=background, edgecolor="none", pad=0.55) if background else None
        self.ax.text(x, y, value, ha=align, va="center", fontsize=size, color=color,
                     fontproperties=DIAGRAM_FONTS[weight], math_fontfamily="stix",
                     linespacing=1.06, bbox=bbox, zorder=8)

    def box(self, x, y, width, height, *, fill=WHITE, edge=EDGE, radius=.9, lw=.72,
            layer=1):
        self.ax.add_patch(FancyBboxPatch((x, y), width, height,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=fill, edgecolor=edge, linewidth=lw, zorder=layer))

    def line(self, points, *, color=INK, lw=.85, layer=3):
        xx, yy = zip(*points)
        self.ax.plot(xx, yy, color=color, linewidth=lw, solid_joinstyle="round",
                     solid_capstyle="round", zorder=layer)

    def arrow(self, points, *, color=INK, lw=.88, head=6.1):
        if len(points) > 2:
            self.line(points[:-1], color=color, lw=lw, layer=4)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                          mutation_scale=head, linewidth=lw, shrinkA=0, shrinkB=0,
                          color=color, zorder=4))

    def dot(self, x, y, radius, *, fill=WHITE, edge=INK, lw=.7, layer=5):
        self.ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=edge,
                                 linewidth=lw, zorder=layer))

    def meter(self, x, y, width):
        """Semicircular meter with its needle pointing toward positive readout."""
        self.ax.add_patch(Arc((x, y), width, width * .68, theta1=180, theta2=360,
                             color=TEAL, linewidth=.92, zorder=6))
        self.ax.add_patch(FancyArrowPatch((x, y), (x + width * .31, y - width * .42),
                          arrowstyle="-|>", mutation_scale=4.3, linewidth=.78,
                          shrinkA=0, shrinkB=0, color=TEAL, zorder=7))


def molecular_inputs(d):
    d.box(.7, 1.5, 25.8, 28.3, fill=SOFT_BLUE, edge=BLUE)
    d.text(2.3, 5.6, r"Graph  $Q=12$", size=8.6, weight="bold", align="left")
    # Aromatic fragment. The node colors distinguish the first BFS shells.
    cx, cy, radius = 7.6, 17.6, 4.0
    angles = [-pi / 2 + i * pi / 3 for i in range(6)]
    ring = [(cx + radius*cos(a), cy + radius*sin(a)) for a in angles]
    d.line(ring + [ring[0]], lw=.99)
    d.dot(cx, cy, 2.13, fill=SOFT_BLUE, edge=INK, lw=.7)
    for i, ((x, y), angle) in enumerate(zip(ring, angles)):
        x2, y2 = cx + 6.65*cos(angle), cy + 6.65*sin(angle)
        d.line([(x, y), (x2, y2)], lw=.85)
        d.dot(x2, y2, .64, fill=WHITE, edge=BLUE, lw=.69)
        d.dot(x, y, .60, fill=BLUE if i == 0 else GATE, edge=BLUE, lw=.59)
    d.text(14.2, 11.2, "BFS", size=7.8, weight="bold", align="left")
    for y, word in zip([15.1, 18.65, 22.2, 25.75], ["atom", "bond", "ring", "depth"]):
        d.dot(14.65, y, .25, edge=BLUE, fill=BLUE, lw=.45)
        d.text(15.8, y, word, size=7.5, align="left")

    d.box(.7, 32.2, 25.8, 65.5, fill=SOFT_BLUE, edge=BLUE)
    d.text(2.3, 36.1, "Ethene frames", size=8.6, weight="bold", align="left")
    d.text(2.3, 40.8, r"$Q=15,\ T=20$", size=8.2, align="left")
    # Schematic ethene frames, not sampled molecular geometries.
    for y, label in [(48.5, r"$t=1$"), (59.4, r"$t=2$"), (73.4, r"$t=T$")]:
        left, right = 6.0, 12.2
        for dy in [-.39, .39]:
            d.line([(left, y+dy), (right, y+dy)], lw=.88)
        for x, sign in [(left, -1), (right, 1)]:
            for dy in [-2.62, 2.62]:
                xh, yh = x + sign*2.55, y + dy
                d.line([(x, y), (xh, yh)], lw=1.12)
                d.dot(xh, yh, .74, edge=BLUE, fill=WHITE, lw=.64)
            d.dot(x, y, 1.12, fill=BLUE, edge=INK, lw=.72)
            d.dot(x-.28, y-.30, .31, fill="#93AABB", edge="#93AABB", lw=.3, layer=6)
        d.text(17.0, y, label, size=8.1, align="left")
    d.text(9.1, 65.9, r"$\vdots$", size=9.4, color=BLUE)
    d.line([(2.3, 79.7), (24.8, 79.7)], color=EDGE, lw=.64)
    d.text(2.3, 83.5, "Frame encoding", size=7.65, weight="bold", align="left")
    d.text(2.3, 88.0, r"Distance $\to X$", size=7.6, align="left")
    d.text(2.3, 92.2, r"Radial velocity $\to Z$", size=7.15, align="left")
    d.text(2.3, 95.6, r"$Y=0$", size=7.45, align="left")

    # Both types of classical input enter the same event representation.
    d.line([(26.5, 17.6), (28.1, 17.6), (28.1, 54.0), (26.5, 54.0)],
           color=BLUE, lw=.88)
    d.arrow([(28.1, 36.3), (30.1, 36.3)], color=BLUE, head=5.5)


def event_tensor(d):
    d.box(30.15, 12.3, 18.3, 55.2, fill=WHITE, edge=BLUE)
    d.text(39.3, 16.5, "Token tensor", size=7.55, weight="bold")
    d.text(39.3, 22.0, r"$X_T=(x_1,\ldots,x_T)$", size=7.35)
    d.text(39.3, 27.0, r"$x_t\in\mathbb{R}^{3Q}$", size=7.8)
    d.text(39.4, 33.0, r"At event $t$", size=7.75, color=BLUE)
    col = [34.25, 39.5, 44.75]
    for x, axis in zip(col, "XYZ"):
        d.text(x, 39.0, rf"${axis}$", size=7.8)
    # Each row gives the three generator coefficients for one qubit.
    for y, site in [(44.2, "1"), (53.7, "i"), (63.2, "Q")]:
        d.box(31.25, y-2.65, 16.1, 5.3, fill=GATE, edge=EDGE, radius=.45, lw=.51)
        for x, axis in zip(col, "XYZ"):
            d.text(x, y, rf"$x^{{{axis}}}_{{t,{site}}}$", size=7.15)
    for y in [48.8, 58.25]:
        d.text(39.4, y, r"$\vdots$", size=7.85, color=BLUE)


def reservoir(d):
    d.box(50.8, 1.5, 78.2, 69.7, edge=EDGE, lw=.9)
    d.text(52.9, 5.9, "Coherent Floquet reservoir", size=9.3, weight="bold", align="left")
    d.text(127.0, 6.0, r"$g,\ J_i,\ h_i,\ L$", size=8.1, align="right")
    d.line([(52.7, 9.15), (127.1, 9.15)], color=EDGE, lw=.58)

    # The event gate acts once. Only the blue group is repeated L times.
    d.box(58.0, 20.15, 9.05, 29.2, fill=SOFT_BLUE, edge=EDGE, radius=.6, lw=.55)
    d.box(68.0, 15.9, 34.4, 34.5, fill=FIELD, edge=EDGE, radius=.75)
    d.text(85.2, 12.85, r"$U_F=U_{J,h}U_X,\quad L=3$", size=8.5)
    d.text(62.1, 13.0, r"$U_{\rm in}(x_t)$", size=8.25)
    d.arrow([(48.45, 19.0), (62.55, 19.0), (62.55, 20.1)], color=BLUE, head=5.8)
    d.text(74.6, 18.55, r"$\pi g$", size=7.5)
    d.text(84.8, 18.55, r"$\pi h_i$", size=7.5)
    d.text(96.1, 18.55, r"$J_i$", size=7.5)

    rows = [(24.3, "1"), (30.55, "2"), (40.4, "Q-1"), (46.65, "Q")]
    for y, i in rows:
        d.text(57.8, y, rf"$q_{{{i}}}$", size=7.25, align="right")
        d.line([(58.05, y), (102.15, y)], lw=.78)
        d.box(58.75, y-1.98, 7.6, 3.96, fill=WHITE, edge=BLUE, radius=.52, lw=.7, layer=5)
        d.text(62.55, y, rf"$R_{{{i}}}$", size=7.5)
        for x, label in [(74.6, r"$R_x$"), (84.8, r"$R_z$")]:
            d.box(x-3.65, y-1.98, 7.3, 3.96, fill=GATE, edge=BLUE, radius=.5, lw=.7, layer=5)
            d.text(x, y, label, size=7.7)
    for x in [62.55, 74.6, 84.8, 96.1]:
        d.text(x, 35.4, r"$\vdots$", size=8.65, color=BLUE)
    # A symmetric two-site ZZ operation connects neighbors, with no control/target convention.
    for yc in [27.425, 43.525]:
        d.line([(96.1, yc-3.125), (96.1, yc+3.125)], color=INK, lw=.9)
        for y in [yc-3.125, yc+3.125]:
            d.dot(96.1, y, .55, fill=INK, edge=INK, lw=.55)
        d.box(92.05, yc-1.65, 8.1, 3.3, fill=GATE, edge=BLUE, radius=.42, lw=.65, layer=5)
        d.text(96.1, yc, r"$R_{ZZ}$", size=7.7)
    d.line([(102.45, 23.0), (103.35, 23.0), (103.35, 47.9), (102.45, 47.9)],
           color=BLUE, lw=.74)
    d.text(105.3, 31.7, r"$\rho_t^{-}$", size=7.65)
    d.arrow([(103.35, 35.4), (107.4, 35.4)], head=5.6)

    # The final trajectory frame has no reset; the two timings are listed underneath.
    d.box(107.45, 19.5, 17.0, 28.7, fill=RESET, edge=LOOP, radius=.8)
    d.text(115.95, 23.0, "Global reset", size=7.3, color=PLUM, weight="bold")
    for y, label in [(28.6, r"$(1-p)\,\rho_t^{-}$"), (38.4, r"$p\,\rho_0$")]:
        d.box(109.3, y-2.4, 13.3, 4.8, fill=WHITE, edge=LOOP, lw=.6, radius=.55)
        d.text(115.95, y, label, size=7.65, color=PLUM)
    d.text(115.95, 33.45, r"$+$", size=8.15, color=PLUM)
    d.text(115.95, 44.1, r"$\rho_0=|0^Q\rangle\!\langle0^Q|$", size=6.95, color=PLUM)
    d.arrow([(124.45, 35.4), (126.6, 35.4)], color=PLUM, head=5.35)
    d.dot(126.65, 35.4, .65, fill=ACCENT, edge=INK, lw=.55)
    d.text(126.5, 31.95, r"$\rho_t$", size=7.85, background=WHITE)

    # For t < T the updated state re-enters the register; the last state supplies readout.
    # Keep the return label above its arrow and clear of the gate column.
    d.arrow([(126.65, 35.98), (126.65, 52.7), (51.45, 52.7),
             (51.45, 36.45), (56.5, 36.45)], color=LOOP, head=5.6)
    d.text(86.4, 52.7, r"Next event  $t<T$", size=7.15, color=PLUM, background=WHITE)
    d.text(55.05, 34.0, r"$\rho_{t-1}$", size=7.7)
    d.arrow([(127.2, 35.4), ENDPOINT_BRANCH, (ENDPOINT_BRANCH[0], 17.0), (132.45, 17.0)],
            color=TEAL, head=5.45)
    d.text(126.0, 15.1, r"$t=T$", size=7.25, color=TEAL)

    # Weights at the pre-reset state are the probabilities of surviving intervening resets.
    d.box(52.4, 55.0, 40.6, 14.85, fill=SOFT_BLUE, edge=EDGE, lw=.66)
    d.text(72.7, 58.1, r"History weights at $\rho_t^{-}$", size=7.7, weight="bold")
    for x, event, weight in [(59.15, r"$x_{t-\ell}$", r"$(1-p)^\ell$"),
                             (74.0, r"$x_{t-1}$", r"$1-p$"), (87.4, r"$x_t$", r"$1$")]:
        d.box(x-3.35, 61.25, 6.7, 3.62, fill=GATE, edge=EDGE, lw=.5, radius=.4)
        d.text(x, 63.1, event, size=7.0)
        d.text(x, 67.1, weight, size=7.45)
    d.text(66.9, 63.1, r"$\cdots$", size=7.6)
    d.box(94.5, 55.0, 32.9, 14.85, fill=RESET, edge=LOOP, lw=.66)
    d.text(96.1, 58.15, "Reset timing", size=7.7, weight="bold", color=PLUM, align="left")
    d.text(96.1, 63.0, "Graph   after each event", size=7.0, align="left")
    d.text(96.1, 67.25, "Ethene   between frames", size=7.0, align="left")


def readout_and_targets(d):
    d.box(132.5, 1.5, 21.5, 26.5, fill=WHITE, edge=TEAL)
    d.text(143.25, 5.8, r"$Z$-basis", weight="bold", color=TEAL, size=8.5)
    d.text(143.25, 9.8, "measurement", weight="bold", color=TEAL, size=7.9)
    d.text(143.25, 14.15, r"$b^{(s)}\in\{0,1\}^{Q}$", size=7.8)
    d.box(137.95, 17.3, 10.6, 8.2, fill=MEASURE, edge=TEAL, radius=.55)
    d.meter(143.2, 23.7, 6.7)
    d.arrow([(143.25, 28.0), (143.25, 31.9)], color=TEAL)
    d.box(132.5, 32.0, 21.5, 39.2, fill=WHITE, edge=TEAL)
    d.text(143.25, 36.25, "Feature vector", size=8.0, weight="bold", color=TEAL)
    for y, value in [(42.55, r"$\langle Z_i\rangle$"), (49.7, r"$\langle Z_i Z_j\rangle,\ i<j$")]:
        d.box(133.8, y-2.65, 18.9, 5.3, fill=MEASURE, edge=EDGE, lw=.64)
        d.text(143.25, y, value, size=8.2, color=TEAL)
    d.text(143.25, 56.2, r"$D_Q=Q(Q+1)/2$", size=7.15)
    d.text(143.25, 61.7, "78 graph", size=7.65)
    d.text(143.25, 65.75, "120 ethene", size=7.65)
    d.text(143.25, 69.1, "coordinates", size=7.15)

    # The two arrows represent the task-specific classical decoders, as stated in the caption.
    port = (155.7, 48.6)
    d.line([(154.0, port[1]), port], color=TEAL, lw=.88)
    d.dot(*port, .28, fill=TEAL, edge=TEAL, lw=.4)
    for target_y in [32.6, 58.6]:
        d.arrow([port, (port[0], target_y), (157.35, target_y)], color=TEAL, head=5.9)
    d.box(157.45, 23.0, 19.9, 19.3, fill=SOFT_BLUE, edge=BLUE)
    d.text(167.4, 27.35, "BACE / BBBP", size=7.75, weight="bold")
    d.text(167.4, 33.1, "Classification", size=7.1)
    d.text(167.4, 37.05, "scores", size=7.4)
    d.box(157.45, 47.4, 19.9, 22.45, fill=SOFT_BLUE, edge=BLUE)
    d.text(167.4, 51.7, "SHNITSEL A01", size=7.45, weight="bold")
    d.text(167.4, 55.55, "Ethene gaps", size=7.5)
    d.text(167.4, 61.0, "1, 5, 10 fs", size=7.45)
    d.text(167.4, 65.5, "5-fs minimum", size=7.05)


def temporal_multiplexing(d):
    d.box(30.15, 74.2, 147.2, 23.5, fill="#F6F8F9", edge=EDGE)
    d.text(32.25, 80.7, "Optional TM3", size=8.25, weight="bold", align="left")
    d.text(32.25, 85.25, "Ethene", size=7.65, align="left")
    d.text(32.25, 89.45, "Independent runs", size=7.15, align="left")
    # The ethene endpoint is prepared independently at each sampling depth.
    # Route the common input through the two gutters, clear of the reset-timing box.
    input_rail_x = 55.15
    rows = (78.85, 85.95, 93.05)
    d.line([ENDPOINT_BRANCH, (ENDPOINT_BRANCH[0], 72.7),
            (input_rail_x, 72.7), (input_rail_x, rows[-1])], color=TEAL, lw=.72)
    d.dot(*ENDPOINT_BRANCH, .32, fill=TEAL, edge=TEAL, lw=.4)
    for depth, y in enumerate(rows):
        if y != rows[-1]:
            d.dot(input_rail_x, y, .23, fill=TEAL, edge=TEAL, lw=.35)
        d.arrow([(input_rail_x, y), (57.55, y)], color=TEAL, lw=.72, head=4.8)
        d.box(57.6, y-2.2, 8.1, 4.4, fill=WHITE, edge=EDGE, radius=.55, layer=5)
        d.text(61.65, y, r"$\rho_T$", size=8)
        previous = 65.7
        for x in [72.5, 86.0][:depth]:
            d.arrow([(previous, y), (x, y)], color=INK, head=5.65)
            d.box(x, y-2.2, 8.0, 4.4, fill=GATE, edge=BLUE, radius=.55, layer=5)
            d.text(x+4, y, r"$U_F$", size=7.75)
            previous = x + 8
        d.arrow([(previous, y), (100.25, y)], color=INK, head=5.65)
        d.box(100.3, y-2.3, 21.0, 4.6, fill=MEASURE, edge=TEAL, radius=.55, layer=5)
        d.meter(103.5, y+1.1, 3.6)
        d.text(106.1, y, r"$Z$ readout", size=7.2, color=TEAL, align="left")
        d.arrow([(121.3, y), (126.0, y)], color=TEAL, head=5.6)
        d.box(126.0, y-2.18, 7.0, 4.36, fill=MEASURE, edge=TEAL, radius=.55)
        d.text(129.5, y, rf"$\phi_{depth}$", size=7.75, color=TEAL)
    d.line([(134.6, 78.0), (136.0, 78.0), (136.0, 93.9), (134.6, 93.9)],
           color=TEAL, lw=.78)
    d.arrow([(136.0, 85.95), (139.1, 85.95)], color=TEAL, head=5.55)
    d.box(139.2, 77.1, 35.95, 17.7, fill=MEASURE, edge=TEAL)
    d.text(157.2, 80.95, "Concatenate", size=7.85, weight="bold", color=TEAL)
    d.text(157.2, 86.1, r"$(\phi_0,\,\phi_1,\,\phi_2)$", size=8.0)
    d.text(157.2, 91.3, r"$3\times120=360$ coordinates", size=7.25)


def draw_workflow():
    d = Diagram()
    molecular_inputs(d)
    event_tensor(d)
    reservoir(d)
    readout_and_targets(d)
    temporal_multiplexing(d)
    return d.fig
