"""Draw the repository figures as plain SVG. Standard library only.

    python figures/make_figures.py

Writes figures/pipeline.svg, figures/pipeline_narrow.svg, figures/timeline.svg and timeline_narrow.svg,
then copies them to docs/assets/. Colours follow the viewer's light or dark setting.
"""
import os
import shutil
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"

STYLE = """
<style>
  text { font-family: %s; fill: #24292f; }
  .t { font-size: 14px; font-weight: 600; }
  .d { font-size: 12px; fill: #57606a; }
  .h { font-size: 12px; font-weight: 600; letter-spacing: .04em; fill: #57606a; }
  .box { fill: #ffffff; stroke: #d0d7de; stroke-width: 1; }
  .box.key { fill: #e8f3ee; stroke: #2f7d5b; }
  .box.input { fill: #fff8e8; stroke: #c98500; stroke-dasharray: 4 3; }
  .band { fill: #f6f8fa; stroke: none; }
  .edge { stroke: #57606a; stroke-width: 1.5; fill: none; }
  .edge.dash { stroke-dasharray: 4 3; }
  .arrow { fill: #57606a; }
  .grid { stroke: #d8dee4; stroke-width: 1; }
  .tr { fill: #2a78d6; } .va { fill: #eb6834; } .tn { fill: #1baf7a; } .te { fill: #eda100; }
  @media (prefers-color-scheme: dark) {
    text { fill: #e6edf3; }
    .d, .h { fill: #9da7b3; }
    .box { fill: #161b22; stroke: #3d444d; }
    .box.key { fill: #12261e; stroke: #4aa37c; }
    .box.input { fill: #2a2110; stroke: #c98500; }
    .band { fill: #11161c; }
    .edge { stroke: #9da7b3; } .arrow { fill: #9da7b3; }
    .grid { stroke: #30363d; }
    .tr { fill: #3987e5; } .va { fill: #d95926; } .tn { fill: #199e70; } .te { fill: #c98500; }
  }
</style>""" % FONT

NODES = {
    "video": ("1  Overhead video", ["2304x1296, one camera,", "decoded to 5 fps"], ""),
    "track": ("2  Tracker", ["SAM 3 with a distilled", "TinyViT student. RF-DETR", "detector re-seeds it"], "key"),
    "segs": ("3  Anonymous segments", ["One RLE mask per frame.", "No identity yet"], ""),
    "crops": ("4  Crops, embeddings", ["Masked crops every 15 s,", "DINOv3 ViT-S/16"], ""),
    "anchor": ("Anchor pool", ["30,588 verified crops", "from 04-18 and 04-19"], "input"),
    "ident": ("5  Identity layer", ["3-min windows, Hungarian", "match over runs. Says", "'?' when unsure"], "key"),
    "colour": ("6  Colour filter", ["Hides tracks that are", "not a black and white", "calf"], ""),
    "refine": ("7  Mask refinement", ["Detector repairs mask", "fragments. Identity", "does not change"], ""),
}


def node(x, y, w, h, key):
    title, lines, cls = NODES[key]
    s = ['<rect class="box %s" x="%s" y="%s" width="%s" height="%s" rx="8"/>' % (cls, x, y, w, h),
         '<text class="t" x="%s" y="%s">%s</text>' % (x + 12, y + 23, escape(title))]
    for i, line in enumerate(lines):
        s.append('<text class="d" x="%s" y="%s">%s</text>' % (x + 12, y + 43 + i * 16, escape(line)))
    return "\n".join(s)


def arrow(pts, dash=False):
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    if x1 > x0:
        tri, end = "%s,%s %s,%s %s,%s" % (x1, y1, x1 - 8, y1 - 4, x1 - 8, y1 + 4), (x1 - 8, y1)
    elif x1 < x0:
        tri, end = "%s,%s %s,%s %s,%s" % (x1, y1, x1 + 8, y1 - 4, x1 + 8, y1 + 4), (x1 + 8, y1)
    elif y1 > y0:
        tri, end = "%s,%s %s,%s %s,%s" % (x1, y1, x1 - 4, y1 - 8, x1 + 4, y1 - 8), (x1, y1 - 8)
    else:
        tri, end = "%s,%s %s,%s %s,%s" % (x1, y1, x1 - 4, y1 + 8, x1 + 4, y1 + 8), (x1, y1 + 8)
    d = "M" + " L".join("%s %s" % p for p in pts[:-1] + [end])
    return '<path class="edge%s" d="%s"/><polygon class="arrow" points="%s"/>' % (" dash" if dash else "", d, tri)


def pipeline():
    w, bw, bh, gap = 980, 216, 104, 28
    xs = [20 + i * (bw + gap) for i in range(4)]
    y1, y2 = 56, 228
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d 352" role="img" '
         'aria-label="Pipeline: video, tracker, anonymous segments, crops and embeddings, then an offline '
         'identity layer with an anchor pool, a colour filter and mask refinement">' % w, STYLE,
         '<rect class="band" x="4" y="8" width="%d" height="164" rx="12"/>' % (w - 8),
         '<rect class="band" x="4" y="180" width="%d" height="164" rx="12"/>' % (w - 8),
         '<text class="h" x="20" y="36">PER FRAME, ON A GPU</text>',
         '<text class="h" x="20" y="208">OFFLINE. APPEARANCE NEVER STEERS TRACKING</text>']
    for x, k in zip(xs, ("video", "track", "segs", "crops")):
        s.append(node(x, y1, bw, bh, k))
    for x, k in zip(xs, ("anchor", "ident", "colour", "refine")):
        s.append(node(x, y2, bw, bh, k))
    for i in range(3):
        s.append(arrow([(xs[i] + bw, y1 + bh // 2), (xs[i + 1], y1 + bh // 2)]))
    s.append(arrow([(xs[3] + bw // 2, y1 + bh), (xs[3] + bw // 2, 176), (xs[1] + bw // 2, 176),
                    (xs[1] + bw // 2, y2)]))
    s.append(arrow([(xs[0] + bw, y2 + bh // 2), (xs[1], y2 + bh // 2)], dash=True))
    s.append(arrow([(xs[1] + bw, y2 + bh // 2), (xs[2], y2 + bh // 2)]))
    s.append(arrow([(xs[2] + bw, y2 + bh // 2), (xs[3], y2 + bh // 2)]))
    s.append("</svg>")
    return "\n".join(s)


def pipeline_narrow():
    w, bw, bh, gap = 340, 300, 100, 24
    keys = ("video", "track", "segs", "crops", "ident", "colour", "refine")
    x, y0 = 20, 40
    h = y0 + len(keys) * (bh + gap)
    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" role="img" '
         'aria-label="Pipeline as seven steps from top to bottom">' % (w, h), STYLE,
         '<text class="h" x="20" y="24">PER FRAME (1-3), THEN OFFLINE (4-7)</text>']
    for i, k in enumerate(keys):
        y = y0 + i * (bh + gap)
        s.append(node(x, y, bw, bh, k))
        if i:
            s.append(arrow([(x + bw // 2, y - gap), (x + bw // 2, y)]))
    s.append("</svg>")
    return "\n".join(s)


DAYS = [  # label, recording windows in hours, colour class, text
    ("04-18", [(9 + 27 / 60, 23 + 55 / 60)], "tr", "Train. Identity reviewed. Anchors come from here"),
    ("04-19", [(5 + 58 / 60, 23 + 59 / 60)], "tr", "Train. Identity reviewed. Anchors come from here"),
    ("04-20, 21", [], "", "No recording. 60.75 h gap before the test week"),
    ("04-22", [(12 + 45 / 60, 23 + 38 / 60)], "va", "Validation. Identity verified. Scored"),
    ("04-23", [(5 + 44 / 60, 23 + 51 / 60)], "va", "Validation. Identity verified. Scored"),
    ("04-24", [(5 + 48 / 60, 23 + 53 / 60)], "tn", "Test. Scored against interval notes"),
    ("04-25", [(5 + 44 / 60, 11 + 49 / 60), (12 + 23 / 60, 23 + 49 / 60)], "te", "Test. Expert review, no numeric score"),
    ("04-26", [(5 + 44 / 60, 16 + 55 / 60), (17 + 29 / 60, 23 + 51 / 60)], "te", "Test. Expert review, no numeric score"),
    ("04-27", [(5 + 45 / 60, 23 + 35 / 60)], "te", "Test. Expert review, no numeric score"),
    ("04-28", [(5 + 40 / 60, 23 + 45 / 60)], "te", "Test. Expert review, no numeric score"),
]


def timeline(narrow=False):
    w, left, right, top, rh = (360, 66, 8, 30, 40) if narrow else (500, 70, 12, 30, 40)
    h = top + rh * len(DAYS) + 8

    def sx(hour):
        return left + (w - left - right) * hour / 24

    s = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" role="img" '
         'aria-label="Recording timeline by day and hour. Recording stops for about 6 hours every night.">' % (w, h),
         STYLE]
    for hour in (0, 6, 12, 18, 24):
        s.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (sx(hour), sx(hour), top - 8, h - 6))
        s.append('<text class="d" x="%.1f" y="%d" text-anchor="%s">%02d:00</text>'
                 % (sx(hour), top - 13, "end" if hour == 24 else "middle", hour))
    for i, (label, windows, cls, text) in enumerate(DAYS):
        y = top + i * rh
        s.append('<text class="t" x="%d" y="%d" text-anchor="end" style="font-size:13px">%s</text>'
                 % (left - 8, y + 29, escape(label)))
        s.append('<text class="d" x="%d" y="%d"%s>%s</text>'
                 % (left + 4, y + 13, ' style="font-size:10.5px"' if narrow else "", escape(text)))
        for a, b in windows:
            s.append('<rect class="%s" x="%.1f" y="%d" width="%.1f" height="13" rx="3"/>'
                     % (cls, sx(a), y + 19, sx(b) - sx(a)))
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    out = {"pipeline.svg": pipeline(), "pipeline_narrow.svg": pipeline_narrow(), "timeline.svg": timeline(),
           "timeline_narrow.svg": timeline(narrow=True)}
    assets = os.path.join(HERE, "..", "docs", "assets")
    os.makedirs(assets, exist_ok=True)
    for name, text in out.items():
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write(text)
        shutil.copy(os.path.join(HERE, name), assets)
