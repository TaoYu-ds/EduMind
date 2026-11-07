"""Render the two business-flow diagrams as PNGs without a Mermaid viewer.

Requires Pillow. Run from any directory:
    python3 wiki/assets/flows/render_business_flows.py

Labels and edges come from the adjacent .mmd files. Update the layout below
when adding or removing nodes. Pass --font to use another Chinese font.
"""

from pathlib import Path
import argparse
import math
import re

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SCALE = 2
COLORS = {
    "ink": "#213449", "muted": "#68788A", "edge": "#61758A",
    "process": ("#F0F6FF", "#7193C2"),
    "decision": ("#FFF6DC", "#CEAA4C"),
    "terminal": ("#EDF8F1", "#70A68A"),
}
LAYOUTS = {
    "chat-flow-current": {
        "size": (1500, 2000),
        "title": "一条 /chat 请求怎样完成",
        "subtitle": "回执复用直接返回；其余请求继续执行并提交结果",
        "nodes": {
            "A": (750, 190, 800, 100), "B": (750, 350, 820, 110),
            "C": (750, 550, 660, 180), "D": (1290, 550, 300, 110),
            "E": (750, 770, 840, 110), "F": (750, 970, 600, 170),
            "G": (235, 970, 350, 120), "H": (750, 1170, 820, 110),
            "I": (750, 1340, 820, 110), "J": (750, 1510, 820, 110),
            "K": (750, 1680, 820, 110), "M": (750, 1860, 900, 120),
        },
        "terminal": {"A", "D", "M"},
        "routes": {("G", "J"): [(235, 1030), (235, 1510), (340, 1510)]},
    },
    "memory-maintenance-current": {
        "size": (1500, 1740),
        "title": "后台怎样维护会话记忆",
        "subtitle": "先尝试压缩，再更新画像；聊天响应不等待维护完成",
        "nodes": {
            "A": (750, 210, 820, 110), "B": (750, 380, 850, 110),
            "C": (750, 580, 640, 180), "D": (750, 800, 860, 120),
            "E": (750, 1010, 660, 180), "F": (750, 1230, 880, 120),
            "G": (750, 1420, 880, 120), "H": (750, 1610, 880, 110),
        },
        "terminal": {"A", "H"},
        "routes": {
            ("C", "H"): [(1070, 580), (1360, 580), (1360, 1610), (1190, 1610)],
            ("E", "H"): [(1080, 1010), (1360, 1010), (1360, 1610), (1190, 1610)],
        },
    },
}


def point(p):
    return tuple(round(v * SCALE) for v in p)


def font_path(requested):
    if requested:
        return requested
    for candidate in (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(candidate).is_file():
            return candidate
    raise SystemExit("Chinese font not found; pass --font /path/to/font.ttf")


def parse(source):
    node_pattern = re.compile(r'(\w+)\s*(\["([^"]+)"\]|\{"([^"]+)"\})')
    nodes = {}
    edges = []
    for line in source.splitlines():
        for match in node_pattern.finditer(line):
            nodes[match[1]] = (match[3] or match[4], bool(match[4]))
        line = node_pattern.sub(lambda m: m[1], line.strip())
        edge = re.fullmatch(r'(\w+)\s*-->\s*(?:\|([^|]+)\|\s*)?(\w+)', line)
        if edge:
            edges.append((edge[1], edge[3], edge[2] or ""))
        elif line and line != "flowchart TD":
            raise ValueError(f"Unsupported Mermaid line: {line}")
    return nodes, edges


def render(name, layout, font_file):
    nodes, edges = parse((ROOT / f"{name}.mmd").read_text(encoding="utf-8"))
    assert set(nodes) == set(layout["nodes"]), "Update the diagram layout for changed nodes"
    width, height = layout["size"]
    im = Image.new("RGB", point((width, height)), "white")
    draw = ImageDraw.Draw(im)
    fonts = {size: ImageFont.truetype(font_file, size * SCALE) for size in (28, 32, 35, 49)}

    def text_center(x, y, label, size=35, color=None):
        draw.text(point((x, y)), label, font=fonts[size], anchor="mm", fill=color or COLORS["ink"])

    def label_lines(label, max_width):
        result, current = [], ""
        for ch in label:
            if current and draw.textlength(current + ch, font=fonts[35]) > max_width * SCALE:
                result.append(current)
                current = ch
            else:
                current += ch
        if current:
            result.append(current)
        return result

    def arrow(points, label):
        draw.line([point(p) for p in points], fill=COLORS["edge"], width=3 * SCALE, joint="curve")
        before, tip = points[-2:]
        angle = math.atan2(tip[1] - before[1], tip[0] - before[0])
        wings = [(tip[0] - 16 * math.cos(angle + a), tip[1] - 16 * math.sin(angle + a)) for a in (-.48, .48)]
        draw.polygon([point(tip)] + [point(p) for p in wings], fill=COLORS["edge"])
        if label:
            a, b = points[:2]
            x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            if abs(a[0] - b[0]) < 1:
                x += 34
            else:
                y -= 28
            draw.rounded_rectangle([*point((x - 27, y - 22)), *point((x + 27, y + 22))], radius=8 * SCALE, fill="white")
            text_center(x, y, label, 28, COLORS["muted"])

    text_center(width / 2, 58, layout["title"], 49)
    text_center(width / 2, 112, layout["subtitle"], 28, COLORS["muted"])

    for src, dst, label in edges:
        if (src, dst) in layout["routes"]:
            route = layout["routes"][(src, dst)]
        else:
            x1, y1, w1, h1 = layout["nodes"][src]
            x2, y2, w2, h2 = layout["nodes"][dst]
            if y1 == y2:
                direction = 1 if x2 > x1 else -1
                route = [(x1 + direction * w1 / 2, y1), (x2 - direction * w2 / 2, y2)]
            else:
                assert x1 == x2, f"Add a route for {src} -> {dst}"
                route = [(x1, y1 + h1 / 2), (x2, y2 - h2 / 2)]
        arrow(route, label)

    for key, (label, diamond) in nodes.items():
        x, y, w, h = layout["nodes"][key]
        fill, border = COLORS["decision" if diamond else "terminal" if key in layout["terminal"] else "process"]
        if diamond:
            draw.polygon([point(p) for p in ((x, y-h/2), (x+w/2, y), (x, y+h/2), (x-w/2, y))], fill=fill)
            draw.line([point(p) for p in ((x, y-h/2), (x+w/2, y), (x, y+h/2), (x-w/2, y), (x, y-h/2))], fill=border, width=2*SCALE)
        else:
            draw.rounded_rectangle([*point((x-w/2, y-h/2)), *point((x+w/2, y+h/2))], radius=20*SCALE, fill=fill, outline=border, width=2*SCALE)
        lines = label_lines(label, w * .64 if diamond else w - 48)
        assert len(lines) * 46 <= h - 12, f"Node {key} needs more height"
        for index, line in enumerate(lines):
            text_center(x, y + (index - (len(lines)-1)/2) * 46, line)

    im = im.resize((width, height), Image.Resampling.LANCZOS)
    target = ROOT / f"{name}.png"
    im.save(target, optimize=True)
    print(f"Saved {target.name}: {width}x{height}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Path to a Chinese TrueType/OpenType font")
    args = parser.parse_args()
    font_file = font_path(args.font)
    for name, layout in LAYOUTS.items():
        render(name, layout, font_file)
