"""Build the README banners from the gallery images.

Usage: python scripts/make_banner.py

Writes docs/assets/banner.svg (desktop) and docs/assets/banner-mobile.svg. Both are self-contained:
the output images are embedded as WebP data URIs. Rerun after the gallery or the catalog size
changes.

The animation: the input prompt types in word by word, signal dots run through a small neural
network, and the output card switches through the gallery images, one model per pass.
"""

from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from PIL import Image

from imagehub.catalog import available_models, model_info

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "docs/assets/gallery"
ASSETS = ROOT / "docs/assets"

PROMPT_LINES = ("a photograph of an", "astronaut riding", "a horse")

# Apache, MIT or OpenRAIL weight licenses only; see docs/design/2026-11-02-readme-banner-design.md.
OUTPUTS = (
    ("CogView4", "cogview4-6b"),
    ("LongCat-Image", "longcat-image"),
    ("Kolors", "kolors"),
    ("Kandinsky 2.2", "kandinsky-2.2"),
    ("AuraFlow", "auraflow-v0.3"),
    ("Würstchen", "wuerstchen-v2"),
    ("PixArt-Σ", "pixart-sigma-xl-1024"),
    ("aMUSEd", "amused-512"),
)

BG = ("#e4f0ff", "#e9e4ff", "#f3e6ff")
INK = "#1e2350"
MUTED = "#4a5080"
INDIGO = "#5b5bd6"
TEAL = "#2bb3a3"
CARD = "#ffffff"
CARD_EDGE = "#c9cdf2"
NODE_FILL = "#f7f6ff"
LINE = "rgba(30,35,80,0.14)"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
EYEBROW = "One API for open text-to-image models."

IMAGE_PX = 256  # embedded resolution, 2x the desktop display size
LAYERS = (3, 4, 4, 3)  # neural network nodes per layer

# Timeline in seconds. One loop types the prompt once, then shows every output in turn.
TYPE_START, WORD_STEP = 0.3, 0.22
PASS_START, PASS_STEP, PASS_TRAVEL = 2.2, 1.7, 0.8
CYCLE = PASS_START + len(OUTPUTS) * PASS_STEP + 0.3
DOTS_PER_PASS = 4


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    text_x: int
    eyebrow_y: int
    title_y: int
    subline_y: int
    eyebrow_size: int
    title_size: int
    subline_size: int
    flow_y: int  # vertical centre of the input card, network and output card
    input_x: int
    input_w: int
    net_x: tuple[int, int]  # first and last layer x
    node_gap: int
    output_x: int
    image: int
    corner_r: tuple[int, int]  # decorative circles: top right, bottom left


DESKTOP = Layout(
    width=1200,
    height=370,
    text_x=62,
    eyebrow_y=135,
    title_y=212,
    subline_y=259,
    eyebrow_size=24,
    title_size=64,
    subline_size=24,
    flow_y=190,
    input_x=590,
    input_w=160,
    net_x=(790, 940),
    node_gap=34,
    output_x=984,
    image=128,
    corner_r=(170, 125),
)
MOBILE = Layout(
    width=600,
    height=440,
    text_x=40,
    eyebrow_y=72,
    title_y=134,
    subline_y=172,
    eyebrow_size=20,
    title_size=56,
    subline_size=20,
    flow_y=310,
    input_x=36,
    input_w=150,
    net_x=(214, 344),
    node_gap=30,
    output_x=380,
    image=128,
    corner_r=(120, 90),
)


def el(tag: str, content: str | None = None, /, **attrs: object) -> str:
    """One SVG element; trailing underscores are dropped and other underscores become hyphens."""
    parts = [tag]
    for key, value in attrs.items():
        name = key.rstrip("_").replace("_", "-")
        parts.append(f"{name}={quoteattr(str(value))}")
    head = " ".join(parts)
    return f"<{head}/>" if content is None else f"<{head}>{content}</{tag}>"


def _counts() -> tuple[int, int]:
    ids = {m for group in available_models().values() for m in group}
    return len({model_info(m).family for m in ids}), len(ids)


def _image_uri(model_id: str) -> str:
    image = Image.open(GALLERY / f"{model_id}.imagehub.webp").convert("RGB")
    image = image.resize((IMAGE_PX, IMAGE_PX), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "WEBP", quality=80, method=6)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()


def _k(seconds: float) -> str:
    return f"{seconds / CYCLE:.4f}"


def _loop(**attrs: object) -> dict[str, object]:
    return {"dur": f"{CYCLE:.1f}s", "repeatCount": "indefinite", **attrs}


def _discrete(values: str, times: str) -> str:
    return el(
        "animate",
        **_loop(attributeName="opacity", values=values, keyTimes=times, calcMode="discrete"),
    )


def _visible(start: float, end: float) -> str:
    """Discrete opacity: hidden, shown between start and end, hidden again."""
    if end >= CYCLE:
        values, times = "0;1", f"0;{_k(start)}"
    else:
        values, times = "0;1;0", f"0;{_k(start)};{_k(end)}"
    return _discrete(values, times)


def _fade_in(start: float, length: float = 0.18) -> str:
    """Opacity 0 until start, fades to 1, stays until the loop restarts."""
    times = f"0;{_k(start)};{_k(start + length)};1"
    return el("animate", **_loop(attributeName="opacity", values="0;0;1;1", keyTimes=times))


def _dot(path_id: str, start: float) -> str:
    """A teal dot that runs along one path once per loop and is hidden the rest of the time."""
    times = f"0;{_k(start)};{_k(start + PASS_TRAVEL)}"
    motion = el(
        "animateMotion",
        el("mpath", href=f"#{path_id}"),
        **_loop(calcMode="linear", keyPoints="0;0;1;1", keyTimes=f"{times};1"),
    )
    fade = _discrete("0;1;0", times)
    return el("circle", motion + fade, r=4, fill=TEAL, opacity=0)


def _network(layout: Layout) -> list[list[tuple[float, float]]]:
    first, last = layout.net_x
    step = (last - first) / (len(LAYERS) - 1)
    return [
        [
            (first + i * step, layout.flow_y + (j - (count - 1) / 2) * layout.node_gap)
            for j in range(count)
        ]
        for i, count in enumerate(LAYERS)
    ]


def build(layout: Layout, families: int, models: int, images: list[tuple[str, str]]) -> str:
    w, h, fy = layout.width, layout.height, layout.flow_y
    nodes = _network(layout)
    rng = random.Random(7)

    gradient = el(
        "linearGradient",
        "".join(el("stop", offset=o, stop_color=c) for o, c in zip((0, 0.5, 1), BG, strict=True)),
        id="bg",
        x1=0,
        y1=0,
        x2=1,
        y2=1,
    )
    shadow = el(
        "filter",
        el("feDropShadow", dx=0, dy=2, stdDeviation=3, flood_color=INK, flood_opacity=0.1),
        id="lift",
        x="-20%",
        y="-20%",
        width="140%",
        height="160%",
    )
    img = layout.image
    ox, oy = layout.output_x, fy - img / 2 - 8
    defs = [
        el("clipPath", el("rect", width=w, height=h, rx=16), id="banner"),
        gradient,
        shadow,
        el("clipPath", el("rect", x=ox + 8, y=oy + 8, width=img, height=img, rx=10), id="out"),
    ]

    # Each pass sends a few dots from the input card through one node per layer to the output card.
    start_x, end_x = layout.input_x + layout.input_w, ox
    dots = []
    for p in range(len(images)):
        for d in range(DOTS_PER_PASS):
            points = [(start_x, fy)] + [rng.choice(layer) for layer in nodes] + [(end_x, fy)]
            path = " L".join(f"{x:.1f} {y:.1f}" for x, y in points)
            defs.append(el("path", id=f"p{p}_{d}", d=f"M{path}", fill="none"))
            dots.append(_dot(f"p{p}_{d}", PASS_START + p * PASS_STEP + d * 0.07))

    body = [
        el("rect", width=w, height=h, fill="url(#bg)"),
        el("circle", cx=w - 50, cy=10, r=layout.corner_r[0], fill="rgba(255,255,255,0.45)"),
        el("circle", cx=10, cy=h + 20, r=layout.corner_r[1], fill="rgba(255,255,255,0.35)"),
    ]

    subline = f"{families} families · {models} models · native PyTorch"
    tx = layout.text_x
    text = [
        el(
            "text",
            escape(EYEBROW),
            x=tx + 2,
            y=layout.eyebrow_y,
            fill=INDIGO,
            font_size=layout.eyebrow_size,
            font_weight=500,
        ),
        el(
            "text",
            "imagehub",
            x=tx,
            y=layout.title_y,
            fill=INK,
            font_size=layout.title_size,
            font_weight=600,
            letter_spacing=f"{-layout.title_size * 0.035:.2f}",
        ),
        el(
            "text",
            escape(subline),
            x=tx + 2,
            y=layout.subline_y,
            fill=MUTED,
            font_size=layout.subline_size,
        ),
    ]
    body.append(el("g", "".join(text), font_family=SANS))

    # Network: edges between neighbouring layers, then the nodes on top.
    edges = [
        el("line", x1=x1, y1=y1, x2=x2, y2=y2)
        for left, right in zip(nodes, nodes[1:], strict=False)
        for x1, y1 in left
        for x2, y2 in right
    ]
    edges += [el("line", x1=start_x, y1=fy, x2=x, y2=y) for x, y in nodes[0]]
    edges += [el("line", x1=x, y1=y, x2=end_x, y2=fy) for x, y in nodes[-1]]
    body.append(el("g", "".join(edges), aria_hidden="true", stroke=LINE, stroke_width=1.3))
    circles = "".join(
        el("circle", cx=x, cy=y, r=8, fill=NODE_FILL, stroke=INDIGO, stroke_width=2)
        for layer in nodes
        for x, y in layer
    )
    body.append(el("g", circles, aria_hidden="true"))
    body.append(el("g", "".join(dots), aria_hidden="true"))

    # Input card: the prompt types in word by word.
    line_h, char_w, font = 18, 7.2, 12
    ix, card_h = layout.input_x, 20 + line_h * len(PROMPT_LINES)
    iy = fy - card_h / 2
    words = []
    t = TYPE_START
    for row, line in enumerate(PROMPT_LINES):
        col = 0
        for word in line.split():
            x = ix + 14 + col * char_w
            y = iy + 28 + row * line_h
            words.append(el("text", _fade_in(t) + escape(word), x=f"{x:.1f}", y=f"{y:.1f}"))
            col += len(word) + 1
            t += WORD_STEP
    caret_x = ix + 14 + len(PROMPT_LINES[-1]) * char_w + 2
    caret_y = iy + 28 + (len(PROMPT_LINES) - 1) * line_h - font + 2
    blink = el(
        "animate",
        attributeName="opacity",
        values="1;0;1",
        dur="1s",
        repeatCount="indefinite",
    )
    card = [
        el(
            "rect",
            x=ix,
            y=f"{iy:.1f}",
            width=layout.input_w,
            height=card_h,
            rx=12,
            fill=CARD,
            stroke=CARD_EDGE,
            filter="url(#lift)",
        ),
        el(
            "text",
            "INPUT",
            x=ix + layout.input_w / 2,
            y=f"{iy - 10:.1f}",
            text_anchor="middle",
            fill=INDIGO,
            font_family=MONO,
            font_size=9.5,
            font_weight=600,
            letter_spacing=1,
        ),
        el("g", "".join(words), font_family=MONO, font_size=font, fill=INK),
        # The caret shows only once the prompt is complete, then blinks.
        el(
            "g",
            _fade_in(t)
            + el(
                "rect",
                blink,
                x=f"{caret_x:.1f}",
                y=f"{caret_y:.1f}",
                width=1.6,
                height=font + 1,
                fill=INDIGO,
            ),
        ),
    ]
    body.append(el("g", "".join(card), aria_hidden="true"))

    # Output card: a small stack of cards; the front one switches image and model name per pass.
    card_w, card_h = img + 16, img + 40
    stack = [
        el(
            "rect",
            x=ox,
            y=oy,
            width=card_w,
            height=card_h,
            rx=12,
            fill=CARD,
            stroke=CARD_EDGE,
            transform=f"rotate({angle} {ox + card_w / 2} {oy + card_h / 2})",
            opacity=0.8,
        )
        for angle in (7, -5)
    ]
    stack.append(
        el(
            "rect",
            x=ox,
            y=oy,
            width=card_w,
            height=card_h,
            rx=12,
            fill=CARD,
            stroke=CARD_EDGE,
            filter="url(#lift)",
        )
    )
    stack.append(
        el(
            "rect",
            x=ox + 8,
            y=oy + 8,
            width=img,
            height=img,
            rx=10,
            fill=NODE_FILL,
            stroke=CARD_EDGE,
            stroke_dasharray="4 4",
        )
    )
    stack.append(
        el(
            "text",
            "OUTPUT",
            x=ox + card_w / 2,
            y=oy - 10,
            text_anchor="middle",
            fill=INDIGO,
            font_family=MONO,
            font_size=9.5,
            font_weight=600,
            letter_spacing=1,
        )
    )
    for p, (name, uri) in enumerate(images):
        shown = PASS_START + p * PASS_STEP + PASS_TRAVEL
        until = shown + PASS_STEP if p < len(images) - 1 else CYCLE
        # The first image is visible in a static render; animation hides it until its pass.
        opacity = 1 if p == 0 else 0
        stack.append(
            el(
                "image",
                _visible(shown, until),
                href=uri,
                x=ox + 8,
                y=oy + 8,
                width=img,
                height=img,
                clip_path="url(#out)",
                preserveAspectRatio="xMidYMid slice",
                opacity=opacity,
            )
        )
        stack.append(
            el(
                "text",
                _visible(shown, until) + escape(name),
                x=ox + card_w / 2,
                y=oy + img + 29,
                text_anchor="middle",
                fill=MUTED,
                font_family=MONO,
                font_size=11,
                opacity=opacity,
            )
        )
    body.append(el("g", "".join(stack), aria_hidden="true"))

    description = (
        "An input prompt types in word by word, passes through a small neural network, and the "
        "output card switches between images generated by different open models. "
        f"{families} families, {models} models, native PyTorch."
    )
    svg = [
        el("title", f"imagehub — {escape(EYEBROW)}", id="title"),
        el("desc", escape(description), id="description"),
        el("defs", "\n".join(defs)),
        el("g", "\n".join(body), clip_path="url(#banner)"),
    ]
    root = el(
        "svg",
        "\n" + "\n".join(svg) + "\n",
        xmlns="http://www.w3.org/2000/svg",
        width=w,
        height=h,
        viewBox=f"0 0 {w} {h}",
        role="img",
        aria_labelledby="title description",
    )
    return root + "\n"


def main() -> None:
    families, models = _counts()
    images = [(name, _image_uri(model_id)) for name, model_id in OUTPUTS]
    for name, layout in (("banner.svg", DESKTOP), ("banner-mobile.svg", MOBILE)):
        path = ASSETS / name
        path.write_text(build(layout, families, models, images), encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
