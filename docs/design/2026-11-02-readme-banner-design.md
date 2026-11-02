# README banner

## Goal

A README banner that says what imagehub does in one look: one text prompt goes through one API and
comes out as images from many open models. It follows the layout of the maintainer's GitHub profile
banner (text on the left, a small animated flow on the right, soft light gradient, rounded corners)
in a cooler palette, so the two read as related but distinct. No characters or mascots.

## Layout (desktop, 1200 × 370)

- **Left: text block**
  - Eyebrow: `One prompt. Every open model.` (accent indigo, 24 px, weight 500)
  - Title: `imagehub` (ink, 64 px, weight 600, tight letter spacing)
  - Subline: `50 families · 118 checkpoints · native PyTorch` (muted, 24 px)
- **Right: flow**, left to right
  1. **Prompt card**: white rounded card, monospace text `a photograph of an astronaut riding a
     horse` over two lines, small text icon.
  2. **imagehub node**: 56 px rounded square with an indigo border and an `ih` monogram.
  3. **Output grid**: 4 × 2 rounded tiles of 64 px, 10 px apart, each with its model name below in
     10 px monospace. Soft curves connect the node to each tile.
- Two faint white circles in the corners, as in the profile banner.

## Colors

| Role | Value |
| --- | --- |
| Background gradient | `#e4f0ff` → `#e9e4ff` → `#f3e6ff` (top-left to bottom-right) |
| Ink (title) | `#1e2350` |
| Muted text | `#4a5080` |
| Accent | `#5b5bd6` (indigo), `#2bb3a3` (teal, flow dots) |
| Cards | `#ffffff`, 1 px `#c9cdf2` border |
| Lines | `rgba(30, 35, 80, 0.16)` |

The banner draws its own background, so it reads the same on GitHub's light and dark themes.

## Tiles

Eight imagehub outputs from `docs/assets/gallery/`, all generated from the same prompt. Only
checkpoints under Apache, MIT or OpenRAIL weight licenses are used; none of the tiles comes from a
non-commercial license:

| Tile label | Gallery image | Weight license |
| --- | --- | --- |
| cogview4 | `cogview4-6b.imagehub.webp` | Apache-2.0 |
| longcat | `longcat-image.imagehub.webp` | Apache-2.0 |
| kolors | `kolors.imagehub.webp` | Apache-2.0 |
| kandinsky | `kandinsky-2.2.imagehub.webp` | Apache-2.0 |
| auraflow | `auraflow-v0.3.imagehub.webp` | Apache-2.0 |
| würstchen | `wuerstchen-v2.imagehub.webp` | MIT |
| pixart-σ | `pixart-sigma-xl-1024.imagehub.webp` | CreativeML-OpenRAIL++-M |
| amused | `amused-512.imagehub.webp` | OpenRAIL++ |

Each tile is downscaled to 128 px (2× the display size) and embedded as a base64 WebP data URI, so
the SVG is self-contained. The Ideogram 4 gallery image is a safety-filter placeholder and is left
out.

## Animation

SMIL only (GitHub renders it inside `<img>`; CSS animations in SVG are less reliable there):

- A teal dot travels from the prompt card to the node, then one dot per curve to each tile,
  staggered by 0.2 s.
- Each tile brightens briefly when its dot arrives.
- One cycle takes about 6 s and repeats.

The resting state shows every tile at full opacity, so a static render (previews, social cards,
reduced-motion contexts) looks complete.

## Mobile (600 × 560)

Stacked: text block on top, then prompt card → node in a row, then the 4 × 2 grid. Same colors and
tiles. The README switches with `<picture>` at `max-width: 600px`, as the profile README does.

## Files

- `scripts/make_banner.py`: builds both SVGs from the gallery images; the tile list and colors are
  constants at the top. Rerun when the gallery or the family count changes.
- `docs/assets/banner.svg`, `docs/assets/banner-mobile.svg`: generated output, committed.
- `README.md`: `<picture>` block at the top with an alt text describing the flow.
- `THIRD_PARTY_NOTICES.md`: one sentence that the banner tiles come from the gallery images and keep
  their recorded licenses.

## Checks

- Render both SVGs in a browser at 1200 px and 375 px width; text does not overlap the flow.
- Each SVG stays under 150 KB.
- The first frame shows all eight tiles.
