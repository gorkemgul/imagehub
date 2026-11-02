# README banner

## Goal

A README banner that shows what imagehub is: one API that runs many open text-to-image models. A
text input goes through a neural network and comes out as images from different models. It follows
the layout of the maintainer's GitHub profile banner (text on the left, a small animated flow on the
right, soft light gradient, rounded corners) in a cooler palette. No characters or mascots.

## Layout (desktop, 1200 × 370)

- **Left: text block**
  - Eyebrow: `One API for open text-to-image models.` (indigo, 24 px, weight 500)
  - Title: `imagehub` (ink, 64 px, weight 600, tight letter spacing)
  - Subline: `50 families · 118 models · native PyTorch` (muted, 24 px)
- **Right: flow**, left to right
  1. **Input card**: `INPUT` label above a white card; the prompt `a photograph of an astronaut
     riding a horse` in monospace over three lines.
  2. **Neural network**: four layers of 3, 4, 4 and 3 nodes with faint edges, as in the profile
     banner. Edges also join the input card to the first layer and the last layer to the output.
  3. **Output card**: `OUTPUT` label above a white card holding one 128 px image and the model name
     under it. Two tilted cards behind it hint that there are many outputs. Before the first pass
     the image slot is a dashed placeholder.
- Two faint white circles in the corners, as in the profile banner.

## Animation

SMIL only (GitHub renders it inside `<img>`). One loop takes about 16 s:

1. The prompt types in word by word (0.22 s per word); a caret blinks after the last word.
2. For each output model in turn, four teal dots run from the input card through one node per layer
   to the output card (0.8 s).
3. When the dots arrive, the output image and model name switch to that model. Each model stays for
   1.7 s, then the next pass starts.
4. The loop restarts with an empty prompt and the placeholder.

Without SMIL (static renders) the full prompt and the first output are shown.

## Colors

| Role | Value |
| --- | --- |
| Background gradient | `#e4f0ff` → `#e9e4ff` → `#f3e6ff` (top-left to bottom-right) |
| Ink (title, prompt) | `#1e2350` |
| Muted text | `#4a5080` |
| Accent | `#5b5bd6` (indigo: labels, nodes, caret), `#2bb3a3` (teal: signal dots) |
| Cards | `#ffffff`, 1 px `#c9cdf2` border; nodes `#f7f6ff` |
| Edges | `rgba(30, 35, 80, 0.14)` |

The banner draws its own background, so it reads the same on GitHub's light and dark themes.

## Output images

Eight imagehub outputs from `docs/assets/gallery/`, all generated from the same prompt. Only
checkpoints under Apache, MIT or OpenRAIL weight licenses are used; none of the images comes from a
non-commercial license:

| Shown as | Gallery image | Weight license |
| --- | --- | --- |
| CogView4 | `cogview4-6b.imagehub.webp` | Apache-2.0 |
| LongCat-Image | `longcat-image.imagehub.webp` | Apache-2.0 |
| Kolors | `kolors.imagehub.webp` | Apache-2.0 |
| Kandinsky 2.2 | `kandinsky-2.2.imagehub.webp` | Apache-2.0 |
| AuraFlow | `auraflow-v0.3.imagehub.webp` | Apache-2.0 |
| Würstchen | `wuerstchen-v2.imagehub.webp` | MIT |
| PixArt-Σ | `pixart-sigma-xl-1024.imagehub.webp` | CreativeML-OpenRAIL++-M |
| aMUSEd | `amused-512.imagehub.webp` | OpenRAIL++ |

Each image is downscaled to 256 px (2× the display size) and embedded as a base64 WebP data URI, so
the SVG is self-contained. The Ideogram 4 gallery image is a safety-filter placeholder and is left
out.

## Mobile (600 × 440)

Text block on top, the input card → network → output card row below. Same colors, images and
animation. The README switches with `<picture>` at `max-width: 600px`, as the profile README does.

## Files

- `scripts/make_banner.py`: builds both SVGs from the gallery images; the output list, colors and
  timeline are constants at the top. Rerun when the gallery or the family count changes.
- `docs/assets/banner.svg`, `docs/assets/banner-mobile.svg`: generated output, committed.
- `README.md`: `<picture>` block at the top with an alt text describing the flow.
- `THIRD_PARTY_NOTICES.md`: one paragraph that the banner images come from the gallery and keep
  their recorded licenses.

## Checks

- Render both SVGs in a browser at 1200 px and 375 px width; text does not overlap the flow.
- Each SVG stays under 150 KB.
- Step through the timeline: words appear in order, the caret shows only after the last word, every
  pass switches the image and the name together.
