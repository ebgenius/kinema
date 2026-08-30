"""Compose a single-panel GIF with a title and a caption."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
STEP = int(sys.argv[3])
WIDTH = int(sys.argv[4])
TITLE = sys.argv[5]
CAPTION = sys.argv[6]
#: Palette size. A flat background and a static camera mean these frames carry
#: far less colour than the default 128 implies, and halving it halves the file
#: with no visible banding on shaded grey robots.
COLORS = int(sys.argv[7]) if len(sys.argv) > 7 else 64


def font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


frames = sorted(SRC.glob("frame_*.png"))[::STEP]
first = Image.open(frames[0])
size = (WIDTH, int(first.height * WIDTH / first.width))
big, small = font(int(size[0] / 26)), font(int(size[0] / 40))

out = []
for path in frames:
    im = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    d = ImageDraw.Draw(im)
    w = d.textlength(TITLE, font=big)
    d.text(((size[0] - w) / 2, 16), TITLE, font=big, fill=(225, 228, 236))
    w = d.textlength(CAPTION, font=small)
    d.text(((size[0] - w) / 2, size[1] - 34), CAPTION, font=small,
           fill=(240, 160, 90))
    out.append(im)

# One palette for the whole animation, and no dithering.
#
# Pillow only honours `dither` when an explicit palette is passed -- with
# method=MEDIANCUT and no palette the argument is silently ignored, which is why
# lowering the colour count alone barely moves the file. Floyd-Steinberg sprays
# high-frequency noise over flat surfaces and noise is exactly what GIF's
# run-length coding cannot pack. A smooth-shaded render under a static camera
# has nothing subtle enough to need it, and a shared palette also spares every
# frame its own colour table.
palette = out[len(out) // 2].quantize(colors=COLORS, method=Image.MEDIANCUT)
q = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in out]
q[0].save(OUT, save_all=True, append_images=q[1:],
          duration=int(1000 / 24 * STEP), loop=0, optimize=True)
print(f"wrote {OUT}  {len(out)} frames  {size[0]}x{size[1]}  "
      f"{OUT.stat().st_size / 1e6:.2f} MB")
