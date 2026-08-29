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

q = [im.quantize(colors=128, method=Image.MEDIANCUT) for im in out]
q[0].save(OUT, save_all=True, append_images=q[1:],
          duration=int(1000 / 24 * STEP), loop=0, optimize=True)
print(f"wrote {OUT}  {len(out)} frames  {size[0]}x{size[1]}  "
      f"{OUT.stat().st_size / 1e6:.2f} MB")
