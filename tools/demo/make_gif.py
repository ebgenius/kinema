"""Compose the rendered frames into a labelled side-by-side GIF."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
WIDTH = int(sys.argv[4]) if len(sys.argv) > 4 else 720

LEFT_LABEL = "Kinema  (PyRoki)"
RIGHT_LABEL = "Blender built-in IK"


def font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    frames = sorted(SRC.glob("frame_*.png"))[::STEP]
    if not frames:
        raise SystemExit(f"no frames in {SRC}")

    first = Image.open(frames[0])
    scale = WIDTH / first.width
    size = (WIDTH, int(first.height * scale))
    big, small = font(int(22 * scale * 1.4)), font(int(15 * scale * 1.4))

    out = []
    for path in frames:
        im = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        d = ImageDraw.Draw(im)
        quarter, mid = size[0] // 4, size[0] // 2

        # A divider makes it read as two panels rather than one wide shot.
        d.line([(mid, 0), (mid, size[1])], fill=(90, 94, 104), width=1)
        for x, text, colour in ((quarter, LEFT_LABEL, (150, 220, 160)),
                                (mid + quarter, RIGHT_LABEL, (230, 170, 160))):
            w = d.textlength(text, font=big)
            d.text((x - w / 2, 14), text, font=big, fill=colour)

        note = "same tool path — two turns of the wrist"
        w = d.textlength(note, font=small)
        d.text(((size[0] - w) / 2, size[1] - 30), note, font=small,
               fill=(150, 154, 164))
        out.append(im)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Adaptive palette: a flat background and static camera keep this small.
    quantised = [im.quantize(colors=128, method=Image.MEDIANCUT) for im in out]
    quantised[0].save(OUT, save_all=True, append_images=quantised[1:],
                      duration=int(1000 / 24 * STEP), loop=0, optimize=True)
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  {len(out)} frames  {size[0]}x{size[1]}  {mb:.2f} MB")


main()
