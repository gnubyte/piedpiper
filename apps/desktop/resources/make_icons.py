"""Renders PiedPiper's desktop icons (original artwork, drawn here) and writes them in the
formats the installers need:

    windows/piedpiper.ico   program icon            (16-256 px)
    windows/archive.ico     .pp file icon           (16-256 px)
    macos/AppIcon.icns      app icon                (16-1024 px)
    macos/ArchiveIcon.icns  .pp document icon       (16-1024 px)

The design: two chevrons squeezing a bar (compression) in white on a teal-to-indigo tile; the
file icon puts that tile on a page with a folded corner. Run with Pillow installed:

    python apps/desktop/resources/make_icons.py
"""

import io
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
TOP = (20, 184, 166)  # teal
BOTTOM = (79, 70, 229)  # indigo
S = 1024  # master canvas


def gradient(size, top, bottom):
    g = Image.new("RGBA", (1, size))
    for y in range(size):
        t = y / (size - 1)
        g.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)) + (255,))
    return g.resize((size, size))


def tile(size, radius_frac=0.225):
    """The coloured tile with the white compress glyph, `size` px square."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=round(size * radius_frac), fill=255)
    img.paste(gradient(size, TOP, BOTTOM), (0, 0), mask)
    d = ImageDraw.Draw(img)
    u = size / 1000
    w = round(92 * u)
    white = (255, 255, 255, 255)
    # Chevrons pointing at the bar from above and below.
    d.line([(255 * u, 205 * u), (500 * u, 395 * u), (745 * u, 205 * u)], fill=white, width=w, joint="curve")
    d.line([(255 * u, 795 * u), (500 * u, 605 * u), (745 * u, 795 * u)], fill=white, width=w, joint="curve")
    for x in (255, 745):
        for y in (205, 795):
            d.ellipse((x * u - w / 2, y * u - w / 2, x * u + w / 2, y * u + w / 2), fill=white)
    d.rounded_rectangle((215 * u, 462 * u, 785 * u, 538 * u), radius=round(38 * u), fill=white)
    return img


def app_icon(mac):
    """Windows: the tile fills the canvas. macOS: the Big Sur grid (824 px tile in 1024)
    with a soft shadow."""
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    if not mac:
        canvas.alpha_composite(tile(S - 32), (16, 16))
        return canvas
    t = tile(824)
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow_mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(shadow_mask).rounded_rectangle((100, 112, 924, 936), radius=185, fill=90)
    shadow.putalpha(shadow_mask.filter(ImageFilter.GaussianBlur(14)))
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(t, (100, 100))
    return canvas


def document_icon():
    """A page with a folded corner and the tile on it."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    left, top, right, bottom, fold = 176, 64, 848, 960, 200
    page = [(left, top), (right - fold, top), (right, top + fold), (right, bottom), (left, bottom)]
    d.polygon(page, fill=(255, 255, 255, 255))
    d.line(page + [page[0]], fill=(148, 163, 184, 255), width=22, joint="curve")
    corner = [(right - fold, top), (right - fold, top + fold), (right, top + fold)]
    d.polygon(corner, fill=(226, 232, 240, 255))
    d.line(corner + [corner[0]], fill=(148, 163, 184, 255), width=22, joint="curve")
    badge = tile(440)
    img.alpha_composite(badge, ((S - 440) // 2, 400))
    return img


def sized(master, px):
    return master.resize((px, px), Image.LANCZOS)


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def write_ico(master, path, sizes=(16, 20, 24, 32, 40, 48, 64, 96, 128, 256)):
    frames = [png_bytes(sized(master, s)) for s in sizes]
    out = struct.pack("<HHH", 0, 1, len(frames))
    offset = 6 + 16 * len(frames)
    for s, data in zip(sizes, frames):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    path.write_bytes(out + b"".join(frames))


ICNS_TYPES = [
    (b"icp4", 16), (b"icp5", 32), (b"icp6", 64), (b"ic07", 128), (b"ic08", 256), (b"ic09", 512),
    (b"ic10", 1024), (b"ic11", 32), (b"ic12", 64), (b"ic13", 256), (b"ic14", 512),
]


def write_icns(master, path):
    body = b""
    for kind, px in ICNS_TYPES:
        data = png_bytes(sized(master, px))
        body += kind + struct.pack(">I", len(data) + 8) + data
    path.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)


def main():
    (HERE / "windows").mkdir(exist_ok=True)
    (HERE / "macos").mkdir(exist_ok=True)
    write_ico(app_icon(mac=False), HERE / "windows" / "piedpiper.ico")
    write_ico(document_icon(), HERE / "windows" / "archive.ico")
    write_icns(app_icon(mac=True), HERE / "macos" / "AppIcon.icns")
    write_icns(document_icon(), HERE / "macos" / "ArchiveIcon.icns")
    app_icon(mac=True).resize((256, 256), Image.LANCZOS).save(HERE / "icon-256.png")
    print("icons written to", HERE)


if __name__ == "__main__":
    main()
