#!/usr/bin/env python3
"""Render a PDF into mobile cards with a consistent top safe area."""
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageChops

pdf_path, output_dir, dpi, safe_area = sys.argv[1:]
output = Path(output_dir)
with fitz.open(pdf_path) as document:
    if not document.page_count:
        raise ValueError("PDF has no pages")
    digits = len(str(document.page_count))
    for index, page in enumerate(document, 1):
        pixmap = page.get_pixmap(dpi=int(dpi), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        background = Image.new("RGB", image.size, image.getpixel((0, 0)))
        bounds = ImageChops.difference(image, background).getbbox()
        if bounds:
            image = image.crop(bounds)
        card = Image.new("RGB", (pixmap.width, pixmap.height), (29, 30, 32))
        card.paste(image, ((card.width - image.width) // 2, int(safe_area)))
        card.save(output / f"page-{index:0{digits}d}.png")
