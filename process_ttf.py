import sys
from tqdm import tqdm
from pathlib import Path
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
import numpy as np

def read_font(fontfile, size=150):
    font = ImageFont.truetype(str(fontfile), size=size)
    return font


def render(font, char, size=(128, 128), pad=20):
    width, height = font.getsize(char)
    max_size = max(width, height)

    if width < height:
        start_w = (height - width) // 2 + pad
        start_h = pad
    else:
        start_w = pad
        start_h = (width - height) // 2 + pad

    img = Image.new("L", (max_size+(pad*2), max_size+(pad*2)), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_w, start_h), char, font=font)
    img = img.resize(size, 2)
    return img

def get_defined_chars(fontfile):
    ttf = TTFont(fontfile)
    chars = [chr(y) for y in ttf["cmap"].tables[0].cmap.keys()]
    return chars

def get_filtered_chars(fontpath):
    ttf = TTFont(fontpath)
    m_dict = ttf.getBestCmap()

    unicode_list = []
    for key, uni in m_dict.items():
        unicode_list.append(key)

    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list


def main():
    root_dir = sys.argv[1]
    print(root_dir)
    ttffiles = sorted(Path(root_dir).rglob("*.ttf"))

    for ttffile in tqdm(ttffiles):
        txtfile = Path(str(ttffile).replace(".ttf", ".txt"))
        try:
            avail_chars = get_filtered_chars(ttffile)
            with open(txtfile, "w") as f:
                f.write("".join(avail_chars))
        except:
            print(ttffile)


if __name__ == "__main__":
    main()