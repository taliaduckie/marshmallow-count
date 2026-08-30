#!/usr/bin/env python3
"""Bake marshmallows.png + the detector's markers into a standalone clicker.html.

The template is kept separate from the 400KB build artifact so git only tracks
the ~20KB of source.
"""
import base64, io, json, sys
import numpy as np
from PIL import Image
from count import segment

THRESHOLD, MIN_DISTANCE = 165, 14   # highest-recall corner: easier to delete a
                                    # spurious dot than to spot a missing one

def main():
    gray = np.array(Image.open("marshmallows.png").convert("L"))
    _, centres, _, n = segment(gray, THRESHOLD, MIN_DISTANCE)
    print(f"seeding clicker with {n} markers (t={THRESHOLD}, min_distance={MIN_DISTANCE})")

    im = Image.open("marshmallows.png").convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()

    data = (f'const IMG_DATA="data:image/jpeg;base64,{b64}";\n'
            f'const IMG_W={im.size[0]},IMG_H={im.size[1]};\n'
            f'const AUTO_PTS={json.dumps([[round(float(c),1),round(float(r),1)] for r,c in centres])};\n')

    tpl = open("clicker_template.html").read()
    if "/*__DATA__*/" not in tpl:
        sys.exit("template is missing the /*__DATA__*/ placeholder")
    open("clicker.html","w").write(tpl.replace("/*__DATA__*/", data))
    print(f"wrote clicker.html ({len(b64)//1024}KB of image data)")

if __name__ == "__main__":
    main()
