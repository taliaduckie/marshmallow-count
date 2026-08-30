"""Are the pan and the marshmallows actually separable by brightness?

Yes -- pan p99.5 = 161, marshmallow-interior p5 = 186. An earlier eyeballed
version of this using hand-drawn rectangles said otherwise; those rectangles
included gap and shadow pixels and the conclusion was wrong.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from _common import IMAGE
from count import segment

g = np.array(Image.open(IMAGE).convert("L")).astype(int)
_, cent, _, n = segment(g.astype(np.uint8), 165, 14)
print(f"using {n} detector markers as marshmallow-interior seeds\n")

seed = np.zeros(g.shape, bool)
for r, c in cent:
    r, c = int(round(r)), int(round(c))
    seed[max(0, r-4):r+5, max(0, c-4):c+5] = True
seed &= ndi.binary_erosion(g > 150, np.ones((3, 3)), iterations=1)

far = ndi.distance_transform_edt(~seed) > 26
fg = ndi.binary_opening(g > 165, np.ones((3, 3)), iterations=2)
pan = far & ~ndi.binary_dilation(fg, np.ones((3, 3)), iterations=4)

for nm, m in [("marshmallow interior", seed), ("pan / tray", pan)]:
    v = g[m]
    print(f"{nm:22s} n={v.size:7d}  p5 {np.percentile(v,5):5.0f}  "
          f"p50 {np.percentile(v,50):5.0f}  p95 {np.percentile(v,95):5.0f}  max {v.max():3d}")
p5, p995 = np.percentile(g[seed], 5), np.percentile(g[pan], 99.5)
print(f"\npan p99.5 = {p995:.0f}  vs  marshmallow p5 = {p5:.0f}  -> "
      f"{'OVERLAP' if p995 > p5 else 'SEPARABLE'}")
print(f"pan pixels above the lowest swept threshold (165): {(g[pan] > 165).mean()*100:.2f}%")
