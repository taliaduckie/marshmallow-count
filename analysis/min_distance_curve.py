"""The count-vs-min_distance curve, extended below the swept range.

The brief swept min_distance 14..22. The curve crosses truth at 13-14 -- i.e. the
sweep began at the optimum and ran away from it in one direction, which is the
entire source of the ~5% low bias in the sweep median.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
from _common import IMAGE, TRUTH

g = np.array(Image.open(IMAGE).convert("L"))
ST = np.ones((3, 3), bool)

def run(t, md):
    mask = ndi.binary_opening(g > t, structure=ST, iterations=2)
    d = ndi.distance_transform_edt(mask)
    co = peak_local_max(d, min_distance=md, labels=mask, exclude_border=False)
    mk = np.zeros(d.shape, int)
    mk[tuple(co.T)] = np.arange(1, len(co) + 1)
    a = np.array([p.area for p in regionprops(watershed(-d, mk, mask=mask))])
    return int((a >= 0.40 * np.median(a)).sum())

print(f"{'min_d':>6}" + "".join(f"{t:>8}" for t in (165, 175, 185, 195)))
for md in [6, 8, 10, 12, 13, 14, 15, 16, 18, 20, 22]:
    note = "  <- brief's sweep starts here" if md == 14 else ""
    print(f"{md:>6}" + "".join(f"{run(t, md):>8}" for t in (165, 175, 185, 195)) + note)
print(f"\ntruth = {TRUTH};  measured half-diameter = 14.7px")
