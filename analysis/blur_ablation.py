"""Does Gaussian blur help? (No -- and blurring the distance transform hurts.)

Blur suppresses small local maxima, which is the cure for OVER-segmentation.
This pipeline's error runs the other way, so blur makes the merging worse.
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

def run(t, md, pre=0.0, dsig=0.0):
    im = ndi.gaussian_filter(g.astype(float), pre) if pre else g.astype(float)
    mask = ndi.binary_opening(im > t, structure=ST, iterations=2)
    d = ndi.distance_transform_edt(mask)
    dp = ndi.gaussian_filter(d, dsig) if dsig else d
    co = peak_local_max(dp, min_distance=md, labels=mask, exclude_border=False)
    mk = np.zeros(d.shape, int)
    mk[tuple(co.T)] = np.arange(1, len(co) + 1)
    a = np.array([p.area for p in regionprops(watershed(-dp, mk, mask=mask))])
    return int((a >= 0.40 * np.median(a)).sum())

MD = [14, 16, 18, 20, 22]
for title, key in [("blur the GRAYSCALE before threshold", "pre"),
                   ("blur the DISTANCE TRANSFORM before peak_local_max", "dsig")]:
    print(f"\n{title}   (t=175)")
    print(f"{'sigma':>7}" + "".join(f"{m:>7}" for m in MD) + "     spread")
    for s in ([0, 0.5, 1.0, 1.5, 2.0] if key == "pre" else [0, 1.0, 2.0, 3.0, 4.0]):
        r = [run(175, m, **{key: s}) for m in MD]
        print(f"{s:>7.1f}" + "".join(f"{v:>7}" for v in r) + f"{max(r)-min(r):>11}")
print(f"\ntruth = {TRUTH}.  Best config across all of the above is sigma=0 everywhere.")
