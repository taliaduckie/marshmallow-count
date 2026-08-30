"""How much of the count comes from thresholding, and how much from watershed?

Answer: thresholding alone gets you less than half. At t=165 the single largest
connected component contains ~160 touching marshmallows.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from _common import IMAGE, TRUTH

g = np.array(Image.open(IMAGE).convert("L"))
ST = np.ones((3, 3), bool)

print(f"{'thresh':>7}{'raw blobs':>11}{'after area cut':>16}{'largest blob':>14}{'= mallows':>11}")
for t in (165, 175, 185, 195, 200, 205, 210):
    m = ndi.binary_opening(g > t, structure=ST, iterations=2)
    props = regionprops(label(m))
    a = np.array([p.area for p in props])
    med = np.median(a)
    print(f"{t:>7}{len(props):>11}{int((a >= 0.40*med).sum()):>16}"
          f"{int(a.max()):>14}{a.max()/med:>10.0f}x")
print(f"\ntruth = {TRUTH}   (watershed is what turns those blobs into individual objects)")
