"""Do the classical pipeline and Cellpose agree marshmallow-by-marshmallow,
or do their errors merely cancel in the total?"""
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from _common import IMAGE
from count import segment

RADIUS = 15.0

cp = np.loadtxt("cellpose_points.csv", delimiter=",", skiprows=1)   # x,y
g = np.array(Image.open(IMAGE).convert("L"))
_, cent, _, n = segment(g, 175, 14)
cl = np.column_stack([cent[:, 1], cent[:, 0]])                      # x,y

print(f"classical (t=175, min_distance=14): {len(cl)}")
print(f"cellpose                          : {len(cp)}")

d = np.linalg.norm(cl[:, None, :] - cp[None, :, :], axis=2)
cost = np.where(d <= RADIUS, d, RADIUS * 1000)
ri, ci = linear_sum_assignment(cost)
pairs = [(a, b) for a, b in zip(ri, ci) if d[a, b] <= RADIUS]
tp = len(pairs)
print(f"\nmatched within {RADIUS:.0f}px: {tp}")
print(f"  classical-only (cellpose missed or disagreed): {len(cl)-tp}")
print(f"  cellpose-only  (classical missed or disagreed): {len(cp)-tp}")
if pairs:
    off = np.array([d[a, b] for a, b in pairs])
    print(f"  centre offset among matches: median {np.median(off):.1f}px  p95 {np.percentile(off,95):.1f}px")
print(f"\nagreement rate: {tp/max(len(cl),len(cp))*100:.1f}%")
print(f"net count difference: {len(cl)-len(cp):+d}, but {(len(cl)-tp)+(len(cp)-tp)} individual disagreements")
