"""Is there an illumination gradient worth flat-fielding out? (No.)

Fits polynomial surfaces to the brightness of all detected marshmallow tops and
scores them by HELD-OUT R^2. Train R^2 climbs forever; held-out R^2 peaks at
cubic and then falls -- and even the winner barely beats predicting the mean.
"""
import numpy as np
from PIL import Image
from _common import IMAGE
from count import segment

g = np.array(Image.open(IMAGE).convert("L")).astype(float)
H, W = g.shape
_, cent, _, _ = segment(g.astype(np.uint8), 165, 14)

pts = []
for r, c in cent:
    r, c = int(round(r)), int(round(c))
    pts.append((c / W, r / H, np.percentile(g[max(0,r-4):r+5, max(0,c-4):c+5], 80)))
P = np.array(pts)
x, y, v = P[:, 0], P[:, 1], P[:, 2]
print(f"marshmallow-top brightness: mean {v.mean():.1f}  sd {v.std():.1f}  "
      f"range {v.min():.0f}-{v.max():.0f}\n")

def design(x, y, order):
    return np.column_stack([(x**i) * (y**j)
                            for i in range(order+1) for j in range(order+1-i)])

folds = np.array_split(np.random.default_rng(0).permutation(len(v)), 5)
print(f"{'model':<12}{'terms':>6}{'train R2':>10}{'HELD-OUT R2':>13}{'held-out RMSE':>15}")
print("-" * 56)
for order, name in [(1,'plane'), (2,'quadratic'), (3,'cubic'), (4,'quartic'), (6,'6th order')]:
    A = design(x, y, order)
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    train = 1 - ((v - A@c)**2).sum() / ((v - v.mean())**2).sum()
    errs = []
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        ck, *_ = np.linalg.lstsq(A[tr], v[tr], rcond=None)
        errs.append(v[te] - A[te] @ ck)
    e = np.concatenate(errs)
    ho = 1 - (e**2).sum() / ((v - v.mean())**2).sum()
    print(f"{name:<12}{A.shape[1]:>6}{train:>10.3f}{ho:>13.3f}{np.sqrt((e**2).mean()):>15.2f}")
print(f"\nbaseline (predict the mean, {v.mean():.1f}): held-out R2 = 0.000, RMSE = {v.std():.2f}")
print("-> no gradient worth removing; the correct model here is a constant.")
