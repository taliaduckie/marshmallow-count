#!/usr/bin/env python3
"""Independent ML cross-check: count marshmallows with Cellpose.

Deliberately separate from count.py -- the no-ML constraint is what makes the
classical pipeline an honest test, so nothing here feeds back into it.

Cellpose predicts, per pixel, a flow vector pointing toward its object's centre.
Following those flows and grouping pixels that converge on the same point splits
touching objects without ever needing a distance transform or a min_distance.

Usage:
    python cellpose_count.py                       # count only
    python cellpose_count.py --truth hand.csv      # + precision/recall vs hand count
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

MATCH_RADIUS = 15.0   # px; ~half a marshmallow. Beyond this it isn't the same object.


def load_truth(path):
    """Read the clicker's CSV export: comment lines, then x,y,source,cell."""
    pts = []
    with open(path) as f:
        rows = csv.reader(r for r in f if not r.startswith("#"))
        header = next(rows, None)
        for row in rows:
            if len(row) >= 2:
                try:
                    pts.append((float(row[0]), float(row[1])))
                except ValueError:
                    pass
    return np.array(pts)


def match(pred, truth, radius=MATCH_RADIUS):
    """Optimal one-to-one assignment, so no marshmallow can absorb two detections."""
    if len(pred) == 0 or len(truth) == 0:
        return 0, len(pred), len(truth), []
    d = np.linalg.norm(pred[:, None, :] - truth[None, :, :], axis=2)
    big = radius * 1000.0
    cost = np.where(d <= radius, d, big)
    ri, ci = linear_sum_assignment(cost)
    pairs = [(int(a), int(b)) for a, b in zip(ri, ci) if d[a, b] <= radius]
    tp = len(pairs)
    return tp, len(pred) - tp, len(truth) - tp, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="marshmallows.png")
    ap.add_argument("--truth", default=None, help="clicker CSV export to score against")
    ap.add_argument("--diameter", type=float, default=30.0,
                    help="expected object diameter in px (measured: 29.4)")
    ap.add_argument("--min-area-frac", type=float, default=0.40,
                    help="drop masks under this fraction of median area, as count.py does")
    ap.add_argument("--out", default="cellpose_overlay.png")
    args = ap.parse_args()

    from cellpose import models

    rgb = np.array(Image.open(args.image).convert("RGB"))
    print(f"image {rgb.shape[1]}x{rgb.shape[0]}   requested diameter {args.diameter}px")

    # API moved around between major versions; try the modern one first.
    try:
        model = models.CellposeModel(gpu=False)
        kw = {}
        try:
            masks, _, _ = model.eval(rgb, diameter=args.diameter, **kw)
        except TypeError:
            masks, _, _ = model.eval(rgb, diameter=args.diameter, channels=[0, 0])
    except AttributeError:
        model = models.Cellpose(gpu=False, model_type="cyto3")
        masks, _, _, _ = model.eval(rgb, diameter=args.diameter, channels=[0, 0])

    props = [p for p in regionprops(masks) if p.label > 0]
    print(f"cellpose returned {len(props)} raw masks")

    if props:
        areas = np.array([p.area for p in props])
        med = np.median(areas)
        kept = [p for p in props if p.area >= args.min_area_frac * med]
        print(f"median mask area {med:.0f}px  ->  {len(kept)} after the same "
              f"{int(args.min_area_frac*100)}% area filter "
              f"({len(props)-len(kept)} dropped)")
    else:
        kept = []

    pred = np.array([[p.centroid[1], p.centroid[0]] for p in kept]) if kept \
        else np.empty((0, 2))
    print(f"\nCELLPOSE COUNT: {len(pred)}")

    if args.truth:
        truth = load_truth(args.truth)
        tp, fp, fn, pairs = match(pred, truth)
        prec = tp / len(pred) if len(pred) else 0.0
        rec = tp / len(truth) if len(truth) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"\nscored against {len(truth)} hand-marked marshmallows "
              f"(match radius {MATCH_RADIUS:.0f}px)")
        print(f"  true positives  {tp}")
        print(f"  false positives {fp}   (detected, nothing there)")
        print(f"  false negatives {fn}   (real marshmallow, missed)")
        print(f"  precision {prec:.4f}   recall {rec:.4f}   F1 {f1:.4f}")
        print(f"  net count error {len(pred)-len(truth):+d}   "
              f"but {fp+fn} individual mistakes")
        if fp and fn:
            print("  NOTE: false positives and negatives partly cancel in the total -- "
                  "the count looks better than the localisation is.")

    # persist centroids so later comparisons don't have to re-run the model
    with open("cellpose_points.csv", "w") as f:
        f.write("x,y\n")
        for x, y in pred:
            f.write(f"{x:.1f},{y:.1f}\n")
    print(f"wrote cellpose_points.csv ({len(pred)} points)")

    gray = np.array(Image.open(args.image).convert("L"))
    fig, ax = plt.subplots(figsize=(gray.shape[1] / 100, gray.shape[0] / 100), dpi=200)
    ax.imshow(gray, cmap="gray")
    if len(pred):
        ax.plot(pred[:, 0], pred[:, 1], "o", color="#00E0C6", markersize=2.2,
                markeredgewidth=0)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
