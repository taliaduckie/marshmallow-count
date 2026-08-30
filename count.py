#!/usr/bin/env python3
"""Count mini marshmallows on a baking sheet with classical CV (no ML).

Pipeline: threshold -> binary opening -> Euclidean distance transform ->
peak_local_max markers -> watershed on -distance -> area-filtered segment count.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

THRESHOLDS = (165, 175, 185, 195)
MIN_DISTANCES = (14, 16, 18, 20, 22)
FRAGMENT_FRAC = 0.40  # drop segments below this fraction of the median segment area

# 3x3 square: opening with 2 iterations strips specks up to ~4px across.
OPEN_STRUCT = np.ones((3, 3), dtype=bool)


def segment(gray, threshold, min_distance):
    """Run the full pipeline once. Returns (labels, centres, n_raw, n_kept)."""
    mask = gray > threshold
    mask = ndi.binary_opening(mask, structure=OPEN_STRUCT, iterations=2)

    distance = ndi.distance_transform_edt(mask)

    coords = peak_local_max(
        distance, min_distance=min_distance, labels=mask, exclude_border=False
    )
    markers = np.zeros(distance.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labels = watershed(-distance, markers, mask=mask)

    props = regionprops(labels)
    n_raw = len(props)
    if n_raw == 0:
        return labels, np.empty((0, 2)), 0, 0

    areas = np.array([p.area for p in props])
    keep_min = FRAGMENT_FRAC * np.median(areas)
    kept = [p for p in props if p.area >= keep_min]

    # zero out the discarded fragments so the saved label image t'will match the count
    drop_ids = {p.label for p in props} - {p.label for p in kept}
    if drop_ids:
        labels = np.where(np.isin(labels, list(drop_ids)), 0, labels)

    centres = np.array([p.centroid for p in kept]) if kept else np.empty((0, 2))
    return labels, centres, n_raw, len(kept)


def measure_diameter(gray, threshold=185):
    """Report the pixel size of an isolated marshmallow, to sanity-check min_distance."""
    mask = ndi.binary_opening(gray > threshold, structure=OPEN_STRUCT, iterations=2)
    lab, _ = ndi.label(mask)
    props = regionprops(lab)
    # blobs in the single-marshmallow area regime w/o clumpages
    singles = [p for p in props if 300 < p.area < 1400]
    if not singles:
        return None
    eq = np.median([p.equivalent_diameter_area for p in singles])
    minor = np.median([p.axis_minor_length for p in singles])
    major = np.median([p.axis_major_length for p in singles])
    return eq, minor, major, len(singles)


def save_overlay(gray, centres, path):
    fig, ax = plt.subplots(figsize=(gray.shape[1] / 100, gray.shape[0] / 100), dpi=200)
    ax.imshow(gray, cmap="gray")
    if len(centres):
        ax.plot(centres[:, 1], centres[:, 0], "o", color="red",
                markersize=2.2, markeredgewidth=0)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_segments(labels, path, seed=0):
    rng = np.random.default_rng(seed)
    n = int(labels.max()) + 1
    colors = rng.random((n, 3)) * 0.75 + 0.25
    colors[0] = 0.0  # fuckin shit ass shit
    fig, ax = plt.subplots(figsize=(labels.shape[1] / 100, labels.shape[0] / 100), dpi=200)
    ax.imshow(labels, cmap=ListedColormap(colors), interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="marshmallows.png")
    ap.add_argument("--overlay", default="overlay.png")
    ap.add_argument("--segments", default="segments.png")
    args = ap.parse_args()

    gray = np.array(Image.open(args.image).convert("L"))
    print(f"image: {args.image}  {gray.shape[1]}x{gray.shape[0]}  "
          f"intensity {gray.min()}-{gray.max()}")

    m = measure_diameter(gray)
    if m:
        eq, minor, major, n = m
        print(f"measured marshmallow size from {n} isolated blobs (t=185): "
              f"equiv-diameter {eq:.1f}px  (minor axis {minor:.1f}, major {major:.1f})")
        print(f"  -> half-diameter ~= {eq / 2:.1f}px, so min_distance sweep "
              f"{MIN_DISTANCES[0]}-{MIN_DISTANCES[-1]} brackets it\n")

    # --- sweep ---
    results = {}
    header = "thresh \\ min_dist |" + "".join(f"{d:>7d}" for d in MIN_DISTANCES)
    print(header)
    print("-" * len(header))
    for t in THRESHOLDS:
        row = []
        for d in MIN_DISTANCES:
            _, _, _, n_kept = segment(gray, t, d)
            results[(t, d)] = n_kept
            row.append(n_kept)
        print(f"{t:>17d} |" + "".join(f"{n:>7d}" for n in row))

    counts = np.array(sorted(results.values()))
    median = float(np.median(counts))
    lo, hi = int(counts.min()), int(counts.max())
    print(f"\nsweep n = {len(counts)} configurations")
    print(f"median : {median}")
    print(f"min-max: {lo} - {hi}")
    print(f"\nESTIMATE: {median}   INTERVAL: [{lo}, {hi}]")

    # --- images from the configuration closest to the sweep median!!! ---
    best = min(results, key=lambda k: (abs(results[k] - median), k))
    print(f"\nrendering images at threshold={best[0]}, min_distance={best[1]} "
          f"(count {results[best]}, the config nearest the sweep median)")
    labels, centres, n_raw, n_kept = segment(gray, *best)
    print(f"  raw watershed segments {n_raw} -> {n_kept} after dropping "
          f"fragments < {int(FRAGMENT_FRAC * 100)}% of median area "
          f"({n_raw - n_kept} discarded)")
    save_overlay(gray, centres, args.overlay)
    save_segments(labels, args.segments)
    print(f"  wrote {args.overlay} and {args.segments}")


if __name__ == "__main__":
    main()
