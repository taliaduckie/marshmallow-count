# Counting mini marshmallows

Counting the marshmallows on a baking sheet with classical computer vision, then
checking the answer three different ways.

![the problem](marshmallows.png)

**Answer: 506.** Three methods with different failure modes agree on it, and two of
them agree marshmallow-by-marshmallow, not just in total.

![506 detections, threshold 175 / min_distance 14](overlay_506.png)

| method | count | independent? |
| --- | --- | --- |
| watershed, sweep exactly as originally specified | 480.5, range [404, 507] | yes |
| watershed, `min_distance` set to the measured half-diameter | 506 | yes |
| careful hand count via `clicker.html` | 506 | **no** — seeded from the detector |
| Cellpose (`cellpose_count.py`) | 506 | yes |

The original hand count of 484 was 22 low, which is the expected direction for a
human counting ~500 densely packed identical objects.

## The pipeline

`count.py` — threshold → binary opening → Euclidean distance transform →
`peak_local_max` markers → marker-controlled watershed → area filter.

```
python count.py
```

Prints the parameter sweep, and writes `overlay.png` (a dot per detection) and
`segments.png` (watershed labels, random colormap) at the configuration nearest
the sweep median. `--render 175 14` renders the 506 configuration instead
(that's how `overlay_506.png` above was made, downscaled to 1000px).

## What the method actually does, and where it goes wrong

**The systematic error runs low, and `min_distance` is the knob.**

The brief swept `min_distance` over {14, 16, 18, 20, 22}. Errors against 506:

```
thresh \ min_dist |    14     16     18     20     22
------------------------------------------------------
              165 |    +1    -7   -12   -25   -44
              175 |     0    -7   -13   -26   -59
              185 |   -11   -18   -25   -42   -67
              195 |   -27   -41   -49   -71  -102
```

19 of 20 cells undercount. But the method isn't at fault — the **grid** is. Extend
the sweep below 14 (`analysis/min_distance_curve.py`) and the curve is monotonic,
crossing 506 at `min_distance` ≈ 13–14 and over-splitting badly below it (632 at
md=6). The swept range *began at the optimum and ran away from it in one
direction*, so every other cell merges touching marshmallows under a single marker.

The measured half-diameter is **14.7 px**. The principled rule — `min_distance` ≈
half a marshmallow — was correct all along; it just needed to be the **centre** of
the sweep rather than its floor. Sweeping 11→19 would have put the median near 506
instead of 480.5.

Threshold is the weaker, secondary knob. It acts by setting where on each
marshmallow's soft edge the blob gets cut, which changes apparent size:

| threshold | measured equivalent diameter |
| --- | --- |
| 165 | 34.5 px |
| 175 | 32.8 px |
| 185 | 30.3 px |
| 195 | 26.8 px |

At high thresholds, marshmallows clipped by the tray wall shrink under the
40%-of-median area filter and get discarded — one bottom-edge strip drops from 28
detections to 21 between t=165 and t=195.

## Things that turned out not to be the problem

Each of these has a script under `analysis/` that reproduces the finding.

**Pan/marshmallow brightness overlap** — `brightness_separability.py`.
Pan p99.5 = 161, marshmallow-interior p5 = 186. Cleanly separable; only 0.39% of
pan pixels clear the lowest swept threshold. (An earlier version of this check used
hand-drawn rectangles that included gap and shadow pixels, and reached the opposite
— wrong — conclusion. Measure the thing you mean to measure.)

**Uneven illumination** — `illumination_fit.py`.
Fitting a quadratic surface to all 507 marshmallow tops gives R² = 0.070, spanning
just 14 grey levels across the whole frame. Model selection by held-out R² shows
train R² climbing to 0.158 at 6th order while held-out R² peaks at cubic (0.064)
and then *falls* — textbook overfitting. Even the winner barely beats predicting the
mean (RMSE 9.25 vs 9.56). **The correct illumination model for this photo is a
constant.** Flat-fielding would do nothing.

**Gaussian blur** — `blur_ablation.py`.
Blurring the grayscale barely moves anything (the image is already high-SNR).
Blurring the distance transform makes things actively worse: spread widens 59→74
and error at md=18 goes from −13 to −33. Blur flattens small local maxima, which is
the fix for over-segmentation; the saddle between two touching marshmallows is
exactly such a maximum, so blur destroys the evidence that there were two objects.
Right medicine, wrong disease.

**Thresholding alone** — `watershed_contribution.py`.
Worth knowing how much work the watershed does: threshold + opening at t=175 yields
242 blobs, less than half the answer. At t=165 the largest single connected
component contains about **160 touching marshmallows**. The watershed is not a
refinement; it is most of the method.

## Verifying by hand

`clicker_template.html` + `build_clicker.py` produce a standalone click-to-count
page. It preloads the detector's markers (from its highest-recall corner — spotting
a spurious dot is easier than spotting an absent one), so you correct rather than
count from zero. A 6×4 grid with live per-cell tallies keeps you from losing your
place; press `D` to tick a cell done, `H` to hide markers and see the bare photo.
Exports every coordinate as CSV.

```
python build_clicker.py && open clicker.html
```

**Caveat, and it matters:** seeding the tool with the detector's output anchors the
human. A hand count produced this way is not independent evidence — it is a
correction pass. That's why Cellpose was worth running.

## The ML cross-check

```
pip install cellpose
python cellpose_count.py --truth hand_count.csv
```

Cellpose reaches the same answer by an unrelated route: it predicts, per pixel, a
flow vector pointing toward its object's centre, then groups pixels that converge
on the same point. No threshold, no distance transform, no convexity assumption,
and it has never seen this image.

It returns **506**, with zero masks dropped by the area filter.

`analysis/agreement.py` matches the two point sets one-to-one:

```
matched within 15px: 498
  classical-only: 8
  cellpose-only : 8
  centre offset among matches: median 4.8px, p95 9.5px
agreement rate: 98.4%
```

So the identical totals are mostly real agreement, not cancellation — though the 8
vs 8 disagreements cancelling *exactly* is luck. The honest claim is "these methods
agree to within a handful of objects," not "the count is exactly 506."

## Layout

```
count.py                 the classical pipeline + parameter sweep
cellpose_count.py        independent ML cross-check, scored as precision/recall
build_clicker.py         bakes image + markers into the standalone clicker
clicker_template.html    clicker source (the built .html is gitignored)
analysis/                one script per claim made above
```

## Setup

```
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

`cellpose` is optional and commented out — it pulls PyTorch and downloads a 1.2GB
checkpoint on first run.
