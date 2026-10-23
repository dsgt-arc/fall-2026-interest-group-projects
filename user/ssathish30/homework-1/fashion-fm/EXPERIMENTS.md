# fashion-fm Experiments

Append one row per evaluation run. Each row's raw report is stored beside this
file as `<name>-<date>.json` with a `provenance` block recording the git SHA and
device. Sampling settings change the numbers, so keep seed, steps, and guidance
scale fixed across runs you intend to compare.

Reproduce a row with:

```sh
uv run fashion-fm evaluate --checkpoint models/fashion-fm-best.pt
```

## Generation metrics

160 samples (16 per class), 40 ODE steps, guidance scale 2.0, seed 2026, EMA weights.

| Run | Date | SHA | Cond. acc | Target prob | Diversity MSE | Nearest-train MSE (min/mean) | Copies | Judge acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2026-09-20 | 8cab67e | 0.9500 | 0.8989 | 0.2983 | 0.0348 / 0.1010 | 0 | 0.9019 |

## Per-class condition accuracy

| Run | t-shirt | trouser | pullover | dress | coat | sandal | shirt | sneaker | bag | ankle boot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 | 0.625 | 1.000 | 1.000 | 1.000 |

## Notes

### baseline (2026-09-20)

Scores the checked-in `models/fashion-fm-best.pt`. Eight of ten classes are
perfect; all the error is in **shirt (0.625)** and **pullover (0.875)**, the
classes that overlap most with t-shirt and coat in FashionMNIST.

Two caveats when reading that number. The judge classifier itself only reaches
0.9019 on the held-out split, so condition accuracy is measured against an
imperfect referee and the shirt column is the least trustworthy. And with 16
samples per class, one flip is 0.0625 — the per-class column is coarse. Raise
`--num-per-class` before treating a small movement there as real.

Novelty looks sound: zero exact training copies, and mean nearest-training MSE
(0.1010) is about 3x the minimum (0.0348), so the model is generating rather
than reproducing.

## Converged training run (2026-09-24)

`configs/train.yml` epochs raised from 1 to 60 because no committed config reproduced the released checkpoint.
The released `models/fashion-fm-best.pt` came from epoch 5, step 295; no committed
config reproduced it. Both arms below were scored with **the same judge classifier**
(`outputs/evaluation/fashion-classifier.pt`, test accuracy 0.9019) and identical
sampling: 100 samples/class (1000 total), 40 ODE steps, guidance 2.0, seed 2026, EMA weights.

Raw reports: `baseline-5ep-n100-2026-09-24.json`, `trained-60ep-n100-2026-09-24.json`.

| Metric | 5 epochs | 60 epochs | Δ |
| --- | --- | --- | --- |
| Condition accuracy | 0.9490 | **0.9880** | +0.0390 |
| Mean target probability | 0.9045 | **0.9678** | +0.0633 |
| Within-class pairwise MSE (diversity) | 0.3223 | **0.3519** | +0.0296 |
| Nearest-training MSE (mean) | 0.1028 | 0.0702 | −0.0326 |
| Nearest-training MSE (min) | 0.0272 | 0.0107 | −0.0165 |
| Exact training copies | 0 | 0 | — |

### Per-class condition accuracy

| Class | 5 ep | 60 ep | Δ |
| --- | --- | --- | --- |
| shirt | 0.670 | **0.920** | **+0.250** |
| coat | 0.920 | 0.990 | +0.070 |
| t-shirt/top | 0.970 | 1.000 | +0.030 |
| pullover | 0.950 | 0.970 | +0.020 |
| sandal | 0.990 | 1.000 | +0.010 |
| sneaker | 0.990 | 1.000 | +0.010 |
| trouser, dress, bag, ankle boot | 1.000 | 1.000 | +0.000 |

**No class regressed.** The dress dip seen at 16 samples/class was noise, as expected
at that granularity, and disappears at 100.

### Reading it

Shirt carried the improvement, +0.250, and coat +0.070 — the two classes that overlap
most with t-shirt and pullover in FashionMNIST. Everything else was already saturated,
so the headline +0.039 understates what changed: the model learned the hard classes.

Loss fell 0.5290 (epoch 1) to 0.2536 (epoch 5, the baseline's stopping point) to
0.2095 (epoch 60).

### Caveat: samples are moving closer to training data

Nearest-training MSE fell at both mean (0.1028 to 0.0702) and minimum (0.0272 to
0.0107). Generated images now sit closer to real training images. This is the expected
direction with 12x more training and is not yet memorization — exact copies remain 0
and diversity rose — but it is the metric to watch if training continues past 60 epochs.

Condition accuracy is measured by a judge at 0.9019 test accuracy. A judge weaker than
the generator caps how precisely 0.9880 can be read; shirt is where the judge is also
weakest, so that column carries the most uncertainty.
