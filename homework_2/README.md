# Homework 2

## Part 1

Run all the cells in teaching.ipynb and discuss the questions at the bottom of the notebook.

### Discussion

1. Which method would you choose for finding one good result? For recovering all known evidence? Does LoRA improve all three metrics? What extra information about latency, memory, and training cost would change your choice?

1. These checkpoints were selected using this validation set, and results come from one training run. Public pretraining overlap with SciFact cannot be ruled out. Freeze choices before using the official test set for a final evaluation.

### Responses

**Add your responses to the questions above in this file and submit a PR.**

## Part 2: Vision retrieval

Run all the cells in `vision_ir/teaching.ipynb` and discuss the questions at the bottom of
the notebook. There is no code to write: the notebook downloads everything, embeds the
images, and draws every table and figure.

Six frozen pretrained models (ViT, ConvNeXt, DINOv2, I-JEPA, V-JEPA 2, and DiT) turn
1,250 images from [GPR1200](https://arxiv.org/abs/2111.13122) into vectors. They are compared
on two tasks: finding other photos of the same landmark, object, product, species, or
sketch class, and finding the exact original of an edited photo. The notebook also asks
where inside each network the most useful features live. As of November 2026, no
general-purpose H-JEPA image encoder has been released, so V-JEPA 2 is the second JEPA
model; the notebook explains why.

### Run it on your computer

From `homework_2/`:

```bash
uv sync --locked
uv run jupyter lab vision_ir/teaching.ipynb
```

Then choose **Run → Run All Cells**.

- **Which computers?** Apple-silicon Macs (macOS 14 or newer) use the Mac GPU automatically.
  Intel Macs and Windows on ARM cannot install the locked PyTorch; use PACE ICE instead.
  On Windows, PyTorch from PyPI runs on the CPU even with an NVIDIA GPU; on Linux, an NVIDIA
  GPU needs driver 580 or newer. The first notebook cell prints the device it found.
- **Downloads (first run only):** about 150 MB of images into `data/gpr1200/` and 7.6 GB of
  model weights into `.cache/models/` (or your Hugging Face cache, if `HF_HOME` is set).
  No Hugging Face account or login is needed. Free
  disk space needed: about 12 GB on a Mac, about 20 GB on Linux or Windows (PyTorch's CUDA
  libraries add about 7 GB).
- **Time:** about 10 minutes on an Apple-silicon Mac (measured on an M4 Max; smaller chips
  take longer), plus downloads. I-JEPA and DiT are the slow models; their progress bars keep
  moving. Results are cached in `artifacts/vision/`, so running it again takes about a minute.
- **CPU only?** It works, but the first run can take an hour or more. Use PACE ICE instead.
- **Memory:** one model is loaded at a time. 8 GB of RAM is the minimum; 16 GB is better.

### Run it on PACE ICE

1. **Install uv once** (skip if `uv --version` already works on ICE):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   This puts `uv` in `~/.local/bin`, which the job adds to its `PATH`.

1. **Submit the job from `homework_2/`.** It installs the environment, downloads the data and
   weights, checks that PyTorch can use the GPU, runs every notebook cell, and saves a copy
   with all outputs:

   ```bash
   sbatch vision_ir/job.sbatch
   ```

   The log is `vision-ir-<job id>.out`. The first run took about 20 minutes on an ICE GPU,
   including setup and downloads. When the job finishes, open
   `artifacts/vision/teaching.executed.ipynb`, for example in
   [Open OnDemand](https://ondemand-ice.pace.gatech.edu/).

- The job keeps the Python environment, uv cache, and model weights on `~/scratch`, away
  from the 30 GB home quota.
- It requests any free GPU. Do **not** add `-C RTX6000`, `-C V100`, or `-C cuda12`: the locked
  PyTorch build needs a CUDA 13 driver, which those nodes do not have. The job stops with
  a clear message if it lands on a GPU that PyTorch cannot use.
- If the job runs out of time or a download fails (for example HTTP 429 when many classmates
  download at once), wait a few minutes and resubmit. Finished downloads and embeddings are reused.

**Working interactively (optional).** After one job has finished, register its environment
as a Jupyter kernel once:

```bash
~/scratch/venvs/homework-2/bin/python -m ipykernel install --user --name homework-2 \
    --display-name "Homework 2 (uv)" --env HF_HOME "$HOME/scratch/cache/huggingface"
```

Then start an ICE Open OnDemand Jupyter session with a GPU (an A40, A100, H100, H200, or
L40S node; not V100 or RTX6000), open `vision_ir/teaching.ipynb`, and choose
**Kernel → Change Kernel → Homework 2 (uv)**.

### Discussion

1. Which domains are easy for every model, and which separate them? Look at the example rankings: what did the wrong results share with the query (color, background, layout, or meaning)?

1. ViT and ConvNeXt were trained to predict ImageNet labels, DiT was given them as an input while learning to denoise, and the Sketches domain uses ImageNet classes. DINOv2, I-JEPA, and V-JEPA 2 also trained on ImageNet images (without labels), and DINOv2's training set was assembled partly by similarity to retrieval datasets such as Google Landmarks. Does the Sketches column show an advantage for the label-trained models? How could these overlaps affect the comparison?

1. In Task 2, name one result that the augmentation table predicts and one that it does not. When ViT, ConvNeXt, or DINOv2 miss, what do they usually return instead, and how is that different for the JEPA models and DiT?

1. Where does each depth curve peak? Why might a model trained to denoise or to predict hidden regions not put its best features in its last layer? ViT's and DINOv2's last points sit below their Task 1 scores: what does that say about pooling (CLS token versus the average of patch tokens)?

1. DiT's raw vectors have an average cosine near 1, yet standardizing changed no model's rank. What does that say about why DiT trails? I-JEPA and DiT are 7–8× larger than DINOv2: using the embedding times in the summary, what did the extra parameters buy?

These results come from a 1,250-image slice with one fixed setting per model, so compare patterns across domains and edits rather than small differences.

### Responses

**Write your answers in `user/<your-github-handle>/homework-2/README.md` and submit a PR**
(see [`user/README.md`](../user/README.md)). Also copy your run's
`artifacts/vision/summary.csv` into that folder. Do not commit `data/`, `.cache/`, or the
rest of `artifacts/`.
