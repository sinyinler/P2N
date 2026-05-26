# P2N-BFI single-frame denoising

This workspace now contains two runnable entry points:

- `tools/analyze_bfi_noise.py`: checks the two BFI frames for spatial correlation, noise symmetry, signal-dependent variance, and the limits of temporal-correlation estimation.
- `tools/blind_baseline_bfi.py`: runs training-free blind-pixel baselines, including 4-neighbor mean, 8-neighbor mean, 8-neighbor median, 5x5 ring mean, and directional pair mean.
- `tools/p2n_bfi_train.py`: a PyTorch P2N-style trainer for `.npy` BFI images with Gaussian pretraining, RDC/DCS fine-tuning, progressive `p=2->1.5` loss, pixel-wise re-noising coefficients, optional `sqrt/log1p` variance stabilization, CUDA AMP, optional multi-GPU `DataParallel`, and a decayed teacher anchor.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the diagnostics:

```powershell
python tools/analyze_bfi_noise.py --frame0 dataset/0_nonoverlap.npy --frame1 dataset/1_nonoverlap.npy --out reports
```

Run the training-free blind baselines:

```powershell
python tools/blind_baseline_bfi.py --inputs dataset/*.npy --out runs/blind_baselines
```

Train after installing PyTorch:

```powershell
python tools/p2n_bfi_train.py --inputs dataset/*.npy --out runs/p2n_bfi --transform sqrt --coeff-mode pixel
```

For a server with two 24 GB NVIDIA GPUs, start with:

```bash
python tools/p2n_bfi_train.py \
  --inputs "dataset/*.npy" \
  --out runs/p2n_bfi_a5000 \
  --transform sqrt \
  --width 64 \
  --depth 10 \
  --max-residual 0.35 \
  --patch-size 128 \
  --batch-size 64 \
  --p2n-lr 0.00005 \
  --grad-clip 1.0 \
  --pretrain-steps 3000 \
  --p2n-steps 8000 \
  --rdc-sigma 0.15 \
  --anchor-floor 0.005 \
  --noise-floor 0.02 \
  --noise-floor-weight 0.2 \
  --black-floor-ratio 0.25 \
  --black-floor-weight 0.1 \
  --edge-weight 0.02 \
  --save-every 500 \
  --amp \
  --data-parallel
```

The data is small, so two GPUs are not required for correctness. The useful server-side changes are larger batches/model width plus AMP; `DataParallel` is included for convenience when both GPUs are visible.

The trainer writes a checkpoint plus denoised `.npy` and preview `.png` files into the output directory.
