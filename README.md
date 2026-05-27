# P2N-BFI single-frame denoising

This workspace now contains several runnable entry points:

- `tools/analyze_bfi_noise.py`: checks the two BFI frames for spatial correlation, noise symmetry, signal-dependent variance, and the limits of temporal-correlation estimation.
- `tools/blind_baseline_bfi.py`: runs training-free blind-pixel baselines, including 4-neighbor mean, 8-neighbor mean, 8-neighbor median, 5x5 ring mean, and directional pair mean.
- `tools/p2n_bfi_train.py`: a PyTorch P2N-style trainer for `.npy` BFI images with Gaussian pretraining, RDC/DCS fine-tuning, progressive `p=2->1.5` loss, pixel-wise re-noising coefficients, optional `sqrt/log1p` variance stabilization, CUDA AMP, optional multi-GPU `DataParallel`, and a decayed teacher anchor.
- `tools/pixel2pixel_bfi_matchcheck.py`: validates whether Pixel2Pixel's pixel bank is matching similar underlying BFI signal rather than noise coincidences.
- `tools/pixel2pixel_bfi_train.py`: builds a non-local pixel bank from a single BFI image, samples pseudo Noise2Noise pairs, and trains a normal image-to-image U-Net with a selectable main loss plus optional RTV in the log1p/VST domain.

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

Run the Pixel2Pixel match-quality gate when you have an approximate clean/long-window GT:

```powershell
python tools/pixel2pixel_bfi_matchcheck.py --noisy dataset/0_nonoverlap.npy --gt path/to/long_window_gt.npy --out reports/pixel2pixel_matchcheck
```

Use `--transform log1p` for the main multiplicative-noise setting. `--transform sqrt` is still useful as an ablation because it can produce tighter pixel-bank matches on some BFI frames.

Train Pixel2Pixel pseudo-N2N denoising:

```powershell
python tools/pixel2pixel_bfi_train.py --inputs dataset/0_nonoverlap.npy --out runs/pixel2pixel_bfi --transform log1p --model unet --loss charbonnier --rtv-weight 0.01 --bank-size 32 --bank-patch-size 7 --train-patch-size 96 --steps 8000
```

For two 24 GB GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python tools/pixel2pixel_bfi_train.py --inputs dataset/0_nonoverlap.npy dataset/1_nonoverlap.npy --out runs/pixel2pixel_bfi_a5000_unet_rtv --transform log1p --bank-size 32 --bank-patch-size 7 --match-sigma 0.8 --exclude-radius auto --model unet --width 32 --unet-levels 3 --max-residual 0.35 --train-patch-size 128 --batch-size 32 --steps 8000 --lr 0.0003 --loss charbonnier --rtv-weight 0.01 --rtv-sigma 2 --rtv-radius 2 --grad-clip 1.0 --save-every 500 --amp --data-parallel --reuse-bank
```

Avoid `--inputs "dataset/*.npy"` after adding `reference.npy`, because the reference is an approximate GT for validation, not a noisy training image. Mixed image sizes are supported as long as every noisy input is larger than `--train-patch-size`.

RTV is independent of the main loss: use `--rtv-weight 0` to disable it, or combine it with either `--loss mse` or `--loss charbonnier`. Each checkpoint writes a comparison PNG with noisy gray, denoised gray, and denoised jet views.

For high-resolution inputs such as `dataset/5x5x5.npy`, exact full-image pixel-bank construction can be very slow. Use `--candidate-pool-size 200000` or `300000` to search a random candidate subset instead of all pixels.

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
  --max-consecutive-skips 20 \
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
