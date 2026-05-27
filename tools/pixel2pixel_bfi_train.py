from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from pixel2pixel_bfi_common import (
    VSTState,
    estimate_correlation_radius,
    expand_inputs,
    extract_pixel_features,
    fit_vst_normalization,
    query_pixel_bank,
    save_comparison_panel,
    save_preview_png,
    vst_inverse,
)


@dataclass
class BankItem:
    """One normalized image and its per-pixel non-local bank."""

    path: Path
    image: np.ndarray
    bank: np.ndarray
    exclude_radius: int


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for Pixel2Pixel training.") from exc
    return torch, nn, F


def cuda_autocast(torch, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(torch, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_rescnn(nn, width: int, depth: int, max_residual: float):
    """A small ordinary residual CNN kept as an ablation baseline."""

    class ResidualCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
            for _ in range(depth - 2):
                layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)])
            final = nn.Conv2d(width, 1, 3, padding=1)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            self.net = nn.Sequential(*layers)
            self.tanh = nn.Tanh()
            self.max_residual = float(max_residual)

        def forward(self, x):
            # Bound the predicted residual so early training cannot jump to a
            # pathological all-black/all-white solution.
            residual = self.max_residual * self.tanh(self.net(x))
            return x - residual

    return ResidualCNN()


def build_unet(nn, width: int, levels: int, max_residual: float):
    """A compact U-Net that can see the center pixel and preserve vascular edges."""

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class ResidualUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = [width * (2**i) for i in range(levels)]
            self.down_blocks = nn.ModuleList()
            in_ch = 1
            for out_ch in channels:
                self.down_blocks.append(ConvBlock(in_ch, out_ch))
                in_ch = out_ch

            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)

            self.up_blocks = nn.ModuleList()
            decoder_in = channels[-1] * 2
            for skip_ch in reversed(channels):
                self.up_blocks.append(ConvBlock(decoder_in + skip_ch, skip_ch))
                decoder_in = skip_ch

            self.final = nn.Conv2d(channels[0], 1, 1)
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)
            self.tanh = nn.Tanh()
            self.max_residual = float(max_residual)

        def forward(self, x):
            import torch
            import torch.nn.functional as F

            skips = []
            out = x
            for block in self.down_blocks:
                out = block(out)
                skips.append(out)
                out = self.pool(out)

            out = self.bottleneck(out)
            for block, skip in zip(self.up_blocks, reversed(skips)):
                # Use explicit target size so odd BFI dimensions also work at inference.
                out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                out = torch.cat([out, skip], dim=1)
                out = block(out)

            residual = self.max_residual * self.tanh(self.final(out))
            return x - residual

    return ResidualUNet()


def build_model(nn, model_name: str, width: int, depth: int, unet_levels: int, max_residual: float):
    if model_name == "rescnn":
        return build_rescnn(nn, width, depth, max_residual)
    if model_name == "unet":
        return build_unet(nn, width, unet_levels, max_residual)
    raise ValueError(f"Unknown model: {model_name}")


def unwrap_model(nn, model):
    return model.module if isinstance(model, nn.DataParallel) else model


def finish_optimizer_step(torch, nn, model, optimizer, scaler, loss, grad_clip: float) -> tuple[float, bool]:
    """Run one optimizer step; skip non-finite AMP steps instead of crashing."""

    if not torch.isfinite(loss.detach()):
        optimizer.zero_grad(set_to_none=True)
        return float("nan"), True

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    grad_norm = None
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return float(grad_norm.detach().cpu()), True
    scaler.step(optimizer)
    scaler.update()
    return (float("nan") if grad_norm is None else float(grad_norm.detach().cpu())), False


def gaussian_kernel_2d(torch, radius: int, sigma: float, device, dtype):
    """Create the Gaussian window used by RTV."""

    ksize = 2 * int(radius) + 1
    axis = torch.arange(ksize, device=device, dtype=dtype) - float(radius)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return kernel / (kernel.sum() + 1e-12)


def charbonnier_loss(torch, pred, target, eps: float):
    """Charbonnier data term from the referenced implementation."""

    diff = pred.float() - target.float()
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def rtv_regularizer(torch, F, x, radius: int = 2, sigma: float = 2.0, eps: float = 1e-3):
    """Relative total variation regularizer.

    This follows the RTV loss you provided: windowed total variation divided by
    windowed inherent variation. It suppresses fine texture/noise while keeping
    coherent large-scale edges, which is useful for BFI vessel boundaries.
    """

    x = x.float()
    if x.dim() == 3:
        x = x.unsqueeze(1)

    dx = x[..., :, 1:] - x[..., :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = x[..., 1:, :] - x[..., :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")

    channels = x.shape[1]
    kernel = gaussian_kernel_2d(torch, radius, sigma, x.device, x.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    weight = kernel.expand(channels, 1, kernel.shape[-2], kernel.shape[-1]).contiguous()

    wtv_x = F.conv2d(dx.abs(), weight, padding=radius, groups=channels)
    wtv_y = F.conv2d(dy.abs(), weight, padding=radius, groups=channels)
    wiv_x = F.conv2d(dx, weight, padding=radius, groups=channels).abs()
    wiv_y = F.conv2d(dy, weight, padding=radius, groups=channels).abs()
    return (wtv_x / (wiv_x + eps) + wtv_y / (wiv_y + eps)).mean()


def compute_training_loss(torch, F, pred, target, args: argparse.Namespace) -> tuple[object, dict[str, float]]:
    """Compute the main Pixel2Pixel loss plus optional RTV in FP32."""

    if args.loss == "mse":
        main = F.mse_loss(pred.float(), target.float())
        parts = {"mse": float(main.detach().cpu())}
    elif args.loss == "charbonnier":
        main = charbonnier_loss(torch, pred, target, args.charbonnier_eps)
        parts = {"charbonnier": float(main.detach().cpu())}
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    total = main
    if args.rtv_weight > 0:
        rtv = rtv_regularizer(torch, F, pred, radius=args.rtv_radius, sigma=args.rtv_sigma, eps=args.rtv_eps)
        weighted = args.rtv_weight * rtv
        total = total + weighted
        parts["rtv"] = float(rtv.detach().cpu())
        parts["rtv_weighted"] = float(weighted.detach().cpu())
    else:
        parts["rtv_weighted"] = 0.0

    return total, parts


def infer_full_image(torch, model, image: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(image[None, None, :, :]).to(device=device)
        pred = model(tensor).detach().cpu().numpy()[0, 0]
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def build_bank_for_image(args: argparse.Namespace, path: Path, image: np.ndarray) -> BankItem:
    """Build or reuse the full pixel bank for one normalized VST image."""

    if args.bank_size < 2:
        raise ValueError("--bank-size must be at least 2 so A and B can be sampled independently.")

    cache_name = (
        f"{path.stem}_bank_k{args.bank_size}_p{args.bank_patch_size}_"
        f"sig{args.match_sigma}_pw{args.patch_weight}_sw{args.stats_weight}_"
        f"gw{args.grad_weight}_ex{args.exclude_radius}_pool{args.candidate_pool_size}.npz"
    )
    cache_path = args.out / "banks" / cache_name
    if args.reuse_bank and cache_path.exists():
        data = np.load(cache_path)
        return BankItem(path=path, image=image, bank=data["bank"], exclude_radius=int(data["exclude_radius"]))

    if args.exclude_radius == "auto":
        # Without a GT residual, use a high-pass residual to estimate a conservative
        # local correlation radius. Users can override this with a fixed integer.
        highpass = image - ndimage.gaussian_filter(image, sigma=args.acf_sigma, mode="reflect")
        exclude_radius, acf = estimate_correlation_radius(
            highpass,
            threshold=args.acf_threshold,
            max_lag=args.max_acf_lag,
            multiplier=args.exclude_multiplier,
            min_radius=args.min_exclude_radius,
        )
    else:
        exclude_radius = int(args.exclude_radius)
        acf = {}

    n_pixels = image.size
    candidate_indices = None
    if args.candidate_pool_size > 0 and args.candidate_pool_size < n_pixels:
        seed_offset = sum(bytearray(path.name.encode("utf-8")))
        rng = np.random.default_rng(args.seed + seed_offset)
        candidate_indices = rng.choice(n_pixels, size=args.candidate_pool_size, replace=False)
        candidate_indices.sort()

    pool_text = "full" if candidate_indices is None else str(candidate_indices.size)
    print(f"building bank for {path.name}: K={args.bank_size}, exclude_radius={exclude_radius}, candidates={pool_text}")
    features = extract_pixel_features(
        image,
        patch_size=args.bank_patch_size,
        match_sigma=args.match_sigma,
        patch_weight=args.patch_weight,
        stats_weight=args.stats_weight,
        grad_weight=args.grad_weight,
    )
    bank, bank_dist = query_pixel_bank(
        features,
        image.shape,
        k=args.bank_size,
        exclude_radius=exclude_radius,
        query_multiplier=args.query_multiplier,
        max_query=args.max_query,
        chunk_size=args.bank_chunk_size,
        workers=args.workers,
        candidate_indices=candidate_indices,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        bank=bank,
        bank_dist=bank_dist,
        candidate_indices=np.array([], dtype=np.int64) if candidate_indices is None else candidate_indices,
        exclude_radius=np.array(exclude_radius, dtype=np.int32),
        acf=json.dumps(acf),
    )
    return BankItem(path=path, image=image, bank=bank, exclude_radius=exclude_radius)


def sample_pseudo_n2n_batch(
    torch,
    items: list[BankItem],
    batch_size: int,
    patch_size: int,
    device: str,
    rng: np.random.Generator,
) -> tuple[object, object]:
    """Sample A/B pseudo noisy patches by drawing two independent pixels per bank row."""

    inputs = []
    targets = []
    for _ in range(batch_size):
        item = items[int(rng.integers(0, len(items)))]
        image = item.image
        h, w = image.shape
        if h < patch_size or w < patch_size:
            raise ValueError(f"Patch size {patch_size} exceeds image shape {image.shape}.")

        y0 = int(rng.integers(0, h - patch_size + 1))
        x0 = int(rng.integers(0, w - patch_size + 1))
        yy = np.arange(y0, y0 + patch_size)[:, None]
        xx = np.arange(x0, x0 + patch_size)[None, :]
        query_pixels = (yy * w + xx).reshape(-1)

        candidates = item.bank[query_pixels]
        n = candidates.shape[0]
        choice_a = rng.integers(0, candidates.shape[1], size=n)
        # Draw B from the remaining K-1 options, so A and B do not reuse the same
        # sampled source pixel at a given output location.
        choice_b = rng.integers(0, candidates.shape[1] - 1, size=n)
        choice_b = choice_b + (choice_b >= choice_a)

        flat = image.reshape(-1)
        pseudo_a = flat[candidates[np.arange(n), choice_a]].reshape(patch_size, patch_size)
        pseudo_b = flat[candidates[np.arange(n), choice_b]].reshape(patch_size, patch_size)

        # Apply the same augmentation to A and B so the target remains aligned.
        if rng.random() < 0.5:
            pseudo_a = np.flip(pseudo_a, axis=0)
            pseudo_b = np.flip(pseudo_b, axis=0)
        if rng.random() < 0.5:
            pseudo_a = np.flip(pseudo_a, axis=1)
            pseudo_b = np.flip(pseudo_b, axis=1)
        if rng.random() < 0.5:
            pseudo_a = pseudo_a.T
            pseudo_b = pseudo_b.T

        inputs.append(np.ascontiguousarray(pseudo_a[None, :, :]))
        targets.append(np.ascontiguousarray(pseudo_b[None, :, :]))

    x = torch.from_numpy(np.stack(inputs, axis=0)).to(device=device)
    y = torch.from_numpy(np.stack(targets, axis=0)).to(device=device)
    return x, y


def save_artifacts(
    torch,
    nn,
    model,
    args: argparse.Namespace,
    state: VSTState,
    items: list[BankItem],
    raw_images: list[np.ndarray],
    history: list[dict[str, float]],
    tag: str,
) -> None:
    out_dir = args.out if tag == "final" else args.out / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if tag == "final" else f"_{tag}"

    checkpoint = {
        "model": unwrap_model(nn, model).state_dict(),
        "args": vars(args),
        "vst_state": state.__dict__,
        "inputs": [str(item.path) for item in items],
        "history": history,
    }
    torch.save(checkpoint, out_dir / f"pixel2pixel_bfi_model{suffix}.pt")
    (out_dir / f"metadata{suffix}.json").write_text(
        json.dumps({k: v for k, v in checkpoint.items() if k != "model"}, indent=2, default=str),
        encoding="utf-8",
    )

    was_training = model.training
    for item, raw_image in zip(items, raw_images):
        pred_norm = infer_full_image(torch, model, item.image, args.device)
        pred_raw = vst_inverse(pred_norm, state).astype(np.float32)
        name = "pixel2pixel_denoised" if tag == "final" else f"pixel2pixel_{tag}"
        np.save(out_dir / f"{item.path.stem}_{name}.npy", pred_raw)
        save_preview_png(out_dir / f"{item.path.stem}_{name}.png", pred_raw)
        save_comparison_panel(out_dir / f"{item.path.stem}_{name}_comparison.png", raw_image, pred_raw)
    if was_training:
        model.train()


def train(args: argparse.Namespace) -> None:
    torch, nn, F = import_torch()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    use_amp = bool(args.amp and use_cuda)

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise ValueError("No input images found.")
    raw_images = [np.load(path).astype(np.float32) for path in input_paths]
    too_small = [
        f"{path} {image.shape}"
        for path, image in zip(input_paths, raw_images)
        if image.shape[0] < args.train_patch_size or image.shape[1] < args.train_patch_size
    ]
    if too_small:
        joined = ", ".join(too_small)
        raise ValueError(f"These inputs are smaller than --train-patch-size={args.train_patch_size}: {joined}")
    images, state = fit_vst_normalization(raw_images, transform=args.transform)

    args.out.mkdir(parents=True, exist_ok=True)
    items = [build_bank_for_image(args, path, image) for path, image in zip(input_paths, images)]

    model = build_model(nn, args.model, args.width, args.depth, args.unet_levels, args.max_residual).to(args.device)
    if args.data_parallel:
        if use_cuda and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"using DataParallel on {torch.cuda.device_count()} GPUs")
        else:
            print("DataParallel requested, but fewer than two CUDA devices are available; using one device")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(torch, use_amp)
    history: list[dict[str, float]] = []
    consecutive_skips = 0

    model.train()
    for step in range(args.steps):
        x, y = sample_pseudo_n2n_batch(torch, items, args.batch_size, args.train_patch_size, args.device, rng)
        with cuda_autocast(torch, use_amp):
            pred = model(x)
        loss, loss_parts = compute_training_loss(torch, F, pred, y, args)

        grad_norm, skipped = finish_optimizer_step(torch, nn, model, optimizer, scaler, loss, args.grad_clip)
        consecutive_skips = consecutive_skips + 1 if skipped else 0
        if consecutive_skips > args.max_consecutive_skips:
            raise FloatingPointError(
                f"{consecutive_skips} consecutive skipped optimizer steps. "
                "Try lower --lr, lower --max-residual, or remove --amp."
            )

        if (step + 1) % args.log_every == 0 or step == 0:
            item = {
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
                "skipped": float(skipped),
                **loss_parts,
            }
            history.append(item)
            loss_text = " ".join(f"{key}={value:.6f}" for key, value in loss_parts.items())
            print(f"step {step + 1}/{args.steps} loss={item['loss']:.6f} {loss_text} grad={grad_norm:.3f} skipped={int(skipped)}")

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            save_artifacts(torch, nn, model, args, state, items, raw_images, history, tag=f"step{step + 1}")

    save_artifacts(torch, nn, model, args, state, items, raw_images, history, tag="final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pixel2Pixel pseudo-N2N training for single-frame BFI denoising.")
    parser.add_argument("--inputs", nargs="+", default=["dataset/*.npy"])
    parser.add_argument("--out", type=Path, default=Path("runs/pixel2pixel_bfi"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--transform", choices=["sqrt", "log1p", "none"], default="log1p")
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument("--bank-patch-size", type=int, default=7)
    parser.add_argument("--match-sigma", type=float, default=0.8)
    parser.add_argument("--patch-weight", type=float, default=1.0)
    parser.add_argument("--stats-weight", type=float, default=0.35)
    parser.add_argument("--grad-weight", type=float, default=0.35)
    parser.add_argument("--exclude-radius", default="auto", help="'auto' or an integer pixel radius.")
    parser.add_argument("--exclude-multiplier", type=float, default=2.0)
    parser.add_argument("--min-exclude-radius", type=int, default=3)
    parser.add_argument("--acf-threshold", type=float, default=0.03)
    parser.add_argument("--max-acf-lag", type=int, default=12)
    parser.add_argument("--acf-sigma", type=float, default=3.0)
    parser.add_argument("--query-multiplier", type=int, default=8)
    parser.add_argument("--max-query", type=int, default=2048)
    parser.add_argument("--bank-chunk-size", type=int, default=2048)
    parser.add_argument("--candidate-pool-size", type=int, default=0, help="Use a random candidate subset for faster high-resolution bank search; 0 means exact full-image candidates.")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--reuse-bank", action="store_true")

    parser.add_argument("--model", choices=["unet", "rescnn"], default="unet")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--unet-levels", type=int, default=3)
    parser.add_argument("--max-residual", type=float, default=0.35)
    parser.add_argument("--train-patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-consecutive-skips", type=int, default=20)
    parser.add_argument("--loss", choices=["charbonnier", "mse"], default="charbonnier")
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--rtv-weight", type=float, default=0.01)
    parser.add_argument("--rtv-radius", type=int, default=2)
    parser.add_argument("--rtv-sigma", type=float, default=2.0)
    parser.add_argument("--rtv-eps", type=float, default=1e-3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
