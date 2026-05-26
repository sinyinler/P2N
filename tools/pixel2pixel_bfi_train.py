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


def build_model(nn, width: int, depth: int, max_residual: float):
    """A small ordinary image-to-image residual CNN, not a blind-spot network."""

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
            self.max_residual = float(max_residual)

        def forward(self, x):
            # Bound the predicted residual so early training cannot jump to a
            # pathological all-black/all-white solution.
            residual = self.max_residual * torch.tanh(self.net(x))
            return x - residual

    return ResidualCNN()


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
        f"gw{args.grad_weight}_ex{args.exclude_radius}.npz"
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

    print(f"building bank for {path.name}: K={args.bank_size}, exclude_radius={exclude_radius}")
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
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        bank=bank,
        bank_dist=bank_dist,
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
    for item in items:
        pred_norm = infer_full_image(torch, model, item.image, args.device)
        pred_raw = vst_inverse(pred_norm, state).astype(np.float32)
        name = "pixel2pixel_denoised" if tag == "final" else f"pixel2pixel_{tag}"
        np.save(out_dir / f"{item.path.stem}_{name}.npy", pred_raw)
        save_preview_png(out_dir / f"{item.path.stem}_{name}.png", pred_raw)
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
    if len({image.shape for image in raw_images}) != 1:
        raise ValueError("All input images must have the same shape.")
    images, state = fit_vst_normalization(raw_images, transform=args.transform)

    args.out.mkdir(parents=True, exist_ok=True)
    items = [build_bank_for_image(args, path, image) for path, image in zip(input_paths, images)]

    model = build_model(nn, args.width, args.depth, args.max_residual).to(args.device)
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
            # Pixel2Pixel/N2N theory uses L2 when noise is approximately zero-mean.
            loss = F.mse_loss(pred.float(), y.float())

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
            }
            history.append(item)
            print(
                f"step {step + 1}/{args.steps} "
                f"mse={item['loss']:.6f} grad={grad_norm:.3f} skipped={int(skipped)}"
            )

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            save_artifacts(torch, nn, model, args, state, items, history, tag=f"step{step + 1}")

    save_artifacts(torch, nn, model, args, state, items, history, tag="final")


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
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--reuse-bank", action="store_true")

    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--max-residual", type=float, default=0.35)
    parser.add_argument("--train-patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-consecutive-skips", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
