from __future__ import annotations

import argparse
import copy
import glob
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class TransformState:
    transform: str
    low: float
    high: float
    raw_min: float


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in patterns:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(item))
    out = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def forward_transform(x: np.ndarray, transform: str, raw_min: float) -> np.ndarray:
    x = x.astype(np.float32)
    if transform == "none":
        return x
    if transform == "sqrt":
        shifted = np.maximum(x - min(raw_min, 0.0), 0.0)
        return np.sqrt(shifted).astype(np.float32)
    if transform == "log1p":
        shifted = np.maximum(x - min(raw_min, 0.0), 0.0)
        return np.log1p(shifted).astype(np.float32)
    raise ValueError(f"Unknown transform: {transform}")


def inverse_transform(x: np.ndarray, state: TransformState) -> np.ndarray:
    x = x.astype(np.float32) * (state.high - state.low) + state.low
    if state.transform == "none":
        return x
    if state.transform == "sqrt":
        return x * x + min(state.raw_min, 0.0)
    if state.transform == "log1p":
        return np.expm1(x) + min(state.raw_min, 0.0)
    raise ValueError(f"Unknown transform: {state.transform}")


def load_and_normalize(paths: list[Path], transform: str) -> tuple[list[np.ndarray], TransformState]:
    raw = [np.load(path).astype(np.float32) for path in paths]
    raw_min = float(min(np.min(x) for x in raw))
    transformed = [forward_transform(x, transform, raw_min) for x in raw]
    joined = np.concatenate([x.ravel() for x in transformed])
    low = float(np.percentile(joined, 0.1))
    high = float(np.percentile(joined, 99.9))
    if high <= low:
        raise ValueError("Normalization percentiles collapsed; check input data.")
    normed = [np.clip((x - low) / (high - low), 0.0, 1.0).astype(np.float32) for x in transformed]
    return normed, TransformState(transform=transform, low=low, high=high, raw_min=raw_min)


def save_preview_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    lo, hi = np.percentile(image, [0.5, 99.5])
    if hi <= lo:
        scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        scaled = np.clip((image - lo) / (hi - lo), 0, 1)
        scaled = (scaled * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install torch first, then rerun this script."
        ) from exc
    return torch, nn, F


def build_model(nn, width: int, depth: int):
    class ResidualDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
            for _ in range(depth - 2):
                layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)])
            layers.append(nn.Conv2d(width, 1, 3, padding=1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            noise = self.net(x)
            return torch.clamp(x - noise, 0.0, 1.0)

    torch, _, _ = import_torch()
    return ResidualDenoiser()


def random_patch_batch(torch, arrays: list[np.ndarray], batch: int, patch: int, device: str):
    patches = []
    for _ in range(batch):
        arr = random.choice(arrays)
        h, w = arr.shape
        if h < patch or w < patch:
            raise ValueError(f"Patch size {patch} exceeds image shape {arr.shape}")
        y = random.randint(0, h - patch)
        x = random.randint(0, w - patch)
        p = arr[y : y + patch, x : x + patch]
        if random.random() < 0.5:
            p = np.flip(p, axis=0)
        if random.random() < 0.5:
            p = np.flip(p, axis=1)
        if random.random() < 0.5:
            p = p.T
        patches.append(np.ascontiguousarray(p[None, :, :]))
    batch_np = np.stack(patches, axis=0)
    return torch.from_numpy(batch_np).to(device=device)


def lp_loss(torch, pred, target, p: float, eps: float = 1e-6):
    return torch.mean(torch.pow(torch.abs(pred - target) + eps, p))


def p_schedule(step: int, total: int) -> float:
    if total <= 1:
        return 1.5
    t = min(1.0, max(0.0, step / float(total - 1)))
    return 2.0 - 0.5 * t


def coeff_like(torch, y, std: float, mode: str):
    if mode == "pixel":
        coeff = torch.randn_like(y) * std + 1.0
    elif mode == "patch":
        coeff = torch.randn((y.shape[0], 1, 1, 1), device=y.device, dtype=y.dtype) * std + 1.0
    else:
        raise ValueError(f"Unknown coefficient mode: {mode}")
    return torch.clamp(coeff, min=0.0)


def train(args: argparse.Namespace) -> None:
    torch, nn, _ = import_torch()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    use_amp = bool(args.amp and use_cuda)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    input_paths = expand_inputs(args.inputs)
    arrays, state = load_and_normalize(input_paths, args.transform)
    args.out.mkdir(parents=True, exist_ok=True)

    model = build_model(nn, args.width, args.depth).to(args.device)
    if args.data_parallel:
        if use_cuda and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"using DataParallel on {torch.cuda.device_count()} GPUs")
        else:
            print("DataParallel requested, but fewer than two CUDA devices are available; using one device")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, float]] = []
    model.train()
    for step in range(args.pretrain_steps):
        clean = random_patch_batch(torch, arrays, args.batch_size, args.patch_size, args.device)
        sigma = torch.empty((args.batch_size, 1, 1, 1), device=args.device).uniform_(
            args.gaussian_sigma_min, args.gaussian_sigma_max
        )
        noisy = torch.clamp(clean + sigma * torch.randn_like(clean), 0.0, 1.0)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(noisy)
            loss = lp_loss(torch, pred, clean, 2.0)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % args.log_every == 0 or step == 0:
            item = {"phase": 0.0, "step": float(step + 1), "loss": float(loss.detach().cpu())}
            history.append(item)
            print(f"pretrain {step + 1}/{args.pretrain_steps} loss={item['loss']:.6f}")

    teacher = copy.deepcopy(model).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    for step in range(args.p2n_steps):
        y = random_patch_batch(torch, arrays, args.batch_size, args.patch_size, args.device)
        with torch.no_grad():
            x_hat = model(y)
            n_hat = y - x_hat
            sigma_p = coeff_like(torch, y, args.rdc_sigma, args.coeff_mode)
            sigma_n = coeff_like(torch, y, args.rdc_sigma, args.coeff_mode)
            y_pos = torch.clamp(x_hat + sigma_n * n_hat, 0.0, 1.0)
            y_neg = torch.clamp(x_hat - sigma_p * n_hat, 0.0, 1.0)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out_pos = model(y_pos)
            out_neg = model(y_neg)
            p = p_schedule(step, args.p2n_steps)
            loss = lp_loss(torch, out_pos, out_neg, p)

            if args.anchor_weight > 0:
                anchor_t = min(1.0, step / float(max(1, args.anchor_decay_steps)))
                weight = args.anchor_weight * (1.0 - anchor_t)
                if weight > 0:
                    anchor_pred = model(y)
                    with torch.no_grad():
                        anchor_target = teacher(y) if args.anchor_target == "teacher" else y
                    loss = loss + weight * lp_loss(torch, anchor_pred, anchor_target, p)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % args.log_every == 0 or step == 0:
            item = {
                "phase": 1.0,
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "p": p,
            }
            history.append(item)
            print(f"p2n {step + 1}/{args.p2n_steps} p={p:.3f} loss={item['loss']:.6f}")

    checkpoint = {
        "model": (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
        "args": vars(args),
        "transform": asdict(state),
        "inputs": [str(p) for p in input_paths],
        "history": history,
    }
    torch.save(checkpoint, args.out / "p2n_bfi_model.pt")
    (args.out / "metadata.json").write_text(
        json.dumps({k: v for k, v in checkpoint.items() if k != "model"}, indent=2, default=str),
        encoding="utf-8",
    )

    model.eval()
    with torch.no_grad():
        for path, arr in zip(input_paths, arrays):
            tensor = torch.from_numpy(arr[None, None, :, :]).to(device=args.device)
            den = model(tensor).detach().cpu().numpy()[0, 0]
            den_raw = inverse_transform(den, state).astype(np.float32)
            stem = path.stem
            np.save(args.out / f"{stem}_p2n_denoised.npy", den_raw)
            save_preview_png(args.out / f"{stem}_p2n_denoised.png", den_raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2N-style single-frame denoising for BFI NPY images.")
    parser.add_argument("--inputs", nargs="+", default=["dataset/*.npy"])
    parser.add_argument("--out", type=Path, default=Path("runs/p2n_bfi"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--transform", choices=["sqrt", "log1p", "none"], default="sqrt")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--pretrain-steps", type=int, default=1000)
    parser.add_argument("--gaussian-sigma-min", type=float, default=0.02)
    parser.add_argument("--gaussian-sigma-max", type=float, default=0.12)

    parser.add_argument("--p2n-steps", type=int, default=3000)
    parser.add_argument("--rdc-sigma", type=float, default=0.2)
    parser.add_argument("--coeff-mode", choices=["pixel", "patch"], default="pixel")
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--anchor-decay-steps", type=int, default=500)
    parser.add_argument("--anchor-target", choices=["teacher", "input"], default="teacher")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--data-parallel", action="store_true", help="Use torch.nn.DataParallel when multiple CUDA GPUs are visible.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
