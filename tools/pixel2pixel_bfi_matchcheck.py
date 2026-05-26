from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from pixel2pixel_bfi_common import (
    VSTState,
    estimate_correlation_radius,
    extract_pixel_features,
    query_pixel_bank,
    vst_forward,
)


def normalize_noisy_and_gt(noisy: np.ndarray, gt: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray, VSTState]:
    """Use the noisy frame's robust range, then measure GT variation in that same domain."""

    raw_min = float(min(np.min(noisy), np.min(gt)))
    noisy_t = vst_forward(noisy, transform, raw_min)
    gt_t = vst_forward(gt, transform, raw_min)
    low = float(np.percentile(noisy_t, 0.1))
    high = float(np.percentile(noisy_t, 99.9))
    if high <= low:
        raise ValueError("Normalization percentiles collapsed; check the noisy input.")
    noisy_norm = np.clip((noisy_t - low) / (high - low), 0.0, 1.0).astype(np.float32)
    # Do not clip GT metrics: clipping would artificially reduce the bank variance.
    gt_norm = ((gt_t - low) / (high - low)).astype(np.float32)
    return noisy_norm, gt_norm, VSTState(transform=transform, low=low, high=high, raw_min=raw_min)


def resize_gt_like_noisy(gt: np.ndarray, noisy_shape: tuple[int, int], mode: str) -> tuple[np.ndarray, str]:
    """Resize a high-resolution reference to the noisy BFI grid for match checking."""

    if gt.shape == noisy_shape:
        return gt.astype(np.float32), "none"
    if mode == "none":
        raise ValueError(f"Shape mismatch: noisy {noisy_shape}, gt {gt.shape}. Use --resize-gt auto.")
    if mode != "auto":
        raise ValueError(f"Unknown --resize-gt mode: {mode}")

    scale_y = noisy_shape[0] / gt.shape[0]
    scale_x = noisy_shape[1] / gt.shape[1]
    # When downsampling the quasi-GT, a light pre-blur reduces aliasing while
    # preserving the smooth long-window signal used only for bank validation.
    sigma_y = max(0.0, (1.0 / max(scale_y, 1e-6) - 1.0) * 0.25)
    sigma_x = max(0.0, (1.0 / max(scale_x, 1e-6) - 1.0) * 0.25)
    blurred = ndimage.gaussian_filter(gt.astype(np.float32), sigma=(sigma_y, sigma_x), mode="reflect")
    resized = ndimage.zoom(blurred, zoom=(scale_y, scale_x), order=1)
    resized = resized[: noisy_shape[0], : noisy_shape[1]]
    if resized.shape != noisy_shape:
        padded = np.empty(noisy_shape, dtype=np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        if resized.shape[0] < noisy_shape[0]:
            padded[resized.shape[0] :, : resized.shape[1]] = resized[-1:, :]
        if resized.shape[1] < noisy_shape[1]:
            padded[:, resized.shape[1] :] = padded[:, resized.shape[1] - 1 : resized.shape[1]]
        resized = padded
    return resized.astype(np.float32), f"auto_linear_from_{gt.shape[0]}x{gt.shape[1]}"


def make_report_plot(
    out_path: Path,
    noisy: np.ndarray,
    gt: np.ndarray,
    sample_indices: np.ndarray,
    bank: np.ndarray,
    bank_gt_var: np.ndarray,
    noise_var: float,
) -> None:
    h, w = noisy.shape
    gt_flat = gt.ravel()
    center_gt = gt_flat[sample_indices]
    bank_gt = gt_flat[bank]
    bank_mean = bank_gt.mean(axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    im0 = axes[0, 0].imshow(noisy, cmap="magma")
    axes[0, 0].set_title("noisy sqrt-normalized")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(gt, cmap="magma")
    axes[0, 1].set_title("GT sqrt-normalized")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    ratios = bank_gt_var / max(noise_var, 1e-12)
    axes[0, 2].hist(ratios, bins=80, color="#3a6ea5")
    axes[0, 2].axvline(1.0, color="crimson", linestyle="--", linewidth=1)
    axes[0, 2].set_title("bank GT variance / noise variance")
    axes[0, 2].set_xlabel("ratio")

    axes[1, 0].hexbin(center_gt, bank_mean, gridsize=60, mincnt=1)
    axes[1, 0].plot([center_gt.min(), center_gt.max()], [center_gt.min(), center_gt.max()], color="white", linewidth=1)
    axes[1, 0].set_title("center GT vs bank mean GT")
    axes[1, 0].set_xlabel("center")
    axes[1, 0].set_ylabel("bank mean")

    rng = np.random.default_rng(7)
    examples = rng.choice(len(sample_indices), size=min(8, len(sample_indices)), replace=False)
    axes[1, 1].imshow(noisy, cmap="gray")
    axes[1, 1].set_title("example queried pixels")
    for idx in examples:
        y, x = divmod(int(sample_indices[idx]), w)
        axes[1, 1].scatter([x], [y], s=18, marker="x")

    axes[1, 2].imshow(noisy, cmap="gray")
    axes[1, 2].set_title("example bank matches")
    for idx in examples[:4]:
        y, x = divmod(int(sample_indices[idx]), w)
        by, bx = np.unravel_index(bank[idx, : min(12, bank.shape[1])], (h, w))
        axes[1, 2].scatter([x], [y], s=24, marker="x")
        axes[1, 2].scatter(bx, by, s=10, alpha=0.7)

    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    noisy_raw = np.load(args.noisy).astype(np.float32)
    gt_raw = np.load(args.gt).astype(np.float32)
    gt_raw, resize_mode = resize_gt_like_noisy(gt_raw, noisy_raw.shape, args.resize_gt)

    noisy, gt, state = normalize_noisy_and_gt(noisy_raw, gt_raw, args.transform)
    noise_residual = noisy - gt

    if args.exclude_radius == "auto":
        exclude_radius, acf = estimate_correlation_radius(
            noise_residual,
            threshold=args.acf_threshold,
            max_lag=args.max_acf_lag,
            multiplier=args.exclude_multiplier,
            min_radius=args.min_exclude_radius,
        )
    else:
        exclude_radius = int(args.exclude_radius)
        _, acf = estimate_correlation_radius(
            noise_residual,
            threshold=args.acf_threshold,
            max_lag=args.max_acf_lag,
            multiplier=args.exclude_multiplier,
            min_radius=args.min_exclude_radius,
        )

    features = extract_pixel_features(
        noisy,
        patch_size=args.bank_patch_size,
        match_sigma=args.match_sigma,
        patch_weight=args.patch_weight,
        stats_weight=args.stats_weight,
        grad_weight=args.grad_weight,
    )

    rng = np.random.default_rng(args.seed)
    n_pixels = noisy.size
    sample_count = min(args.sample_count, n_pixels)
    sample_indices = rng.choice(n_pixels, size=sample_count, replace=False)
    bank, bank_dist = query_pixel_bank(
        features,
        noisy.shape,
        k=args.bank_size,
        exclude_radius=exclude_radius,
        query_multiplier=args.query_multiplier,
        max_query=args.max_query,
        chunk_size=args.chunk_size,
        workers=args.workers,
        query_indices=sample_indices,
    )

    gt_flat = gt.ravel()
    noisy_flat = noisy.ravel()
    bank_gt = gt_flat[bank]
    bank_noisy = noisy_flat[bank]
    center_gt = gt_flat[sample_indices]
    bank_gt_var = np.var(bank_gt, axis=1)
    bank_gt_mean = np.mean(bank_gt, axis=1)
    noise_var = float(np.var(noise_residual))
    noise_std = float(np.sqrt(noise_var))
    gt_var_ratio = bank_gt_var / max(noise_var, 1e-12)

    center_bias = bank_gt_mean - center_gt
    report: dict[str, object] = {
        "inputs": {
            "noisy": str(args.noisy),
            "gt": str(args.gt),
        },
        "vst_state": state.__dict__,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gt_resize": resize_mode,
        "estimated_exclude_radius": int(exclude_radius),
        "acf": acf,
        "noise_variance_in_vst_domain": noise_var,
        "noise_std_in_vst_domain": noise_std,
        "bank_gt_variance": {
            "mean": float(np.mean(bank_gt_var)),
            "median": float(np.median(bank_gt_var)),
            "p90": float(np.percentile(bank_gt_var, 90)),
            "mean_ratio_to_noise_var": float(np.mean(gt_var_ratio)),
            "median_ratio_to_noise_var": float(np.median(gt_var_ratio)),
            "p90_ratio_to_noise_var": float(np.percentile(gt_var_ratio, 90)),
        },
        "bank_gt_bias_to_center": {
            "mean": float(np.mean(center_bias)),
            "mean_abs": float(np.mean(np.abs(center_bias))),
            "rmse": float(np.sqrt(np.mean(center_bias * center_bias))),
            "rmse_ratio_to_noise_std": float(np.sqrt(np.mean(center_bias * center_bias)) / max(noise_std, 1e-12)),
        },
        "bank_noisy_variance": {
            "mean": float(np.mean(np.var(bank_noisy, axis=1))),
            "median": float(np.median(np.var(bank_noisy, axis=1))),
        },
        "match_distance": {
            "mean": float(np.mean(bank_dist)),
            "median": float(np.median(bank_dist)),
        },
    }

    median_ratio = report["bank_gt_variance"]["median_ratio_to_noise_var"]
    bias_ratio = report["bank_gt_bias_to_center"]["rmse_ratio_to_noise_std"]
    if median_ratio < 0.25 and bias_ratio < 0.5:
        verdict = "pass"
    elif median_ratio < 1.0 and bias_ratio < 1.0:
        verdict = "caution"
    else:
        verdict = "fail"
    report["verdict"] = verdict

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pixel2pixel_matchcheck_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.out / "pixel2pixel_matchcheck_samples.npz",
        sample_indices=sample_indices,
        bank=bank,
        bank_dist=bank_dist,
        bank_gt_var=bank_gt_var,
        center_gt=center_gt,
        bank_gt_mean=bank_gt_mean,
    )
    make_report_plot(
        args.out / "pixel2pixel_matchcheck.png",
        noisy,
        gt,
        sample_indices,
        bank,
        bank_gt_var,
        noise_var,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pixel2Pixel pixel-bank match quality check for BFI images.")
    parser.add_argument("--noisy", type=Path, required=True, help="Noisy BFI .npy image used for pixel searching.")
    parser.add_argument("--gt", type=Path, required=True, help="Approximate clean/long-window GT .npy image.")
    parser.add_argument("--out", type=Path, default=Path("reports/pixel2pixel_matchcheck"))
    parser.add_argument("--transform", choices=["sqrt", "log1p", "none"], default="log1p")
    parser.add_argument("--resize-gt", choices=["auto", "none"], default="auto")

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

    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--query-multiplier", type=int, default=8)
    parser.add_argument("--max-query", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"verdict": result["verdict"], "estimated_exclude_radius": result["estimated_exclude_radius"]}, indent=2))
