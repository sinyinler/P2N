from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, stats


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def shift_pair(x: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    if dy >= 0:
        a = x[dy:, :]
        b = x[: x.shape[0] - dy, :]
    else:
        a = x[: x.shape[0] + dy, :]
        b = x[-dy:, :]

    if dx >= 0:
        a = a[:, dx:]
        b = b[:, : b.shape[1] - dx]
    else:
        a = a[:, : a.shape[1] + dx]
        b = b[:, -dx:]
    return a, b


def autocorr_at(x: np.ndarray, dy: int, dx: int) -> float:
    a, b = shift_pair(x, dy, dx)
    return corrcoef(a, b)


def acf_map(x: np.ndarray, radius: int) -> np.ndarray:
    out = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    for iy, dy in enumerate(range(-radius, radius + 1)):
        for ix, dx in enumerate(range(-radius, radius + 1)):
            out[iy, ix] = autocorr_at(x, dy, dx)
    return out


def neighbor_predictability(x: np.ndarray, neighbors: int = 8) -> float:
    center = x[1:-1, 1:-1].ravel()
    features = [
        x[1:-1, :-2].ravel(),
        x[1:-1, 2:].ravel(),
        x[:-2, 1:-1].ravel(),
        x[2:, 1:-1].ravel(),
    ]
    if neighbors == 8:
        features.extend(
            [
                x[:-2, :-2].ravel(),
                x[:-2, 2:].ravel(),
                x[2:, :-2].ravel(),
                x[2:, 2:].ravel(),
            ]
        )

    design = np.vstack(features).T
    design = np.column_stack([np.ones(center.size), design])
    coeffs = np.linalg.lstsq(design, center, rcond=None)[0]
    pred = design @ coeffs
    ss_res = float(np.sum((center - pred) ** 2))
    ss_tot = float(np.sum((center - center.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def robust_summary(x: np.ndarray) -> dict[str, object]:
    flat = x.astype(np.float64).ravel()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "mad": float(np.median(np.abs(flat - np.median(flat)))),
        "skew": float(stats.skew(flat)),
        "excess_kurtosis": float(stats.kurtosis(flat)),
        "quantiles": {
            str(q): float(v)
            for q, v in zip(
                [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9],
                np.percentile(flat, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]),
            )
        },
    }


def make_plot(
    y0: np.ndarray,
    y1: np.ndarray,
    diff: np.ndarray,
    acf: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmin = float(np.percentile(np.stack([y0, y1]), 1))
    vmax = float(np.percentile(np.stack([y0, y1]), 99))
    dlim = float(np.percentile(np.abs(diff), 99))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    im0 = axes[0, 0].imshow(y0, cmap="magma", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("frame 0")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(y1, cmap="magma", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("frame 1")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[0, 2].imshow(diff, cmap="coolwarm", vmin=-dlim, vmax=dlim)
    axes[0, 2].set_title("frame 1 - frame 0")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    axes[1, 0].hist(diff.ravel(), bins=120, color="#3a6ea5")
    axes[1, 0].set_title("difference histogram")

    im3 = axes[1, 1].imshow(acf, cmap="coolwarm", vmin=-0.08, vmax=0.08)
    center = acf.shape[0] // 2
    axes[1, 1].scatter([center], [center], color="black", s=12)
    axes[1, 1].set_title("difference ACF")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)

    avg = 0.5 * (y0 + y1)
    axes[1, 2].hexbin(avg.ravel(), np.abs(diff).ravel(), gridsize=60, mincnt=1)
    axes[1, 2].set_title("|difference| vs local signal")
    axes[1, 2].set_xlabel("two-frame average")
    axes[1, 2].set_ylabel("|difference|")

    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def analyze(path0: Path, path1: Path, out_dir: Path, radius: int) -> dict[str, object]:
    y0 = np.load(path0).astype(np.float64)
    y1 = np.load(path1).astype(np.float64)
    if y0.shape != y1.shape:
        raise ValueError(f"Shape mismatch: {path0} {y0.shape} vs {path1} {y1.shape}")

    diff = y1 - y0
    avg = 0.5 * (y0 + y1)
    diff_acf = acf_map(diff, radius)
    center = radius
    first_neighbors = {
        "horizontal_1": float(diff_acf[center, center + 1]),
        "vertical_1": float(diff_acf[center + 1, center]),
        "diag_down_1": float(diff_acf[center + 1, center + 1]),
        "diag_up_1": float(diff_acf[center - 1, center + 1]),
        "horizontal_2": float(diff_acf[center, center + 2]) if radius >= 2 else float("nan"),
        "vertical_2": float(diff_acf[center + 2, center]) if radius >= 2 else float("nan"),
    }

    highpass_temporal = {}
    highpass_spatial = {}
    for sigma in [1, 2, 3, 5, 8]:
        r0 = y0 - ndimage.gaussian_filter(y0, sigma=sigma, mode="reflect")
        r1 = y1 - ndimage.gaussian_filter(y1, sigma=sigma, mode="reflect")
        rd = diff - ndimage.gaussian_filter(diff, sigma=sigma, mode="reflect")
        highpass_temporal[str(sigma)] = corrcoef(r0, r1)
        highpass_spatial[str(sigma)] = {
            "horizontal_1": autocorr_at(rd, 0, 1),
            "vertical_1": autocorr_at(rd, 1, 0),
            "neighbor_r2_8": neighbor_predictability(rd, 8),
        }

    report = {
        "inputs": [str(path0), str(path1)],
        "frame0": robust_summary(y0),
        "frame1": robust_summary(y1),
        "frame_correlation": corrcoef(y0, y1),
        "difference": robust_summary(diff),
        "difference_positive_fraction": float(np.mean(diff > 0)),
        "difference_vs_average_correlation": corrcoef(diff, avg),
        "abs_difference_vs_average_correlation": corrcoef(np.abs(diff), avg),
        "difference_spatial_acf_first_neighbors": first_neighbors,
        "difference_neighbor_predictability_r2": {
            "4_neighbors": neighbor_predictability(diff, 4),
            "8_neighbors": neighbor_predictability(diff, 8),
        },
        "highpass_temporal_correlation_by_gaussian_sigma": highpass_temporal,
        "highpass_difference_spatial_by_gaussian_sigma": highpass_spatial,
        "interpretation": {
            "spatial_j_invariance": "pass"
            if max(abs(v) for v in first_neighbors.values() if np.isfinite(v)) < 0.05
            and neighbor_predictability(diff, 8) < 0.01
            else "caution",
            "noise_symmetry": "caution"
            if abs(float(np.mean(diff))) > 0.05 * float(np.std(diff))
            or abs(float(stats.skew(diff.ravel()))) > 0.2
            else "pass",
            "signal_dependent_noise": "strong"
            if abs(corrcoef(np.abs(diff), avg)) > 0.3
            else "weak",
            "temporal_correlation": "insufficient_two_frames",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "bfi_noise_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_plot(y0, y1, diff, diff_acf, out_dir / "bfi_noise_diagnostics.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BFI noise assumptions for P2N/J-invariance.")
    parser.add_argument("--frame0", default="dataset/0_nonoverlap.npy", type=Path)
    parser.add_argument("--frame1", default="dataset/1_nonoverlap.npy", type=Path)
    parser.add_argument("--out", default="reports", type=Path)
    parser.add_argument("--acf-radius", default=5, type=int)
    args = parser.parse_args()

    report = analyze(args.frame0, args.frame1, args.out, args.acf_radius)
    print(json.dumps(report["interpretation"], indent=2))
    print(f"report: {args.out / 'bfi_noise_report.json'}")
    print(f"figure: {args.out / 'bfi_noise_diagnostics.png'}")


if __name__ == "__main__":
    main()
