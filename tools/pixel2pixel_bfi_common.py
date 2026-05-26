from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass
class VSTState:
    """Parameters needed to invert the sqrt/log transform and robust normalization."""

    transform: str
    low: float
    high: float
    raw_min: float


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    """Expand shell-style globs while preserving deterministic order."""

    import glob

    paths: list[Path] = []
    for item in patterns:
        matches = sorted(glob.glob(item))
        paths.extend(Path(match) for match in matches) if matches else paths.append(Path(item))

    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def vst_forward(x: np.ndarray, transform: str, raw_min: float) -> np.ndarray:
    """Apply the variance-stabilizing transform in raw intensity space."""

    x = x.astype(np.float32)
    shift = min(raw_min, 0.0)
    if transform == "none":
        return x
    if transform == "sqrt":
        return np.sqrt(np.maximum(x - shift, 0.0)).astype(np.float32)
    if transform == "log1p":
        return np.log1p(np.maximum(x - shift, 0.0)).astype(np.float32)
    raise ValueError(f"Unknown transform: {transform}")


def vst_inverse(x: np.ndarray, state: VSTState) -> np.ndarray:
    """Invert robust normalization and the variance-stabilizing transform."""

    x = x.astype(np.float32) * (state.high - state.low) + state.low
    shift = min(state.raw_min, 0.0)
    if state.transform == "none":
        return x
    if state.transform == "sqrt":
        return x * x + shift
    if state.transform == "log1p":
        return np.expm1(x) + shift
    raise ValueError(f"Unknown transform: {state.transform}")


def fit_vst_normalization(
    images: list[np.ndarray],
    transform: str = "sqrt",
    low_percentile: float = 0.1,
    high_percentile: float = 99.9,
) -> tuple[list[np.ndarray], VSTState]:
    """Fit one robust normalization for all images and return normalized VST images."""

    raw_min = float(min(np.min(image) for image in images))
    transformed = [vst_forward(image, transform, raw_min) for image in images]
    joined = np.concatenate([image.ravel() for image in transformed])
    low = float(np.percentile(joined, low_percentile))
    high = float(np.percentile(joined, high_percentile))
    if high <= low:
        raise ValueError("Normalization percentiles collapsed; check input data.")
    normed = [np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32) for image in transformed]
    return normed, VSTState(transform=transform, low=low, high=high, raw_min=raw_min)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def estimate_correlation_radius(
    noise_like: np.ndarray,
    threshold: float = 0.03,
    max_lag: int = 12,
    multiplier: float = 2.0,
    min_radius: int = 3,
) -> tuple[int, dict[str, float]]:
    """Estimate a conservative exclusion radius from horizontal/vertical ACF.

    If a true noise residual is available, pass that residual. Otherwise pass a
    high-pass residual of the noisy image. The returned radius is approximately
    `multiplier * last_lag_above_threshold`.
    """

    x = noise_like.astype(np.float64)
    last = 0
    acf: dict[str, float] = {}
    for lag in range(1, max_lag + 1):
        horizontal = corrcoef(x[:, lag:], x[:, :-lag])
        vertical = corrcoef(x[lag:, :], x[:-lag, :])
        acf[f"h{lag}"] = horizontal
        acf[f"v{lag}"] = vertical
        if max(abs(horizontal), abs(vertical)) > threshold:
            last = lag
    radius = max(min_radius, int(np.ceil(last * multiplier)))
    return radius, acf


def _zscore_columns(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def extract_pixel_features(
    image: np.ndarray,
    patch_size: int = 7,
    match_sigma: float = 0.8,
    patch_weight: float = 1.0,
    stats_weight: float = 0.35,
    grad_weight: float = 0.35,
) -> np.ndarray:
    """Build per-pixel matching features from patch appearance and local structure.

    The feature image is optionally pre-smoothed before patch extraction so that
    nearest-neighbor search is driven more by repeated structure than by noise.
    """

    if patch_size % 2 != 1:
        raise ValueError("--bank-patch-size must be odd.")

    image = image.astype(np.float32)
    base = ndimage.gaussian_filter(image, sigma=match_sigma, mode="reflect") if match_sigma > 0 else image
    radius = patch_size // 2
    padded = np.pad(base, radius, mode="reflect")
    patches = np.lib.stride_tricks.sliding_window_view(padded, (patch_size, patch_size))
    patch_features = patches.reshape(image.size, patch_size * patch_size).astype(np.float32)
    patch_features = _zscore_columns(patch_features) * patch_weight

    local_mean = ndimage.uniform_filter(base, size=patch_size, mode="reflect")
    local_sq_mean = ndimage.uniform_filter(base * base, size=patch_size, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
    stats_features = np.stack([local_mean.ravel(), local_std.ravel()], axis=1).astype(np.float32)
    stats_features = _zscore_columns(stats_features) * stats_weight

    gy, gx = np.gradient(base)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_features = np.stack([gx.ravel(), gy.ravel(), grad_mag.ravel()], axis=1).astype(np.float32)
    grad_features = _zscore_columns(grad_features) * grad_weight

    return np.concatenate([patch_features, stats_features, grad_features], axis=1).astype(np.float32)


def query_pixel_bank(
    features: np.ndarray,
    shape: tuple[int, int],
    k: int = 32,
    exclude_radius: int = 5,
    query_multiplier: int = 8,
    max_query: int = 2048,
    chunk_size: int = 2048,
    workers: int = -1,
    query_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a non-local pixel bank for selected pixels.

    Each row contains K source-pixel indices with similar features. Candidates
    whose image-space distance is <= exclude_radius are removed so two sampled
    pseudo observations are less likely to share correlated noise.
    """

    h, w = shape
    n_pixels = h * w
    if features.shape[0] != n_pixels:
        raise ValueError(f"Feature count {features.shape[0]} does not match image shape {shape}.")
    if query_indices is None:
        query_indices = np.arange(n_pixels, dtype=np.int64)
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64)

    coords = np.column_stack(np.unravel_index(np.arange(n_pixels), shape)).astype(np.int32)
    tree = cKDTree(features)
    bank = np.empty((len(query_indices), k), dtype=np.int64)
    bank_dist = np.empty((len(query_indices), k), dtype=np.float32)

    base_query = min(n_pixels, max(k + 1, k * query_multiplier))
    for start in range(0, len(query_indices), chunk_size):
        end = min(start + chunk_size, len(query_indices))
        current = query_indices[start:end]
        filled = np.zeros(end - start, dtype=bool)
        query_k = base_query

        while not np.all(filled):
            distances, indices = tree.query(features[current], k=min(query_k, n_pixels), workers=workers)
            if distances.ndim == 1:
                distances = distances[:, None]
                indices = indices[:, None]

            for row, global_index in enumerate(current):
                if filled[row]:
                    continue
                candidate_idx = indices[row]
                candidate_dist = distances[row]
                delta = coords[candidate_idx] - coords[global_index]
                spatial_dist = np.sqrt(np.sum(delta.astype(np.float32) ** 2, axis=1))
                valid = (candidate_idx != global_index) & (spatial_dist > exclude_radius)
                candidate_idx = candidate_idx[valid]
                candidate_dist = candidate_dist[valid]
                if candidate_idx.size >= k:
                    bank[start + row] = candidate_idx[:k]
                    bank_dist[start + row] = candidate_dist[:k]
                    filled[row] = True

            if np.all(filled):
                break
            if query_k >= min(max_query, n_pixels):
                # Extremely small images or very large exclusion radii may not
                # have enough candidates. Repeat the available candidates rather
                # than silently changing K.
                for row, global_index in enumerate(current):
                    if filled[row]:
                        continue
                    candidate_idx = indices[row]
                    candidate_dist = distances[row]
                    delta = coords[candidate_idx] - coords[global_index]
                    spatial_dist = np.sqrt(np.sum(delta.astype(np.float32) ** 2, axis=1))
                    valid = (candidate_idx != global_index) & (spatial_dist > exclude_radius)
                    candidate_idx = candidate_idx[valid]
                    candidate_dist = candidate_dist[valid]
                    if candidate_idx.size == 0:
                        raise ValueError("No valid non-local candidates found; reduce --exclude-radius.")
                    repeats = int(np.ceil(k / candidate_idx.size))
                    bank[start + row] = np.tile(candidate_idx, repeats)[:k]
                    bank_dist[start + row] = np.tile(candidate_dist, repeats)[:k]
                break
            query_k = min(query_k * 2, max_query, n_pixels)

    return bank, bank_dist


def save_preview_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    lo, hi = np.percentile(image, [0.5, 99.5])
    if hi <= lo:
        scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        scaled = np.clip((image - lo) / (hi - lo), 0, 1)
        scaled = (scaled * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(scaled).save(path)
