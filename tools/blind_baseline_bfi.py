from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


Array = np.ndarray
Method = Callable[[Array], Array]


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in patterns:
        matches = sorted(glob.glob(item))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(item))
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def corrcoef(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def neighbor_stack(x: Array) -> Array:
    p = np.pad(x.astype(np.float32), 1, mode="reflect")
    return np.stack(
        [
            p[:-2, :-2],
            p[:-2, 1:-1],
            p[:-2, 2:],
            p[1:-1, :-2],
            p[1:-1, 2:],
            p[2:, :-2],
            p[2:, 1:-1],
            p[2:, 2:],
        ],
        axis=0,
    )


def mean4(x: Array) -> Array:
    p = np.pad(x.astype(np.float32), 1, mode="reflect")
    return 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])


def mean8(x: Array) -> Array:
    return np.mean(neighbor_stack(x), axis=0).astype(np.float32)


def median8(x: Array) -> Array:
    return np.median(neighbor_stack(x), axis=0).astype(np.float32)


def ring5_mean(x: Array) -> Array:
    p = np.pad(x.astype(np.float32), 2, mode="reflect")
    acc = np.zeros_like(x, dtype=np.float32)
    count = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) <= 1 and abs(dx) <= 1:
                continue
            acc += p[2 + dy : 2 + dy + x.shape[0], 2 + dx : 2 + dx + x.shape[1]]
            count += 1
    return acc / float(count)


def directional_pair_mean(x: Array) -> Array:
    p = np.pad(x.astype(np.float32), 1, mode="reflect")
    pairs = np.stack(
        [
            0.5 * (p[1:-1, :-2] + p[1:-1, 2:]),
            0.5 * (p[:-2, 1:-1] + p[2:, 1:-1]),
            0.5 * (p[:-2, :-2] + p[2:, 2:]),
            0.5 * (p[:-2, 2:] + p[2:, :-2]),
        ],
        axis=0,
    )
    disagreements = np.stack(
        [
            np.abs(p[1:-1, :-2] - p[1:-1, 2:]),
            np.abs(p[:-2, 1:-1] - p[2:, 1:-1]),
            np.abs(p[:-2, :-2] - p[2:, 2:]),
            np.abs(p[:-2, 2:] - p[2:, :-2]),
        ],
        axis=0,
    )
    choice = np.argmin(disagreements, axis=0)
    return np.take_along_axis(pairs, choice[None, :, :], axis=0)[0].astype(np.float32)


METHODS: dict[str, Method] = {
    "mean4": mean4,
    "mean8": mean8,
    "median8": median8,
    "ring5_mean": ring5_mean,
    "directional_pair_mean": directional_pair_mean,
}


def save_preview_png(path: Path, image: Array, vmin: float, vmax: float) -> None:
    if vmax <= vmin:
        scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        scaled = np.clip((image - vmin) / (vmax - vmin), 0, 1)
        scaled = (scaled * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(scaled).save(path)


def image_metrics(raw: Array, denoised: Array) -> dict[str, float]:
    residual = raw.astype(np.float64) - denoised.astype(np.float64)
    raw_std = float(np.std(raw))
    out_std = float(np.std(denoised))
    return {
        "output_mean": float(np.mean(denoised)),
        "output_std": out_std,
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_mad": float(np.median(np.abs(residual - np.median(residual)))),
        "contrast_retention": out_std / raw_std if raw_std > 0 else float("nan"),
        "residual_output_corr": corrcoef(residual, denoised),
    }


def pair_metrics(outputs: list[Array]) -> dict[str, float]:
    if len(outputs) != 2:
        return {}
    diff = outputs[1].astype(np.float64) - outputs[0].astype(np.float64)
    return {
        "two_frame_output_corr": corrcoef(outputs[0], outputs[1]),
        "two_frame_diff_mean": float(np.mean(diff)),
        "two_frame_diff_std": float(np.std(diff)),
        "two_frame_diff_mad": float(np.median(np.abs(diff - np.median(diff)))),
    }


def make_montage(path: Path, raw: Array, outputs: dict[str, Array], title_prefix: str) -> None:
    names = ["raw", *outputs.keys()]
    images = {"raw": raw, **outputs}
    vmin, vmax = np.percentile(raw, [1, 99])

    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.8 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes, names):
        im = ax.imshow(images[name], cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    for ax in axes[len(names) :]:
        ax.axis("off")
    fig.suptitle(title_prefix)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(paths: list[Path], out_dir: Path) -> dict[str, object]:
    if not paths:
        raise ValueError("No input files found.")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_images = [np.load(path).astype(np.float32) for path in paths]
    if len({image.shape for image in raw_images}) != 1:
        raise ValueError("All inputs must have the same shape.")

    report: dict[str, object] = {
        "inputs": [str(path) for path in paths],
        "methods": {},
    }

    all_outputs: dict[str, list[Array]] = {name: [] for name in METHODS}
    raw_vmin, raw_vmax = np.percentile(np.stack(raw_images), [1, 99])

    for path, raw in zip(paths, raw_images):
        frame_outputs: dict[str, Array] = {}
        for name, fn in METHODS.items():
            denoised = fn(raw).astype(np.float32)
            frame_outputs[name] = denoised
            all_outputs[name].append(denoised)

            np.save(out_dir / f"{path.stem}_{name}.npy", denoised)
            save_preview_png(out_dir / f"{path.stem}_{name}.png", denoised, raw_vmin, raw_vmax)

        make_montage(out_dir / f"{path.stem}_blind_montage.png", raw, frame_outputs, path.stem)

    raw_pair = pair_metrics(raw_images)
    report["raw_two_frame_metrics"] = raw_pair

    for name, outputs in all_outputs.items():
        per_frame = {
            path.stem: image_metrics(raw, denoised)
            for path, raw, denoised in zip(paths, raw_images, outputs)
        }
        pair = pair_metrics(outputs)
        if "two_frame_diff_std" in pair and "two_frame_diff_std" in raw_pair:
            pair["diff_std_ratio_vs_raw"] = (
                pair["two_frame_diff_std"] / raw_pair["two_frame_diff_std"]
                if raw_pair["two_frame_diff_std"] > 0
                else float("nan")
            )
        report["methods"][name] = {
            "per_frame": per_frame,
            "two_frame": pair,
        }

    report_path = out_dir / "blind_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-free blind-pixel baselines for BFI NPY images.")
    parser.add_argument("--inputs", nargs="+", default=["dataset/*.npy"])
    parser.add_argument("--out", type=Path, default=Path("runs/blind_baselines"))
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    report = analyze(paths, args.out)
    print(f"inputs: {len(paths)}")
    print(f"report: {args.out / 'blind_baseline_report.json'}")
    for name, data in report["methods"].items():
        two_frame = data["two_frame"]
        ratio = two_frame.get("diff_std_ratio_vs_raw", float("nan"))
        corr = two_frame.get("two_frame_output_corr", float("nan"))
        print(f"{name}: diff_std_ratio_vs_raw={ratio:.3f}, two_frame_corr={corr:.3f}")


if __name__ == "__main__":
    main()
