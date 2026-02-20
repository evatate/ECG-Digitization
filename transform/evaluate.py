import argparse
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import resample
from scipy.stats import pearsonr
from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import get_cfg
from src.utils import find_config_path


def crop_ahus_raw_timeseries(gt: npt.NDArray[Any]) -> npt.NDArray[Any]:
    if gt.shape[0] > 5000:
        return gt[:5000]
    return gt


def find_ground_truth_csvs(gt_dir: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for entry in os.scandir(gt_dir):
        if entry.is_dir():
            expected = os.path.join(entry.path, f"digital_signal_{entry.name}.csv")
            if os.path.isfile(expected):
                results.append((entry.name, expected))
    return results


def find_digitized_csv(gt_folder: str) -> dict[str, str]:
    results: dict[str, str] = {}
    if not os.path.isdir(gt_folder):
        return results
    for fname in os.listdir(gt_folder):
        if fname.endswith("_timeseries_canonical.csv"):
            stem = fname[: -len("_timeseries_canonical.csv")]
            results[stem] = os.path.join(gt_folder, fname)
    return results


def load_signal_csv(path: str) -> npt.NDArray[Any]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data.T if data.shape[0] < data.shape[1] else data  # (leads, samples)


def resample_to_length(signal: npt.NDArray[Any], target_length: int) -> npt.NDArray[Any]:
    leads, _ = signal.shape
    return np.stack([resample(signal[lead], target_length) for lead in range(leads)])


def shift_and_crop(arr: npt.NDArray[Any], shift: int) -> npt.NDArray[Any]:
    if shift == 0:
        return arr
    elif shift > 0:
        return arr[..., shift:]
    else:
        return arr[..., :shift]


def compute_metrics(gt: npt.NDArray[Any], pred: npt.NDArray[Any], n_translations: int = 0) -> dict[str, list[float]]:
    gt = crop_ahus_raw_timeseries(gt)
    gt, pred = gt.T, pred.T  # (leads, samples) in units of mV

    metrics: dict[str, list[float]] = {
        "pearson": [],
        "rms": [],
        "snr_db": [],
        "shift": [],
        "nans_fraction": [],
    }
    max_shift = n_translations // 2
    min_shift = -n_translations // 2

    if n_translations == 1:
        shifts = [0]
    else:
        shifts = np.arange(min_shift, max_shift + 1).tolist()

    for lead in range(gt.shape[0]):
        best_snr = -np.inf
        best_metrics = {"pearson": np.nan, "rms": np.nan, "snr_db": np.nan, "shift": 0, "nans_fraction": 0.0}
        for shift in shifts:
            if shift == 0:
                gt_aligned = gt[lead]
                pred_shifted = pred[lead]
            elif shift > 0:
                gt_aligned = gt[lead][shift:]
                pred_shifted = pred[lead][:-shift]
            else:  # shift < 0
                gt_aligned = gt[lead][:shift]
                pred_shifted = pred[lead][-shift:]

            mask = ~(np.isnan(gt_aligned) | np.isnan(pred_shifted))
            if not np.any(mask):
                continue
            gt_valid = gt_aligned[mask]
            pred_valid = pred_shifted[mask]
            gt_valid -= np.mean(gt_valid)
            pred_valid -= np.mean(pred_valid)

            try:
                pearson = pearsonr(gt_valid, pred_valid)[0]
            except Exception:
                pearson = np.nan
            rms = float(np.sqrt(np.mean((gt_valid - pred_valid) ** 2)))
            power_signal = float(np.mean(gt_valid**2))
            power_noise = float(np.mean((gt_valid - pred_valid) ** 2))
            snr_db = 10 * np.log10(power_signal / power_noise) if power_noise > 0 else np.nan

            if best_snr < snr_db:
                best_snr = snr_db
                best_metrics = {
                    "pearson": pearson,
                    "rms": rms,
                    "snr_db": snr_db,
                    "shift": shift,
                    "nans_fraction": np.mean(np.isnan(pred_shifted)),
                }

        metrics["pearson"].append(best_metrics["pearson"])
        metrics["rms"].append(best_metrics["rms"])
        metrics["snr_db"].append(best_metrics["snr_db"])
        metrics["shift"].append(best_metrics["shift"])
        metrics["nans_fraction"].append(best_metrics["nans_fraction"])

    return metrics


def main(cfg: CN) -> None:
    digitized_dir: str = cfg["digitized_dir"]
    ground_truth_dir: str = cfg["ground_truth_dir"]
    results_csv: str = cfg["results_csv"]
    n_translations: int = cfg.get("shift_steps", 0)

    gt_files = find_ground_truth_csvs(ground_truth_dir)
    if not gt_files:
        print("No ground truth files found.")
        return

    results: list[dict[str, Any]] = []

    for folder, gt_csv in tqdm(gt_files):
        digitized_folder = os.path.join(digitized_dir, folder)
        digitized_csvs = find_digitized_csv(digitized_folder)

        gt_signal = load_signal_csv(gt_csv)
        for stem, digitized_csv in digitized_csvs.items():
            digitized_signal = load_signal_csv(digitized_csv)

            metrics = compute_metrics(gt_signal, digitized_signal, n_translations)

            row: dict[str, Any] = {
                "folder": folder,
                "gt_csv": os.path.basename(gt_csv),
                "digitized_csv": os.path.basename(digitized_csv),
                **{f"pearson_{i+1}": v for i, v in enumerate(metrics["pearson"])},
                **{f"rms_{i+1}": v for i, v in enumerate(metrics["rms"])},
                **{f"snr_db_{i+1}": v for i, v in enumerate(metrics["snr_db"])},
                **{f"shift_{i+1}": v for i, v in enumerate(metrics["shift"])},
                **{f"nans_fraction_{i+1}": v for i, v in enumerate(metrics["nans_fraction"])},
            }
            results.append(row)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    df.to_csv(results_csv, index=False)
    print(f"Saved metrics to {results_csv}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate digitized ECGs against ground truth.")
    argparser.add_argument(
        "--config",
        type=str,
        default="evaluate.yml",
        help="Config file name or path (searched in . and src/config/). Default: evaluate.yml",
    )

    args = argparser.parse_args()
    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)
    main(cfg)