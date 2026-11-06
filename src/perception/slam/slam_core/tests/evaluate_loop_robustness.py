"""Sweep false loop translation and rotation errors for three noise models."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from test_pose_graph_loops import run_backend, run_gaussian_false_loop


def plot_sweep(errors, series, xlabel, ylabel, title, output):
    figure, axes = plt.subplots(figsize=(8, 5))
    for label, values in series.items():
        axes.plot(errors, values, linewidth=2, label=label)
    axes.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def print_samples(errors, series, sample_values):
    print("error," + ",".join(series.keys()))
    for sample in sample_values:
        index = int(np.argmin(np.abs(errors - sample)))
        values = ",".join(f"{series[name][index]:.4f}" for name in series)
        print(f"{errors[index]:.2f},{values}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-error", type=float, default=1.0,
                        help="maximum false loop translation error in meters")
    parser.add_argument("--max-rotation-deg", type=float, default=15.0,
                        help="maximum false loop rotation error in degrees")
    parser.add_argument("--points", type=int, default=61,
                        help="number of samples in each sweep")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.max_error <= 0 or args.max_rotation_deg <= 0 or args.points < 2:
        parser.error("both maximum errors must be positive and --points at least 2")

    labels = ("Plain Gaussian", "Huber", "Cauchy")
    translation_errors = np.linspace(0.0, args.max_error, args.points)
    translation_results = {label: [] for label in labels}
    for error_m in translation_errors:
        translation_results["Plain Gaussian"].append(
            run_gaussian_false_loop(error_m, false_rotation=False)[0]
        )
        translation_results["Huber"].append(
            run_backend("huber", True, error_m, false_rotation=False)[0]
        )
        translation_results["Cauchy"].append(
            run_backend("cauchy", True, error_m, false_rotation=False)[0]
        )

    rotation_errors = np.linspace(0.0, args.max_rotation_deg, args.points)
    rotation_results = {label: [] for label in labels}
    for error_deg in rotation_errors:
        rotation_results["Plain Gaussian"].append(
            run_gaussian_false_loop(
                0.0, false_rotation_error_deg=error_deg
            )[1]
        )
        rotation_results["Huber"].append(
            run_backend(
                "huber", True, 0.0,
                false_rotation_error_deg=error_deg,
            )[1]
        )
        rotation_results["Cauchy"].append(
            run_backend(
                "cauchy", True, 0.0,
                false_rotation_error_deg=error_deg,
            )[1]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    translation_plot = args.output_dir / "loop_robustness_translation.png"
    rotation_plot = args.output_dir / "loop_robustness_rotation.png"
    plot_sweep(
        translation_errors,
        translation_results,
        "False loop translation error (m)",
        "ATE translation RMSE (m)",
        "ATE vs false loop translation error (0–1 m)",
        translation_plot,
    )
    plot_sweep(
        rotation_errors,
        rotation_results,
        "False loop rotation error (deg)",
        "ATE rotation RMSE (deg)",
        "Rotational ATE vs false loop rotation error",
        rotation_plot,
    )

    baseline = run_backend()
    print(f"VO-only reference: translation ATE={baseline[0]:.4f} m; rotation ATE={baseline[1]:.4f} deg")
    print("\nTranslation sweep: x=translation error in meters; y=translation ATE in meters")
    print_samples(
        translation_errors, translation_results,
        np.linspace(0.0, args.max_error, 5),
    )
    print("\nRotation sweep: x=rotation error in degrees; y=rotation ATE in degrees")
    print_samples(
        rotation_errors, rotation_results,
        np.linspace(0.0, args.max_rotation_deg, 7),
    )
    print(f"\nSaved plots:\n  {translation_plot}\n  {rotation_plot}")


if __name__ == "__main__":
    main()
