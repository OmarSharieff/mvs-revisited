"""Check cube depths and plot results for cameras 0 and 3."""

from pathlib import Path

import matplotlib
import numpy as np

# Save plots without opening a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic import scene


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_comparison(view, photo, error_mm, missed_cube, extra_background):
    """Show the depth checks for one photo in four panels."""
    figure, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(photo)
    error_image = axes[1].imshow(error_mm, cmap="magma", vmin=0)
    figure.colorbar(error_image, ax=axes[1], label="mm")

    # White pixels mark the problem shown in each panel.
    axes[2].imshow(missed_cube, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(extra_background, cmap="gray", vmin=0, vmax=1)
    titles = ("Photo", "Depth error", "Missed cube", "Extra background")
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.axis("off")

    output_path = PROJECT_ROOT / f"cube_depth_view_{view}.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved {output_path}")


def main():
    maps_path = PROJECT_ROOT / "results" / "baseline" / "maps.npz"
    if not maps_path.is_file():
        raise SystemExit("Baseline missing. Run: python code/main.py --size 48")

    # Make the 48-by-48 cube scene and load the saved depths.
    images, _, truth, _ = scene(size=48, texture=True)
    true_depth = np.asarray(truth)
    with np.load(maps_path) as maps:
        predicted_depth = maps["depth"]

    if predicted_depth.shape != true_depth.shape:
        raise SystemExit(
            f"Expected depth shape {true_depth.shape}, got {predicted_depth.shape}. "
            "Regenerate the baseline with: python code/main.py --size 48"
        )

    is_cube = np.isfinite(true_depth)
    has_prediction = np.isfinite(predicted_depth)
    matched_cube = is_cube & has_prediction
    missed_cube = is_cube & ~has_prediction
    extra_background = has_prediction & ~is_cube

    # Compare pixels with both depths and convert the error to mm.
    error_mm = np.full(true_depth.shape, np.nan)
    error_mm[matched_cube] = (
        np.abs(predicted_depth[matched_cube] - true_depth[matched_cube]) * 1000
    )

    # Count all views, then plot cameras 0 and 3.
    if matched_cube.any():
        print(f"Mean error: {error_mm[matched_cube].mean():.3f} mm")
    else:
        print("Mean error: N/A (no cube pixels have predictions)")
    coverage = 100 * matched_cube.sum() / is_cube.sum()
    print(f"Coverage: {coverage:.2f}%")
    print(f"Background predictions: {extra_background.sum()}")

    for view in (0, 3):
        save_comparison(
            view, images[view], error_mm[view],
            missed_cube[view], extra_background[view],
        )


if __name__ == "__main__":
    main()
