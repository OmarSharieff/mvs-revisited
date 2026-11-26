"""Visual inspection of the real Temple MVS reconstruction from a single camera view."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from mvs import project

# --- Configuration ---
RESULTS_DIR = Path("results/real")
CAMERA_INDEX = 0  # First camera in the dataset

# Set to (left, right, bottom, top) in pixels to generate a cropped close-up,
# or leave as None to render the full overview.
CROP_REGION = (45, 115, 65, 25)

def load_camera_view(results_dir: Path, camera_index: int):
    """Load the photo, camera matrices (K, R, t), and estimated depth map for one view."""
    with np.load(results_dir / "inputs.npz") as inputs:
        photo = inputs["images"][camera_index]
        intrinsics = inputs["K"][camera_index]
        rotation = inputs["R"][camera_index]
        translation = inputs["t"][camera_index]

    with np.load(results_dir / "maps.npz") as maps:
        depth_map = maps["depth"][camera_index]

    return photo, intrinsics, rotation, translation, depth_map


def load_ply_vertices(ply_path: Path) -> np.ndarray:
    """Read 3D (x, y, z) vertex coordinates from an ASCII PLY mesh file."""
    vertex_count = 0

    with open(ply_path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break

        vertices = np.array(
            [list(map(float, file.readline().split()[:3])) for _ in range(vertex_count)],
            dtype=float,
        )

    return vertices


def project_visible_vertices(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Project 3D mesh vertices into the camera image and keep points in front of the lens."""
    pixel_coords, point_depths = project(vertices, intrinsics, rotation, translation)

    in_front_of_camera = point_depths > 0
    has_valid_coords = np.isfinite(pixel_coords).all(axis=1)
    is_visible = in_front_of_camera & has_valid_coords

    return pixel_coords[is_visible]


def save_inspection_figure(
    photo: np.ndarray,
    depth_map: np.ndarray,
    projected_points: np.ndarray,
    output_path: Path,
    crop_region: tuple[float, float, float, float] | None = None,
) -> None:
    """Plot the photo, estimated depth map, and projected mesh vertices side by side."""
    image_height, image_width = depth_map.shape

    fig, (ax_photo, ax_depth, ax_overlay) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4),
        sharex=True,
        sharey=True,
    )

    # Panel 1: Original input photo
    ax_photo.imshow(photo)
    ax_photo.set_title("Photo")

    # Panel 2: Single-view estimated depth map
    depth_image = ax_depth.imshow(depth_map)
    fig.colorbar(depth_image, ax=ax_depth, label="metres", shrink=0.6)
    ax_depth.set_title("Estimated depth")

    # Panel 3: Reconstructed 3D mesh vertices projected onto the photo
    ax_overlay.imshow(photo)
    ax_overlay.scatter(
        projected_points[:, 0],
        projected_points[:, 1],
        s=0.2,
        c="cyan",
        alpha=0.3,
    )
    ax_overlay.set_title("Projected mesh points")

    # Set axis bounds (full image by default, or zoomed to the selected crop)
    if crop_region is not None:
        left, right, bottom, top = crop_region
        x_limits, y_limits = (left, right), (bottom, top)
    else:
        x_limits = (-0.5, image_width - 0.5)
        y_limits = (image_height - 0.5, -0.5)

    for axis in (ax_photo, ax_depth, ax_overlay):
        axis.set(xlim=x_limits, ylim=y_limits)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    photo, intrinsics, rotation, translation, depth_map = load_camera_view(
        RESULTS_DIR, CAMERA_INDEX
    )
    vertices = load_ply_vertices(RESULTS_DIR / "mesh.ply")

    if len(vertices) == 0:
        raise ValueError("Mesh is empty. Record this finding.")

    projected_points = project_visible_vertices(
        vertices, intrinsics, rotation, translation
    )

    prefix = "temple_detail" if CROP_REGION is not None else "temple_view"
    output_path = Path(f"{prefix}_{CAMERA_INDEX}.png")

    save_inspection_figure(
        photo=photo,
        depth_map=depth_map,
        projected_points=projected_points,
        output_path=output_path,
        crop_region=CROP_REGION,
    )


if __name__ == "__main__":
    main()