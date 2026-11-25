"""Make cube photos and compute their true camera depths."""

import numpy as np


def scene(size=64, texture=True):
    """Return 12 photos, camera settings, true depths, and a search box."""
    images = []
    cameras = []
    truth = []

    K = np.array([
        [75 * size / 64, 0, (size - 1) / 2],
        [0, 75 * size / 64, (size - 1) / 2],
        [0, 0, 1.],
    ])
    pixel_y, pixel_x = np.mgrid[:size, :size]
    rays = np.stack(
        [pixel_x, pixel_y, np.ones_like(pixel_x)], -1
    ) @ np.linalg.inv(K).T

    # Place 12 cameras in a circle facing the cube.
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        center = np.array([.35 * np.cos(angle), .35 * np.sin(angle), .17])
        forward = -center / np.linalg.norm(center)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.array([right, down, forward])
        t = -R @ center

        world_rays = rays @ R
        near = np.full((size, size), -np.inf)
        far = np.full((size, size), np.inf)

        # Find where each ray enters and leaves the cube, which is 0.15 m wide.
        for axis in range(3):
            with np.errstate(divide='ignore', invalid='ignore'):
                first = (-.075 - center[axis]) / world_rays[:, :, axis]
                second = (.075 - center[axis]) / world_rays[:, :, axis]
            near = np.maximum(near, np.minimum(first, second))
            far = np.minimum(far, np.maximum(first, second))

        valid = (far >= near) & (near > 0)
        z = np.where(valid, near, np.nan)
        points = center + world_rays * np.nan_to_num(z)[:, :, None]

        # Set the color from the 3D point so it stays the same across views.
        if texture:
            rgb = np.stack([
                .5
                + .22 * np.sin(130 * points[:, :, 0] + 31 * points[:, :, 2])
                + .2 * np.cos(105 * points[:, :, 1]),
                .5
                + .25 * np.sin(119 * points[:, :, 1] + 43 * points[:, :, 0]),
                .5
                + .24 * np.cos(
                    113 * points[:, :, 2]
                    + 61 * points[:, :, 0]
                    + 41 * points[:, :, 1]
                ),
            ], -1)
        else:
            rgb = np.full((size, size, 3), .7)

        rgb[~valid] = .05
        images.append(rgb.clip(0, 1))
        truth.append(z)
        cameras.append(dict(K=K.copy(), R=R, t=t))

    return images, cameras, truth, np.array([[-.09] * 3, [.09] * 3])
