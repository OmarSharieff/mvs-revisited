"""Estimate depth from photos and use it to build a surface.

X_camera = R @ X_world + t. Depth is camera z, in meters.
"""

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes


def project(X, K, R, t):
    """Map world points to pixels and return their camera depths."""
    camera_points = X @ R.T + t
    homogeneous_pixels = camera_points @ K.T
    return (
        homogeneous_pixels[..., :2] / homogeneous_pixels[..., 2:3],
        camera_points[..., 2],
    )


def backproject(uv, z, K, R, t):
    """Use a pixel and its depth to find a world point."""
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    return (rays * z[:, None] - t) @ R


def neighbors(cams, i, k=4):
    """Pick up to k nearby views for camera i."""
    optical_axes = np.array([camera['R'][2] for camera in cams])
    angles = np.degrees(
        np.arccos(np.clip(optical_axes @ optical_axes.T, -1, 1))
    )
    selected = []

    # Pick close views first, keeping at least 4 degrees between views.
    for candidate in np.argsort(angles[i]):
        if (
            candidate != i
            and angles[i, candidate] >= 4
            and all(angles[candidate, other] >= 4 for other in selected)
        ):
            selected.append(int(candidate))
        if len(selected) == k:
            break
    return selected


def patches(im, uv, m):
    """Read m-by-m RGB patches and mark which ones fit in the image.

    The shape is (number of pixels, m*m, 3). Blend values between pixels.
    """
    offsets = np.arange(m) - m // 2
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing='ij')
    x = uv[:, 0, None] + offset_x.ravel()
    y = uv[:, 1, None] + offset_y.ravel()

    valid = (
        (x.min(1) >= 0)
        & (y.min(1) >= 0)
        & (x.max(1) < im.shape[1] - 1)
        & (y.max(1) < im.shape[0] - 1)
        & np.isfinite(uv).all(1)
    )
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    sampled_patches = np.stack(
        [
            map_coordinates(im[:, :, channel], [y, x], order=1, mode='nearest')
            for channel in range(3)
        ],
        -1,
    )
    return sampled_patches, valid


def ncc(a, b):
    """Score two sets of RGB patches. Flat patches give NaN."""
    # Subtract each color mean before scoring all colors together.
    a = a - a.mean(-2, keepdims=True)
    b = b - b.mean(-2, keepdims=True)
    energy_a = (a * a).sum((-2, -1))
    energy_b = (b * b).sum((-2, -1))
    denominator = np.sqrt(energy_a * energy_b)

    return np.divide(
        (a * b).sum((-2, -1)),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=(energy_a > 1e-10) & (energy_b > 1e-10),
    ).clip(-1, 1)


def vote(scores, threshold):
    """Combine match scores to get a score and confidence for each depth."""
    accepted = np.isfinite(scores) & (scores > threshold)
    support_count = accepted.sum(0)

    # Require two passing views and take the mean of the passing scores.
    correlation = np.divide(
        np.where(accepted, scores, 0).sum(0),
        support_count,
        out=np.full(support_count.shape, -np.inf),
        where=support_count >= 2,
    )
    confidence = (
        np.where(accepted, scores - threshold, 0).sum(0)
        / (len(scores) * (1 - threshold))
    )
    return correlation, confidence


def ray_bounds(uv, cam, box):
    """Find the depth range where each ray is inside the search box."""
    origin = -cam['R'].T @ cam['t']
    rays = (
        np.c_[uv, np.ones(len(uv))]
        @ np.linalg.inv(cam['K']).T
        @ cam['R']
    )
    lower = np.zeros(len(uv))
    upper = np.full(len(uv), np.inf)

    for axis in range(3):
        parallel = abs(rays[:, axis]) < 1e-12
        first = np.divide(
            box[0, axis] - origin[axis],
            rays[:, axis],
            out=np.full(len(uv), -np.inf),
            where=~parallel,
        )
        second = np.divide(
            box[1, axis] - origin[axis],
            rays[:, axis],
            out=np.full(len(uv), np.inf),
            where=~parallel,
        )
        lower = np.maximum(lower, np.minimum(first, second))
        upper = np.minimum(upper, np.maximum(first, second))

        # A ray parallel to a pair of box faces misses if it lies outside them.
        if not box[0, axis] <= origin[axis] <= box[1, axis]:
            upper[parallel] = -1
    return lower, upper


def depth(images, cams, i, box, m=5, threshold=.6, step=.005):
    """Return depth, confidence, and match score maps for view i."""
    camera = cams[i]
    height, width = images[i].shape[:2]
    pixel_y, pixel_x = np.mgrid[:height, :width]
    uv = np.c_[pixel_x.ravel(), pixel_y.ravel()].astype(float)

    reference_patches, valid = patches(images[i], uv, m)
    lower, upper = ray_bounds(uv, camera, box)
    valid &= upper > lower
    valid_ids = np.where(valid)[0]

    estimated_depth = np.full(height * width, np.nan)
    confidence = np.zeros(height * width)
    correlation = np.full(height * width, np.nan)
    neighbor_ids = neighbors(cams, i)
    if len(neighbor_ids) < 2:
        return (
            estimated_depth.reshape(height, width),
            confidence.reshape(height, width),
            correlation.reshape(height, width),
        )

    # Work on a group of pixels at a time to save memory.
    for start in range(0, len(valid_ids), 512):
        batch_ids = valid_ids[start:start + 512]
        pixels = uv[batch_ids]
        best_score = np.full(len(batch_ids), -np.inf)
        best_depth = np.full(len(batch_ids), np.nan)
        best_confidence = np.zeros(len(batch_ids))

        def evaluate(z):
            # Guess depth, find the 3D point, then compare patches in other views.
            points = backproject(
                pixels, z, camera['K'], camera['R'], camera['t']
            )
            neighbor_scores = []
            for neighbor_id in neighbor_ids:
                projected_pixels, neighbor_depth = project(
                    points, **cams[neighbor_id]
                )
                neighbor_patches, patch_valid = patches(
                    images[neighbor_id], projected_pixels, m
                )
                score = ncc(reference_patches[batch_ids], neighbor_patches)
                score[~patch_valid | (neighbor_depth <= 0)] = np.nan
                neighbor_scores.append(score)

            candidate_score, candidate_confidence = vote(
                np.array(neighbor_scores), threshold
            )
            candidate_score[
                (z < lower[batch_ids]) | (z > upper[batch_ids])
            ] = -np.inf
            return candidate_score, candidate_confidence

        # Try depths spaced by step and keep the best score.
        for offset in np.arange(
            0, np.max(upper[batch_ids] - lower[batch_ids]) + step * .5, step
        ):
            z = lower[batch_ids] + offset
            score, candidate_confidence = evaluate(z)
            improved = score > best_score
            best_score[improved] = score[improved]
            best_depth[improved] = z[improved]
            best_confidence[improved] = candidate_confidence[improved]

        # Try smaller steps around the best depth from the first pass.
        coarse_depth = best_depth.copy()
        for offset in np.linspace(-step, step, 21):
            z = np.where(
                np.isfinite(coarse_depth),
                coarse_depth + offset,
                lower[batch_ids],
            )
            score, candidate_confidence = evaluate(z)
            improved = (score > best_score) & np.isfinite(coarse_depth)
            best_score[improved] = score[improved]
            best_depth[improved] = z[improved]
            best_confidence[improved] = candidate_confidence[improved]

        estimated_depth[batch_ids] = best_depth
        confidence[batch_ids] = best_confidence
        correlation[batch_ids] = np.where(
            np.isfinite(best_score), best_score, np.nan
        )

    return (
        estimated_depth.reshape(height, width),
        confidence.reshape(height, width),
        correlation.reshape(height, width),
    )


def fuse(depths, confs, cams, box, voxel=.004, uniform=False):
    """Build a mesh by combining depth differences near the surface."""
    axes = [
        np.arange(box[0, axis], box[1, axis] + voxel * .5, voxel)
        for axis in range(3)
    ]
    shape = tuple(map(len, axes))
    grid_points = np.stack(
        np.meshgrid(*axes, indexing='ij'), -1
    ).reshape(-1, 3)
    weighted_residuals = np.zeros(len(grid_points))
    total_weights = weighted_residuals.copy()
    band = 4 * voxel

    for depth_map, confidence_map, camera in zip(depths, confs, cams):
        uv, z = project(grid_points, **camera)
        u = np.rint(uv[:, 0]).astype(int)
        v = np.rint(uv[:, 1]).astype(int)
        valid = (
            (z > 0)
            & (u >= 0)
            & (u < depth_map.shape[1])
            & (v >= 0)
            & (v < depth_map.shape[0])
        )
        indices = np.where(valid)[0]
        measured = depth_map[v[indices], u[indices]]
        weight = confidence_map[v[indices], u[indices]]

        # Positive means in front of the measured surface. Negative means behind.
        residual = measured - z[indices]
        keep = (
            np.isfinite(residual) & (abs(residual) <= band) & (weight > 0)
        )
        indices = indices[keep]
        weight = weight[keep]
        if uniform:
            weight = np.ones_like(weight)
        weighted_residuals[indices] += weight * residual[keep] / band
        total_weights[indices] += weight

    field = np.divide(
        weighted_residuals,
        total_weights,
        out=np.ones_like(weighted_residuals),
        where=total_weights > 0,
    ).reshape(shape)
    observed = (total_weights > 0).reshape(shape)
    if not (field.min() < 0 < field.max()):
        return np.empty((0, 3)), np.empty((0, 3), int)

    # Build triangles where the depth difference crosses zero.
    vertices, faces, _, _ = marching_cubes(
        field, level=0, spacing=(voxel,) * 3
    )
    centers = vertices[faces].mean(1) / voxel
    cells = np.floor(centers).astype(int)
    cells = np.minimum(cells, np.array(shape) - 2)

    # Keep faces only if all eight corners of their grid cell were seen.
    keep = np.ones(len(faces), bool)
    for offset_x in (0, 1):
        for offset_y in (0, 1):
            for offset_z in (0, 1):
                keep &= observed[
                    cells[:, 0] + offset_x,
                    cells[:, 1] + offset_y,
                    cells[:, 2] + offset_z,
                ]
    return vertices + box[0], faces[keep]


def save_ply(path, vertices, faces):
    """Save vertices and triangle indices as a text PLY file."""
    with open(path, 'w') as output:
        output.write(
            f'ply\n'
            f'format ascii 1.0\n'
            f'element vertex {len(vertices)}\n'
            f'property float x\n'
            f'property float y\n'
            f'property float z\n'
            f'element face {len(faces)}\n'
            f'property list uchar int vertex_indices\n'
            f'end_header\n'
        )
        np.savetxt(output, vertices, fmt='%.8f')
        np.savetxt(
            output, np.c_[np.full(len(faces), 3), faces], fmt='%d'
        )
