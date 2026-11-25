"""Test the main steps with small examples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

import numpy as np
from mvs import *
from synthetic import scene


def test_projection():
    # Projecting a point and projecting it back should give the same point.
    images, cameras, truth, box = scene(16)
    camera = cameras[3]
    points = np.array([[.02, .01, .03], [-.01, .02, -.03]])
    uv, z = project(points, **camera)
    np.testing.assert_allclose(
        backproject(uv, z, **camera), points, atol=1e-12
    )


def test_ncc():
    patches = np.random.default_rng(0).random((4, 25, 3))

    # Changing brightness and contrast should keep the score at 1.
    np.testing.assert_allclose(ncc(patches, 2 * patches + 3), 1, atol=1e-12)

    # A patch with no color changes has no valid score.
    assert np.isnan(ncc(patches, np.ones_like(patches))).all()


def test_votes():
    # Two views pass: the mean score is 0.85 and confidence is 0.3125.
    correlation, confidence = vote(np.array([[.9], [.8], [.2], [.1]]), .6)
    np.testing.assert_allclose(correlation, .85)
    np.testing.assert_allclose(confidence, .3125)

    # A score equal to the limit does not count.
    assert np.isneginf(vote(np.array([[.9], [.6]]), .6)[0]).all()


def test_patch():
    pixel_y, pixel_x = np.mgrid[:8, :8]
    image = np.repeat(
        (pixel_x + 2 * pixel_y)[:, :, None], 3, axis=2
    ).astype(float)
    patch, valid = patches(image, np.array([[3.25, 2.5]]), 1)

    # The image value between pixels should still be x + 2*y.
    assert valid[0]
    np.testing.assert_allclose(patch, 8.25)


def test_fusion():
    images, cameras, truth, box = scene(32)
    vertices, faces = fuse(
        truth,
        [np.isfinite(depth_map).astype(float) for depth_map in truth],
        cameras,
        box,
        voxel=.006,
    )
    assert len(faces) > 100

    # On the cube surface, max(|x|, |y|, |z|) is 0.075 m.
    used_vertices = vertices[np.unique(faces)]
    error = abs(np.max(abs(used_vertices), axis=1) - .075)
    assert np.median(error) < .004


def test_bounds():
    camera = dict(K=np.eye(3), R=np.eye(3), t=np.array([0., 0., 0.]))
    uv = np.array([[0., 0.], [3., 0.]])
    lower, upper = ray_bounds(
        uv, camera, np.array([[-1, -1, 2], [1, 1, 4]])
    )

    # The first ray hits the box. The second ray misses it.
    assert lower[0] == 2 and upper[0] == 4
    assert upper[1] < lower[1]


def test_single_neighbor_cannot_vote():
    assert np.isneginf(vote(np.array([[.99]]), .6)[0]).all()


def test_neighbor_separation():
    _, cameras, _, _ = scene(16)
    for reference_id in range(len(cameras)):
        neighbor_ids = neighbors(cameras, reference_id)
        assert reference_id not in neighbor_ids and len(neighbor_ids) == 4
        for neighbor_id in neighbor_ids:
            assert np.degrees(np.arccos(np.clip(
                cameras[reference_id]['R'][2] @ cameras[neighbor_id]['R'][2],
                -1,
                1,
            ))) >= 4
