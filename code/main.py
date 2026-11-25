import argparse
import json
import time
import csv
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib
# Save plots without opening a window.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mvs import depth, fuse, save_ply
from synthetic import scene
ROOT = Path(__file__).resolve().parents[1]

def run(name, images, cams, box, truth=None, **cfg):
    # Run one experiment with the settings in cfg.
    dest = ROOT / 'results' / name
    dest.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # Store depth, confidence, and match scores for all views.
    ds = []
    cs = []
    ss = []
    for i in range(len(images)):
        # Use each photo as the reference view in turn.
        d, c, s = depth(images, cams, i, box, **cfg)
        ds.append(d)
        cs.append(c)
        ss.append(s)
        print(name, i + 1, '/', len(images), flush=True)

    # Time the depth step, then build the mesh.
    elapsed = time.perf_counter() - start
    v, f = fuse(ds, cs, cams, box)
    save_ply(dest / 'mesh.ply', v, f)

    np.savez_compressed(
        dest / 'maps.npz',
        depth=ds,
        confidence=cs,
        correlation=ss,
    )
    np.savez_compressed(
        dest / 'inputs.npz',
        images=images,
        K=[c['K'] for c in cams],
        R=[c['R'] for c in cams],
        t=[c['t'] for c in cams],
        box=box,
    )

    result = dict(
        run=name,
        **cfg,
        seconds=elapsed,
        views=len(images),
        valid_fraction=float(np.isfinite(ds).mean()),
        vertices=len(v),
        faces=len(f),
    )

    # Use known depths to measure error and coverage.
    if truth is not None:
        gt = np.array(truth)
        pred = np.array(ds)
        ok = np.isfinite(gt) & np.isfinite(pred)
        err = abs(gt[ok] - pred[ok]) * 1000
        result.update(
            mae_mm=float(err.mean()) if len(err) else None,
            p90_mm=float(np.percentile(err, 90)) if len(err) else None,
            foreground_coverage=float(ok.sum() / np.isfinite(gt).sum()),
            background_predictions=int(
                (np.isfinite(pred) & ~np.isfinite(gt)).sum()
            ),
        )

    # Plot the first photo and its three result maps.
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(images[0])
    ax[0].set_title('Reference RGB')
    for a, data, title in zip(
        ax[1:],
        [ds[0], cs[0], ss[0]],
        ['Camera-z (m)', 'Confidence', 'Correlation'],
    ):
        p = a.imshow(
            data,
            vmin=0 if title == 'Confidence' else None,
            vmax=1 if title == 'Confidence' else None,
        )
        fig.colorbar(p, ax=a, shrink=.7)
        a.set_title(title)
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(dest / 'diagnostics.png', dpi=140)
    plt.close(fig)

    # Save a preview if the mesh has faces.
    if len(f):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(
            v[:, 0], v[:, 1], v[:, 2],
            triangles=f,
            color='steelblue',
            linewidth=0,
        )
        ax.set_box_aspect((1, 1, 1))
        fig.savefig(dest / 'mesh.png', dpi=130)
        plt.close(fig)

    (dest / 'metrics.json').write_text(json.dumps(result, indent=2))
    return result

def main():
    # Read options from the command line.
    p = argparse.ArgumentParser()
    p.add_argument('--suite', action='store_true')  # Run all cube experiments.
    p.add_argument('--data', type=Path)  # Folder with real photos.
    p.add_argument('--box', type=float, nargs=6)  # Two corners of the search box.
    p.add_argument('--size', type=int, default=48)  # Width and height of cube images.
    args = p.parse_args()
    np.random.seed(0)

    results = []

    # Load real photos and their camera settings.
    if args.data:
        from PIL import Image

        folder = ROOT / args.data
        files = list(folder.glob('*_par.txt'))
        if len(files) != 1 or args.box is None:
            p.error(
                'Supply a dataset folder containing one *_par.txt and --box '
                'xmin ymin zmin xmax ymax zmax in camera units (meters).'
            )

        images = []
        cams = []
        for line in files[0].read_text().splitlines():
            tok = line.split()
            if len(tok) != 22:
                continue

            # Each line has a file name, 9 values for K, 9 for R, and 3 for t.
            a = np.array(tok[1:], float)
            im = Image.open(folder / tok[0]).convert('RGB')

            # Resize the photo and update K to match.
            ow, oh = im.size
            nw = 160
            nh = round(oh * nw / ow)
            im = im.resize((nw, nh), Image.Resampling.BILINEAR)
            K = a[:9].reshape(3, 3)
            A = np.array([
                [nw / ow, 0, (nw / ow - 1) / 2],
                [0, nh / oh, (nh / oh - 1) / 2],
                [0, 0, 1],
            ])
            images.append(np.asarray(im, float) / 255)
            cams.append(dict(
                K=A @ K,
                R=a[9:18].reshape(3, 3),
                t=a[18:],
            ))

        if len(cams) < 3:
            raise ValueError('At least three calibrated images required')
        results.append(run(
            'real',
            images,
            cams,
            np.array(args.box).reshape(2, 3),
            m=5,
            threshold=.6,
            step=.0025,
        ))
    else:
        # Run the cube baseline by default.
        # Each row gives the name, texture flag, patch size, and score limit.
        settings = [('baseline', True, 5, .6)]
        if args.suite:
            settings += [
                ('window3', True, 3, .6),
                ('window9', True, 9, .6),
                ('threshold04', True, 5, .4),
                ('threshold08', True, 5, .8),
                ('textureless', False, 5, .6),
            ]

        for name, texture, m, threshold in settings:
            # Use gt to check the results after depth estimation.
            images, cams, gt, box = scene(args.size, texture)
            results.append(run(
                name,
                images,
                cams,
                box,
                gt,
                m=m,
                threshold=threshold,
                step=.005,
            ))

    # Save the run summary, metrics, and system details.
    out = ROOT / 'results'
    (out / 'summary.json').write_text(json.dumps(results, indent=2))
    keys = list(dict.fromkeys(k for r in results for k in r))
    with (out/'metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    (out / 'environment.json').write_text(json.dumps(
        dict(
            python=sys.version,
            platform=platform.platform(),
            processor=platform.processor(),
            seed=0,
        ),
        indent=2,
    ))


# Start here when this file is run directly.
if __name__ == '__main__':
    main()
