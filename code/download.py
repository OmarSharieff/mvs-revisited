"""Download the temple data and record its file hash."""

from pathlib import Path
import urllib.request
import zipfile
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
url = 'https://vision.middlebury.edu/mview/data/data/templeSparseRing.zip'
archive_path = ROOT / 'data' / 'templeSparseRing.zip'
archive_path.parent.mkdir(exist_ok=True)

# Reuse the local data if the camera file exists.
if not (ROOT / 'data/templeSparseRing/templeSR_par.txt').exists():
    urllib.request.urlretrieve(url, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        # Check that every file will stay inside the data folder.
        for item in archive.infolist():
            target = (ROOT / 'data' / item.filename).resolve()
            if not target.is_relative_to((ROOT / 'data').resolve()):
                raise ValueError('Unsafe archive path')
        archive.extractall(ROOT / 'data')

    # Save the SHA-256 hash of the downloaded file.
    (ROOT / 'data/download.json').write_text(json.dumps(
        dict(
            url=url,
            sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        ),
        indent=2,
    ))
