"""Download the original archive; retain a checksum and safe flat extraction."""
from pathlib import Path
import urllib.request,zipfile,hashlib,json
R=Path(__file__).resolve().parents[1];url='https://vision.middlebury.edu/mview/data/data/templeSparseRing.zip'
p=R/'data'/'templeSparseRing.zip';p.parent.mkdir(exist_ok=True)
if not (R/'data/templeSparseRing/templeSR_par.txt').exists():
    urllib.request.urlretrieve(url,p)
    with zipfile.ZipFile(p) as z:
        for item in z.infolist():
            target=(R/'data'/item.filename).resolve()
            if not target.is_relative_to((R/'data').resolve()):raise ValueError('Unsafe archive path')
        z.extractall(R/'data')
    (R/'data/download.json').write_text(json.dumps(dict(url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
