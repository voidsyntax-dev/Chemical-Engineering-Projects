"""Pack the project sources into ONE line of Python that unpacks them inside a Kaggle notebook.

  python kaggle/make_cell.py      ->  kaggle/cell_unpack.py  (paste as the first notebook cell)
"""
import base64
import io
import tarfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["cstr/__init__.py", "cstr/vdv.py", "cstr/env.py", "cstr/baseline.py",
         "train.py", "evaluate.py", "check_model.py"]

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tar:
    for f in FILES:
        tar.add(ROOT / f, arcname=f)
blob = base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode()
line = ("import base64,io,tarfile,zlib;tarfile.open(fileobj=io.BytesIO(zlib.decompress(base64.b64decode('"
        + blob + "'))),mode='r').extractall('/kaggle/working/rlvalve');print('unpacked',len('" + blob[:8] + "'))\n")
out = ROOT / "kaggle" / "cell_unpack.py"
out.write_text(line)
print(f"{out}: {len(line):,} chars")
