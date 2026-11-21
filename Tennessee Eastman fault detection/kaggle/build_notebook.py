"""Convert notebook_src.py (percent format) into tep_neural_kaggle.ipynb."""
import json, pathlib, re
here = pathlib.Path(__file__).parent
src = (here / "notebook_src.py").read_text(encoding="utf-8")
cells = []
for block in re.split(r"(?m)^# %%", src)[1:]:
    head, _, body = block.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in head:
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(here / "tep_neural_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", len(cells), "cells")
