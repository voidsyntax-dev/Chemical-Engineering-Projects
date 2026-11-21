"""Thin wrapper around the compiled Downs & Vogel Fortran (via f2py; source: camaramm/tep2py,
modified temain_mod.f + teprob.f). Returns 52 columns: XMEAS(1-41) then XMV(1-11)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
import temain_mod  # noqa: E402

COLS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]


def simulate(idv, nsamples=None):
    """idv: int array (nsamples, 20) of disturbance flags per 3-min sample (IDV1..IDV20)."""
    idv = np.asarray(idv, dtype=np.int32)
    assert idv.ndim == 2 and idv.shape[1] == 20
    n = idv.shape[0]
    x = temain_mod.temain(np.int32(60 * 3 * n), np.int32(n), np.asfortranarray(idv), np.int32(1))
    return np.asarray(x)
