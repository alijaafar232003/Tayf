"""Tayf: RF spectrum monitoring, detection, investigation and classification."""
import os
import sys


def _add_conda_dll_paths() -> None:
    """Conda packages on Windows (SciPy's LAPACK, SoapySDR's HackRF module) load
    their DLLs from <env>/Library/bin, which is on PATH only when the env is
    activated, not when python.exe is started directly (pytest, an IDE, a
    notebook kernel). Without it, scipy.linalg crashes the process. Add it here,
    before anything below imports SciPy."""
    if os.name != "nt":
        return
    lib = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(lib) and lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(lib)


_add_conda_dll_paths()
