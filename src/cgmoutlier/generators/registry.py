"""Name -> generator class. Heavy generators are imported lazily so that using
copy_paste (or just the data/MIA code) does not require torch/lightning."""
from __future__ import annotations

from .base import GeneratorBase


def get(name: str):
    """Return the GeneratorBase subclass registered under `name`."""
    name = name.lower()
    if name == "copy_paste":
        from .copy_paste import CopyPasteGenerator
        return CopyPasteGenerator
    if name == "diffusion_ts":
        from .diffusion_ts import DiffusionTSGenerator
        return DiffusionTSGenerator
    if name == "fourier_diff":
        from .fourier_diff import FourierDiffGenerator
        return FourierDiffGenerator
    if name == "timevae":
        from .timevae import TimeVAEGenerator
        return TimeVAEGenerator
    if name == "diffwave":
        from .diffwave import DiffWaveGenerator
        return DiffWaveGenerator
    if name == "padts":
        from .padts import PaDTSGenerator
        return PaDTSGenerator
    if name == "igfm":
        from .igfm import IGFMGenerator
        return IGFMGenerator
    if name == "dimts":
        # importable in the ordinary environment; it only needs the DiM-TS one when
        # you call fit(). See docs/DIMTS.md.
        from .dimts import DiMTSGenerator
        return DiMTSGenerator
    raise KeyError(f"unknown generator '{name}'")


ALL = ["copy_paste", "diffusion_ts", "fourier_diff", "timevae", "diffwave",
       "padts", "igfm", "dimts"]
