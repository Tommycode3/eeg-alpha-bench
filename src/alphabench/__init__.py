"""alphabench: a reproducible eyes-open / eyes-closed alpha (Berger effect) benchmark."""

__version__ = "0.1.0"

from .analysis import FAIL, INCONCLUSIVE, PASS, Result, analyze  # noqa: E402
from .config import Config  # noqa: E402
from .io import Recording, load  # noqa: E402

__all__ = ["analyze", "load", "Config", "Recording", "Result", "PASS", "FAIL", "INCONCLUSIVE",
           "__version__"]
