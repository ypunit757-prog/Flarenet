"""FlareNet: pipeline-based solar flare nowcasting and forecasting."""

import os as _os
import sys as _sys

__version__ = "1.0.0"

if _sys.platform == "darwin":
    # Apple GPU (MPS): if PyTorch lacks an MPS kernel for some op, run that op on the CPU instead
    # of raising NotImplementedError. Must be set before torch is imported, hence here.
    _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
