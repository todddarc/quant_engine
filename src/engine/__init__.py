# Temporary shim for backward compatibility; prefer `quant_engine` imports.
try:
    from quant_engine import *  # noqa
except ImportError:
    # Try relative import when running as module
    from ..quant_engine import *  # noqa 