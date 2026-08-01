"""
DreamerV3 world model components + dreamerv3-torch integration.

The RSSM, actor-critic, and training loop are now provided by
dreamerv3-torch (PyTorch), NOT CarDreamer's JAX-based DreamerV3.

Our contribution layer:
- encoder_adapter.py: Three-arm encoder factory (CNN, custom JEPA, V-JEPA2)
- cardreamer_encoder_hook.py: Hook to swap our encoders into dreamerv3-torch
"""

from src.dreamer.encoder_adapter import (
    EncoderAdapter,
    CNNEncoder,
    VJEPAEncoder,
    create_encoder,
)
from src.dreamer.cardreamer_encoder_hook import (
    DreamerV3EncoderHook,
    FrameStacker,
    patch_dreamerv3_encoder,
)

__all__ = [
    "EncoderAdapter",
    "CNNEncoder",
    "VJEPAEncoder",
    "create_encoder",
    "DreamerV3EncoderHook",
    "FrameStacker",
    "patch_dreamerv3_encoder",
]
