"""
DreamerV3 world model components + CarDreamer integration.

The RSSM, actor-critic, and training loop are now provided by CarDreamer's
tested DreamerV3 implementation (third_party/CarDreamer/dreamerv3/).

Our contribution layer:
- encoder_adapter.py: Three-arm encoder factory (CNN, custom JEPA, V-JEPA2)
- cardreamer_encoder_hook.py: Hook to swap our encoders into CarDreamer's pipeline
"""

from src.dreamer.encoder_adapter import (
    EncoderAdapter,
    CNNEncoder,
    VJEPAEncoder,
    create_encoder,
)
from src.dreamer.cardreamer_encoder_hook import (
    CarDreamerEncoderHook,
    FrameStacker,
    patch_dreamerv3_encoder,
)

__all__ = [
    "EncoderAdapter",
    "CNNEncoder",
    "VJEPAEncoder",
    "create_encoder",
    "CarDreamerEncoderHook",
    "FrameStacker",
    "patch_dreamerv3_encoder",
]
