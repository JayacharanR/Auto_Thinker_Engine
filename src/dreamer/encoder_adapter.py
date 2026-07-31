"""
Encoder Adapter for Phase 3 three-way comparison.

A thin adapter that maps each encoder's output dimensionality to
the RSSM's expected input dimension. The adapter's capacity is
IDENTICAL across all three arms to avoid being a confound.

Arms:
- CNN: CarDreamer's default small CNN encoder
- Custom JEPA: Phase 2 ViT-Small encoder (frozen or fine-tuned)
- V-JEPA2: Meta's pretrained V-JEPA2 checkpoint (frozen or LoRA)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderAdapter(nn.Module):
    """
    Thin adapter matching any encoder's output to RSSM input dimension.

    This projection layer has IDENTICAL capacity across all arms:
    same target_dim, same activation, same layer norm. The only
    difference is input_dim (which varies by encoder architecture).

    Args:
        input_dim: Dimension of the encoder output.
        target_dim: Dimension expected by the RSSM (typically 1024).
        use_layer_norm: Whether to apply layer norm after projection.
        activation: Activation function after projection.
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int = 1024,
        use_layer_norm: bool = True,
        activation: str = "silu",
    ):
        super().__init__()

        self.projection = nn.Linear(input_dim, target_dim)

        self.layer_norm = nn.LayerNorm(target_dim) if use_layer_norm else nn.Identity()

        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project encoder output to RSSM input dimension.

        Args:
            x: (B, D_enc) or (B, N, D_enc) encoder output.
               If 3D, global average pools over patch dimension first.

        Returns:
            (B, D_target) projected representation.
        """
        # If input is (B, N, D) from ViT, pool over patches
        if x.dim() == 3:
            x = x.mean(dim=1)  # (B, D)

        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x


class CNNEncoder(nn.Module):
    """
    Default CNN encoder from DreamerV3/CarDreamer (Arm 1 baseline).

    Standard convolutional encoder with progressively increasing channels.
    This is the baseline — all Phase 3 improvements are measured against it.

    Args:
        in_channels: Number of input channels (3 for RGB).
        depth: Base channel depth (DreamerV3-small uses 48).
        kernels: Kernel sizes for each conv layer.
        stride: Stride for each conv layer.
        activation: Activation function.
        input_size: Spatial input resolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 48,
        kernels: list[int] = None,
        stride: int = 2,
        activation: str = "silu",
        input_size: int = 128,
    ):
        super().__init__()

        if kernels is None:
            kernels = [4, 4, 4, 4]

        act_fn = nn.SiLU() if activation == "silu" else nn.ReLU()

        layers = []
        in_ch = in_channels
        for i, kernel in enumerate(kernels):
            out_ch = depth * (2**i)
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
                    act_fn,
                ]
            )
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        # Compute output dimension
        self._output_dim = self._compute_output_dim(in_channels, input_size)

    def _compute_output_dim(self, in_channels: int, input_size: int) -> int:
        """Compute flattened output dimension from a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.conv(dummy)
            return out.reshape(1, -1).shape[1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.

        Returns:
            (B, D) flattened CNN features.
        """
        x = self.conv(x)
        return x.reshape(x.shape[0], -1)


class VJEPAEncoder(nn.Module):
    """
    Wrapper for Meta's V-JEPA2 pretrained encoder (Arm 3).

    Loads the ViT-L checkpoint from Hugging Face and extracts features.
    This encoder is used FROZEN (or with LoRA) — do NOT train it from
    scratch, that defeats the purpose of transfer learning.

    Args:
        model_name: Hugging Face model identifier.
        freeze: If True, all parameters are frozen (transfer learning).
        use_lora: If True, applies LoRA adapters for parameter-efficient
                  fine-tuning (only if freeze=False).
        lora_rank: LoRA rank (only used if use_lora=True).
        input_resolution: Target input resolution for the driving task.
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        freeze: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
        input_resolution: int = 224,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.input_resolution = input_resolution

        # Load from HuggingFace
        from transformers import AutoModel, AutoVideoProcessor

        # The processor handles normalization, resizing, and tensor layout
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="sdpa",  # Efficient attention
        )

        # Freeze if transfer learning
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # LoRA (optional fine-tuning)
        if use_lora and not freeze:
            self._apply_lora(lora_rank)

        # Output dimension (ViT-L = 1024)
        self._output_dim = self.model.config.hidden_size

    def _apply_lora(self, rank: int) -> None:
        """Apply LoRA adapters to attention layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] >= 1024:
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                in_feat = module.in_features
                out_feat = module.out_features
                module.lora_A = nn.Parameter(
                    torch.randn(in_feat, rank) * 0.01
                )
                module.lora_B = nn.Parameter(torch.zeros(rank, out_feat))

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video/image through V-JEPA2 using the official processor.

        Input tensor conventions (what WE receive):
            (B, C, T, H, W) — video clip from frame stacker
            (B, C, H, W)    — single frame (adds trivial temporal dim)

        V-JEPA2 expects (from its official processor):
            pixel_values: (B, T, C, H, W)

        The processor handles normalization (ImageNet mean/std).

        Returns:
            (B, N, D) patch features (pooled by the adapter layer).
        """
        device = x.device

        # Normalize input to [0, 1] if needed
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Handle single-frame input: (B, C, H, W) → (B, C, 1, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(2)

        B, C, T, H, W = x.shape

        # Rearrange from our (B, C, T, H, W) to V-JEPA2's (B, T, C, H, W)
        x_video = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # Apply processor normalization
        # The processor expects pixel values in [0, 1] or [0, 255] and applies
        # ImageNet normalization internally. We use it for normalization only,
        # since we already have tensors (not PIL images).
        #
        # For efficiency with batched tensors, we apply the processor's
        # normalization manually rather than going through the full processor
        # pipeline (which expects lists of numpy arrays).
        if hasattr(self.processor, 'image_mean') and hasattr(self.processor, 'image_std'):
            mean = torch.tensor(self.processor.image_mean, device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor(self.processor.image_std, device=device).view(1, 1, 3, 1, 1)
            x_video = (x_video - mean) / std

        # Resize to model's expected resolution if needed
        if H != self.input_resolution or W != self.input_resolution:
            x_flat = x_video.reshape(B * T, C, H, W)
            x_flat = F.interpolate(
                x_flat,
                size=(self.input_resolution, self.input_resolution),
                mode="bilinear",
                align_corners=False,
            )
            x_video = x_flat.reshape(B, T, C, self.input_resolution, self.input_resolution)

        # Forward through V-JEPA2
        # pixel_values shape: (B, T, C, H, W) — the official contract
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(pixel_values=x_video)
        else:
            outputs = self.model(pixel_values=x_video)

        # Extract features — use last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state  # (B, N, D)
        elif isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs

        return features


def create_encoder(arm: str, config: dict, device: str = "cuda") -> tuple[nn.Module, EncoderAdapter]:
    """
    Create encoder + adapter for a Phase 3 arm.

    Args:
        arm: One of 'cnn', 'custom_jepa', 'vjepa2'.
        config: Phase 3 config dict.
        device: Torch device.

    Returns:
        Tuple of (encoder, adapter).
    """
    encoder_configs = config["encoders"]
    adapter_config = config["adapter"]

    if arm == "cnn":
        enc_cfg = encoder_configs["cnn"]
        encoder = CNNEncoder(
            depth=enc_cfg.get("depth", 48),
            kernels=enc_cfg.get("kernels", [4, 4, 4, 4]),
            stride=enc_cfg.get("stride", 2),
            activation=enc_cfg.get("activation", "silu"),
            input_size=config["environment"]["observation"]["camera"]["width"],
        )
        input_dim = encoder.output_dim

    elif arm == "custom_jepa":
        enc_cfg = encoder_configs["custom_jepa"]
        from src.jepa.encoder import ViTEncoder

        encoder = ViTEncoder(
            img_size=enc_cfg.get("input_resolution", 128),
            patch_size=enc_cfg.get("patch_size", 16),
            embed_dim=enc_cfg.get("embed_dim", 384),
            depth=enc_cfg.get("depth", 12),
            num_heads=enc_cfg.get("num_heads", 6),
        )
        # Load pretrained weights from Phase 2
        checkpoint_path = enc_cfg.get("checkpoint")
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
            elif "context_encoder_state_dict" in ckpt:
                encoder.load_state_dict(ckpt["context_encoder_state_dict"], strict=False)

        if enc_cfg.get("freeze", True):
            for param in encoder.parameters():
                param.requires_grad = False

        input_dim = enc_cfg.get("output_dim", 384)

    elif arm == "vjepa2":
        enc_cfg = encoder_configs["vjepa2"]
        encoder = VJEPAEncoder(
            model_name=enc_cfg.get("model_name", "facebook/vjepa2-vitl-fpc64-256"),
            freeze=enc_cfg.get("freeze", True),
            use_lora=enc_cfg.get("use_lora", False),
            lora_rank=enc_cfg.get("lora_rank", 8),
            input_resolution=enc_cfg.get("input_resolution", 128),
        )
        input_dim = encoder.output_dim

    else:
        raise ValueError(f"Unknown arm: {arm}. Must be one of: cnn, custom_jepa, vjepa2")

    # Create adapter with IDENTICAL capacity across all arms
    adapter = EncoderAdapter(
        input_dim=input_dim,
        target_dim=adapter_config.get("target_dim", 1024),
        use_layer_norm=adapter_config.get("use_layer_norm", True),
        activation=adapter_config.get("activation", "silu"),
    )

    encoder = encoder.to(device)
    adapter = adapter.to(device)

    return encoder, adapter
