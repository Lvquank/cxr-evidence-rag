"""Pathology-aware RAD-DINO retrieval heads and checkpoint helpers."""

from pathlib import Path
from typing import Mapping
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTION_SCHEMA_VERSION = 2


def _checkpoint_sha256(checkpoint_path: Path) -> str:
    """Return a stable identity for portable projection artifacts."""
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PathologyProjectionHead(nn.Module):
    """Encoder feature -> 512-D L2-normalized retrieval embedding."""

    def __init__(self, input_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(features), p=2, dim=1)


class ClsPatchFusion(nn.Module):
    """Fuse the global CLS token with the mean local patch representation."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patch_mask is None:
            local_mean = patch_tokens.mean(dim=1)
        else:
            if patch_mask.shape != patch_tokens.shape[:2]:
                raise ValueError("Patch mask does not match RAD-DINO patch-token dimensions.")
            weights = patch_mask.to(dtype=patch_tokens.dtype).unsqueeze(-1)
            local_mean = (patch_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        # Keep the global representation available while learning local corrections.
        return cls_token + self.layers(torch.cat((cls_token, local_mean), dim=1))


def _read_checkpoint(checkpoint_path: str | Path) -> tuple[dict, Path]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"base_model", "input_dim", "output_dim", "projection_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Projection checkpoint is missing fields: {sorted(missing)}")
    schema_version = int(checkpoint.get("schema_version", 1))
    if schema_version not in (1, PROJECTION_SCHEMA_VERSION):
        raise ValueError("Unsupported pathology projection checkpoint schema.")
    return checkpoint, checkpoint_path


def load_pathology_components(
    checkpoint_path: str | Path,
    expected_model_name: str,
    expected_input_dim: int,
    device: torch.device,
) -> tuple[PathologyProjectionHead, ClsPatchFusion | None, dict, dict]:
    """Load projection plus optional fine-tuned encoder and feature-fusion state."""
    checkpoint, checkpoint_path = _read_checkpoint(checkpoint_path)
    if checkpoint["base_model"] != expected_model_name:
        raise ValueError("Projection checkpoint was trained with a different base model.")
    if int(checkpoint["input_dim"]) != expected_input_dim:
        raise ValueError("Projection checkpoint input dimension does not match the base encoder.")

    projection = PathologyProjectionHead(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
    )
    projection.load_state_dict(checkpoint["projection_state_dict"], strict=True)

    pipeline = checkpoint.get("pipeline", checkpoint.get("recipe", "legacy_projection"))
    fusion = None
    fusion_state = checkpoint.get("feature_fusion_state_dict")
    if fusion_state is not None:
        fusion = ClsPatchFusion(hidden_size=int(checkpoint["input_dim"]))
        fusion.load_state_dict(fusion_state, strict=True)

    metadata = {
        "path": str(checkpoint_path.resolve()),
        "file_name": checkpoint_path.name,
        "sha256": _checkpoint_sha256(checkpoint_path),
        "output_dim": int(checkpoint["output_dim"]),
        "pipeline": pipeline,
        "requires_image_encoder": bool(checkpoint.get("encoder_trainable_state_dict") or fusion),
        "feature_fusion": "cls_mean_patch" if fusion else None,
        "training": checkpoint.get("training", {}),
    }
    return projection.to(device).eval(), (fusion.to(device).eval() if fusion else None), checkpoint.get(
        "encoder_trainable_state_dict", {}
    ), metadata


def load_projection_checkpoint(
    checkpoint_path: str | Path,
    expected_model_name: str,
    expected_input_dim: int,
    device: torch.device,
) -> tuple[PathologyProjectionHead, dict]:
    projection, fusion, encoder_state, metadata = load_pathology_components(
        checkpoint_path,
        expected_model_name=expected_model_name,
        expected_input_dim=expected_input_dim,
        device=device,
    )
    if fusion is not None or encoder_state:
        raise ValueError(
            "This checkpoint modifies RAD-DINO features and cannot be applied to cached raw embeddings. "
            "Re-embed images with scripts/compute_image_embeddings.py."
        )
    return projection, metadata


def weighted_jaccard(support: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Pairwise weighted Jaccard relevance for non-negative multi-label support."""
    minimum = torch.minimum(support[:, None, :], support[None, :, :]).sum(dim=-1)
    maximum = torch.maximum(support[:, None, :], support[None, :, :]).sum(dim=-1)
    return minimum / maximum.clamp_min(epsilon)
