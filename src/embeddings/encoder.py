"""RAD-DINO-MAIRA-2 image embedding for chest X-ray retrieval."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as transforms_functional

from src.embeddings.densenet_chexpert import load_densenet_checkpoint
from src.embeddings.pathology_projection import load_pathology_components


RAD_DINO_MODEL_NAME = "microsoft/rad-dino-maira-2"
DENSENET_MODEL_NAME = "local/chexpert-relation-aware-densenet121"
DEFAULT_MODEL_NAME = RAD_DINO_MODEL_NAME
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_SIZE = 14
DEFAULT_DENSENET_CHECKPOINT = PROJECT_ROOT / "model" / "best_E2_relation_aware_weighted_bce.pt"


def build_patch_mask(
    spatial_sizes: list[tuple[int, int]],
    padded_height: int,
    padded_width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Mark real patch tokens, excluding padding introduced only to form a batch."""
    grid_height, grid_width = padded_height // PATCH_SIZE, padded_width // PATCH_SIZE
    mask = torch.zeros((len(spatial_sizes), grid_height, grid_width), dtype=torch.bool, device=device)
    for index, (height, width) in enumerate(spatial_sizes):
        mask[index, : height // PATCH_SIZE, : width // PATCH_SIZE] = True
    return mask.flatten(1)


def _hugging_face_token() -> Optional[str]:
    """Load the local .env once and return an optional Hugging Face token."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return os.getenv("HF_TOKEN") or None


@lru_cache(maxsize=2)
def load_image_processor(model_name: str = RAD_DINO_MODEL_NAME):
    """Load and cache the Hugging Face processor specified by the model card."""
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ImportError(
            "RAD-DINO-MAIRA-2 requires transformers. Install requirements.txt first."
        ) from exc
    return AutoImageProcessor.from_pretrained(model_name, token=_hugging_face_token())


class ImagePreprocessor:
    """Apply the preprocessing associated with the selected retrieval backbone."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.processor = None
        self.shortest_edge = None
        if model_name != DENSENET_MODEL_NAME:
            self.processor = load_image_processor(model_name)
            self.shortest_edge = int(self.processor.size["shortest_edge"])

    def _resize_shortest_edge(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        shortest_edge = min(width, height)
        if shortest_edge == self.shortest_edge:
            return image
        scale = self.shortest_edge / shortest_edge
        resized_size = (round(width * scale), round(height * scale))
        return image.resize(resized_size, resample=Image.Resampling(self.processor.resample))

    def __call__(self, pil_image: Image.Image) -> torch.Tensor:
        if self.model_name == DENSENET_MODEL_NAME:
            image = pil_image.convert("RGB").resize((320, 320), resample=Image.Resampling.BILINEAR)
            pixel_values = transforms_functional.to_tensor(image)
            return transforms_functional.normalize(
                pixel_values,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        image = self._resize_shortest_edge(pil_image.convert("RGB"))
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )["pixel_values"].squeeze(0)
        height, width = pixel_values.shape[-2:]
        pad_height = (-height) % PATCH_SIZE
        pad_width = (-width) % PATCH_SIZE
        if pad_height or pad_width:
            pixel_values = F.pad(pixel_values, (0, pad_width, 0, pad_height))
        return pixel_values

    def signature(self) -> dict:
        if self.model_name == DENSENET_MODEL_NAME:
            return {"policy": "densenet-rgb-320-imagenet-v1"}
        return {"policy": "rad-dino-full-field-round-resize-pad-v1",
                "processor": self.processor.to_dict(), "do_center_crop": False,
                "patch_size": PATCH_SIZE}


class ImageEncoder(nn.Module):
    """RAD-DINO or local DenseNet encoder producing normalized retrieval vectors."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        model_name: str = DEFAULT_MODEL_NAME,
        projection_checkpoint: Optional[Union[str, Path]] = None,
        densenet_checkpoint: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_name = model_name
        if model_name == DENSENET_MODEL_NAME:
            self.model, self.model_revision = load_densenet_checkpoint(
                densenet_checkpoint or DEFAULT_DENSENET_CHECKPOINT,
                self.device,
            )
            self.raw_embedding_dimension = self.model.in_features
        else:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "RAD-DINO-MAIRA-2 requires transformers. Install requirements.txt first."
                ) from exc
            self.model = AutoModel.from_pretrained(model_name, token=_hugging_face_token())
            self.raw_embedding_dimension = int(self.model.config.hidden_size)
            self.model_revision = getattr(self.model.config, "_commit_hash", None)
        self.projection = None
        self.feature_fusion = None
        self.projection_metadata = None
        self.to(self.device)
        if projection_checkpoint:
            (
                self.projection,
                self.feature_fusion,
                encoder_state,
                self.projection_metadata,
            ) = load_pathology_components(
                projection_checkpoint,
                expected_model_name=self.model_name,
                expected_input_dim=self.raw_embedding_dimension,
                device=self.device,
            )
            if encoder_state:
                missing, unexpected = self.model.load_state_dict(encoder_state, strict=False)
                if unexpected:
                    raise ValueError(f"Fine-tuned encoder checkpoint has unexpected keys: {unexpected}")
                # State contains only selected RAD-DINO modules, so unrelated base keys are expected.
            self.embedding_dimension = self.projection.output_dim
        else:
            self.embedding_dimension = self.raw_embedding_dimension

    def forward(self, pixel_values: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.model_name == DENSENET_MODEL_NAME:
            embeddings = self.model(pixel_values)
        else:
            outputs = self.model(pixel_values=pixel_values)
            hidden_states = outputs.last_hidden_state
            if self.feature_fusion is not None:
                if hidden_states.shape[1] < 2:
                    raise ValueError("CLS/local-patch fusion requires patch tokens from RAD-DINO.")
                embeddings = self.feature_fusion(hidden_states[:, 0], hidden_states[:, 1:], patch_mask)
            elif self.projection_metadata and self.projection_metadata["requires_image_encoder"]:
                # Fine-tuned checkpoints use the final-layer CLS token directly.
                embeddings = hidden_states[:, 0]
            else:
                embeddings = getattr(outputs, "pooler_output", None)
                if embeddings is None:
                    embeddings = hidden_states[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return self.projection(embeddings) if self.projection is not None else embeddings


def load_encoder(
    device: Union[str, torch.device] = "cpu",
    model_name: Optional[str] = None,
    projection_checkpoint: Optional[Union[str, Path]] = None,
    densenet_checkpoint: Optional[Union[str, Path]] = None,
) -> ImageEncoder:
    encoder = ImageEncoder(
        device=device,
        model_name=model_name or DEFAULT_MODEL_NAME,
        projection_checkpoint=projection_checkpoint,
        densenet_checkpoint=densenet_checkpoint,
    )
    encoder.eval()
    return encoder
