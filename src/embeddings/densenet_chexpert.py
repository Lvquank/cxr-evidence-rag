"""DenseNet121 backbone loader for the trusted local CheXpert checkpoint."""

import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn as nn
from torchvision import models


def _positive_probability(label: str, logits: torch.Tensor, policies: Mapping[str, str]) -> torch.Tensor:
    if policies[label] == "u_multiclass":
        return torch.softmax(logits[:, :2], dim=1)[:, 1]
    return torch.sigmoid(logits.squeeze(1))


class CheXpertRelationAwareDenseNet121(nn.Module):
    """Checkpoint-compatible architecture; retrieval uses its 1024-D backbone feature."""

    def __init__(self, active_labels: Iterable[str], policies: Mapping[str, str],
                 relation_sources: Mapping[str, Iterable[str]], use_relations: bool,
                 relation_hidden: int):
        super().__init__()
        self.active_labels = list(active_labels)
        self.policies = dict(policies)
        backbone = models.densenet121(weights=None)
        self.in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.base_heads = nn.ModuleDict({
            label: nn.Linear(self.in_features, 3 if self.policies[label] == "u_multiclass" else 1)
            for label in self.active_labels
        })
        self.relation_sources = {}
        self.relation_heads = nn.ModuleDict()
        if use_relations:
            for target, sources in relation_sources.items():
                valid_sources = [source for source in sources if source in self.active_labels]
                if target not in self.active_labels or not valid_sources:
                    continue
                self.relation_sources[target] = valid_sources
                self.relation_heads[target] = nn.Sequential(
                    nn.Linear(self.in_features + 1 + len(valid_sources), relation_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(relation_hidden, 1),
                )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


def load_densenet_checkpoint(checkpoint_path: str | Path, device: torch.device):
    """Load a trusted local training checkpoint and return its feature extractor."""
    checkpoint_path = Path(checkpoint_path)
    # The legacy checkpoint has NumPy metadata, so safe weights-only loading cannot read it.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    hidden = next(
        (int(value.shape[0]) for name, value in state.items()
         if name.startswith("relation_heads.") and name.endswith(".0.weight")),
        256,
    )
    model = CheXpertRelationAwareDenseNet121(
        checkpoint["active_labels"], checkpoint["policies"], checkpoint["relation_sources"],
        bool(checkpoint["use_relations"]), hidden,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    # A content hash remains stable when the same checkpoint is moved from Kaggle
    # back to the local machine with different path and timestamp metadata.
    with checkpoint_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    revision = f"sha256:{digest}"
    return model, revision
