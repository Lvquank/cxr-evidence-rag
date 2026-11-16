"""Predict six CheXpert findings with the local relation-aware DenseNet checkpoint.

Example:
  D:/Miniconda3/python.exe scripts/predict_chexpert_6labels.py --image_path "E:/view1_frontal.jpg"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "best_E2_relation_aware_weighted_bce.pt"
MAIN_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def positive_probability_from_logits(
    label: str, logits: torch.Tensor, policies: Mapping[str, str]
) -> torch.Tensor:
    if policies[label] == "u_multiclass":
        # The checkpoint uses [negative, positive, uncertain] logits.
        return torch.softmax(logits[:, :2], dim=1)[:, 1]
    return torch.sigmoid(logits.squeeze(1))


class RelationAwareDenseNet121(nn.Module):
    """Inference counterpart of the architecture defined in the training notebook."""

    def __init__(
        self,
        active_labels: Iterable[str],
        policies: Mapping[str, str],
        relation_sources: Mapping[str, Iterable[str]],
        use_relations: bool,
        relation_hidden: int,
    ):
        super().__init__()
        self.active_labels = list(active_labels)
        self.policies = dict(policies)
        self.use_relations = use_relations

        backbone = models.densenet121(weights=None)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.base_heads = nn.ModuleDict(
            {
                label: nn.Linear(
                    in_features,
                    3 if self.policies[label] == "u_multiclass" else 1,
                )
                for label in self.active_labels
            }
        )

        self.relation_sources: Dict[str, list[str]] = {}
        self.relation_heads = nn.ModuleDict()
        if use_relations:
            for target, sources in relation_sources.items():
                valid_sources = [source for source in sources if source in self.active_labels]
                if target not in self.active_labels or not valid_sources:
                    continue
                self.relation_sources[target] = valid_sources
                relation_dim = in_features + 1 + len(valid_sources)
                self.relation_heads[target] = nn.Sequential(
                    nn.Linear(relation_dim, relation_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(relation_hidden, 1),
                )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(image)
        base_outputs = {
            label: self.base_heads[label](features) for label in self.active_labels
        }
        outputs = dict(base_outputs)

        for target, sources in self.relation_sources.items():
            source_probabilities = [
                positive_probability_from_logits(source, base_outputs[source], self.policies).unsqueeze(1)
                for source in sources
            ]
            target_logit = base_outputs[target].squeeze(1)
            relation_input = torch.cat(
                [features, target_logit.unsqueeze(1), *source_probabilities], dim=1
            )
            outputs[target] = (
                target_logit + self.relation_heads[target](relation_input).squeeze(1)
            ).unsqueeze(1)

        return outputs


def _relation_hidden_size(state_dict: Mapping[str, torch.Tensor]) -> int:
    for name, value in state_dict.items():
        if name.startswith("relation_heads.") and name.endswith(".0.weight"):
            return int(value.shape[0])
    return 256


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[RelationAwareDenseNet121, dict]:
    # This checkpoint contains training metadata that needs legacy pickle loading.
    # Only use checkpoints from a trusted source.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "active_labels", "policies", "relation_sources", "use_relations"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    missing_main = set(MAIN_LABELS).difference(checkpoint["active_labels"])
    if missing_main:
        raise ValueError(f"Checkpoint is missing main labels: {sorted(missing_main)}")

    model = RelationAwareDenseNet121(
        active_labels=checkpoint["active_labels"],
        policies=checkpoint["policies"],
        relation_sources=checkpoint["relation_sources"],
        use_relations=bool(checkpoint["use_relations"]),
        relation_hidden=_relation_hidden_size(checkpoint["model_state_dict"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def predict(image_path: Path, model: RelationAwareDenseNet121, device: torch.device) -> Dict[str, float]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = {
            label: float(positive_probability_from_logits(label, outputs[label], model.policies)[0])
            for label in MAIN_LABELS
        }
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict six CheXpert label probabilities.")
    parser.add_argument("--image_path", required=True, help="Path to a chest X-ray image")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to a trusted relation-aware DenseNet checkpoint",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output_json", help="Optional path for writing the probability JSON")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    checkpoint_path = Path(args.checkpoint)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    model, _ = load_model(checkpoint_path, device)
    probabilities = predict(image_path, model, device)
    result = {label: round(probabilities[label], 6) for label in MAIN_LABELS}
    print(json.dumps(result, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
