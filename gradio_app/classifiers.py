"""Local classifier adapters; research weights and retrieval stay frozen."""
from pathlib import Path

DEFAULT_CLASSIFIER = "densenet121"
CLASSIFIERS = {"densenet121": "DenseNet121", "resnet50": "ResNet50"}


def classifier_checkpoint(name, research_root):
    if name == "densenet121":
        return Path(research_root) / "model/best_E2_relation_aware_weighted_bce.pt"
    if name == "resnet50":
        return Path(__file__).resolve().parent / "models/best_resnet50_E2_relation_aware_weighted_bce.pt"
    raise ValueError("Model phân loại không hợp lệ.")


def load_classifier(name, checkpoint_path, device):
    from scripts.predict_chexpert_6labels import load_model
    if name == DEFAULT_CLASSIFIER:
        return load_model(checkpoint_path, device)
    if name != "resnet50":
        raise ValueError("Model phân loại không hợp lệ.")

    import torch
    from torch import nn
    from torchvision.models import resnet50
    from scripts.predict_chexpert_6labels import MAIN_LABELS, RelationAwareDenseNet121, _relation_hidden_size

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "active_labels", "policies", "relation_sources", "use_relations"}
    if required.difference(checkpoint) or set(MAIN_LABELS).difference(checkpoint["active_labels"]):
        raise ValueError("Checkpoint ResNet thiếu metadata hoặc nhãn nghiên cứu.")

    class RelationAwareResNet50(RelationAwareDenseNet121):
        # Reuse the research relation-aware forward and probability conversion.
        def __init__(self):
            nn.Module.__init__(self)
            self.active_labels = list(checkpoint["active_labels"])
            self.policies = dict(checkpoint["policies"])
            self.use_relations = bool(checkpoint["use_relations"])
            self.backbone = resnet50(weights=None)
            size = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.base_heads = nn.ModuleDict({
                label: nn.Linear(size, 3 if self.policies[label] == "u_multiclass" else 1)
                for label in self.active_labels})
            self.relation_sources = {}
            self.relation_heads = nn.ModuleDict()
            if self.use_relations:
                hidden = _relation_hidden_size(checkpoint["model_state_dict"])
                for target, sources in checkpoint["relation_sources"].items():
                    sources = [label for label in sources if label in self.active_labels]
                    if target not in self.active_labels or not sources:
                        continue
                    self.relation_sources[target] = sources
                    self.relation_heads[target] = nn.Sequential(
                        nn.Linear(size + 1 + len(sources), hidden), nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2), nn.Linear(hidden, 1))

    model = RelationAwareResNet50()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


class _ClassifierWithFrozenReranker:
    def __init__(self, classifier, backbone):
        self.classifier = classifier
        self.policies = classifier.policies
        self.backbone = backbone

    def __call__(self, image):
        return self.classifier(image)


def load_retrieval_classifier(name, checkpoint_path, research_root, device):
    classifier, metadata = load_classifier(name, checkpoint_path, device)
    if name == DEFAULT_CLASSIFIER:
        return classifier, metadata
    from scripts.predict_chexpert_6labels import load_model
    reranker, _ = load_model(classifier_checkpoint(DEFAULT_CLASSIFIER, research_root), device)
    return _ClassifierWithFrozenReranker(classifier, reranker.backbone), metadata
