"""Train the unified pathology-aware RAD-DINO retrieval pipeline.

The pipeline combines a 512-D projection head, graded pathology supervision,
balanced sampling, online hard-negative mining, fine-tuning of the last two
RAD-DINO blocks, and CLS/local-patch fusion.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ImageDataset
from src.embeddings.encoder import (
    RAD_DINO_MODEL_NAME,
    ImagePreprocessor,
    _hugging_face_token,
    build_patch_mask,
)
from src.embeddings.pathology_projection import (
    PROJECTION_SCHEMA_VERSION,
    ClsPatchFusion,
    PathologyProjectionHead,
    weighted_jaccard,
)
from src.retriever.pathology_support import CHEXPERT_LABELS


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collate_rows(batch):
    tensors, rows = zip(*batch)
    max_height = max(tensor.shape[-2] for tensor in tensors)
    max_width = max(tensor.shape[-1] for tensor in tensors)
    padded = [F.pad(tensor, (0, max_width - tensor.shape[-1], 0, max_height - tensor.shape[-2])) for tensor in tensors]
    spatial_sizes = [(tensor.shape[-2], tensor.shape[-1]) for tensor in tensors]
    return torch.stack(padded), list(rows), spatial_sizes


class BalancedPathologyBatchSampler(Sampler[list[int]]):
    """Sample each CheXpert support class twice before filling a mini-batch."""

    def __init__(self, support: np.ndarray, batch_size: int, batches: int, seed: int):
        if batch_size < len(CHEXPERT_LABELS) * 2:
            raise ValueError("--batch_size must be at least 12 for balanced pathology sampling.")
        self.support = support
        self.batch_size = batch_size
        self.batches = batches
        self.seed = seed
        self.class_members = [np.flatnonzero(support[:, label] > 0) for label in range(support.shape[1])]
        if any(len(members) == 0 for members in self.class_members):
            raise ValueError("Balanced sampling requires at least one eligible row for every pathology label.")

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        all_indices = np.arange(len(self.support))
        for _ in range(self.batches):
            selected = []
            for members in self.class_members:
                selected.extend(rng.choice(members, size=2, replace=len(members) < 2).tolist())
            selected.extend(rng.choice(all_indices, size=self.batch_size - len(selected), replace=False).tolist())
            rng.shuffle(selected)
            yield selected

    def __len__(self) -> int:
        return self.batches


def select_last_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Find the encoder block list across DINO model wrapper versions."""
    candidates = (
        "encoder.layer", "encoder.layers", "dinov2.encoder.layer", "dinov2.encoder.layers",
        "vision_model.encoder.layer", "vision_model.encoder.layers",
    )
    for path in candidates:
        module = model
        try:
            for name in path.split("."):
                module = getattr(module, name)
        except AttributeError:
            continue
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) >= 2:
            return list(module[-2:])
    raise RuntimeError("Could not locate RAD-DINO transformer blocks for fine-tuning.")


def balanced_graded_loss(
    embeddings: torch.Tensor,
    support: torch.Tensor,
    structure_weight: float,
    negative_margin: float,
    graded_power: float,
    hard_negative_ratio: float,
    raw_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Graded pathology loss with online hard-negative selection."""
    similarity = embeddings @ embeddings.T
    relevance = weighted_jaccard(support)
    off_diagonal = ~torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive_mask = (relevance > 0) & off_diagonal
    negative_mask = (relevance == 0) & off_diagonal
    zero = similarity.new_zeros(())

    if positive_mask.any():
        weights = relevance[positive_mask].pow(graded_power)
        positive_loss = (weights * (similarity[positive_mask] - relevance[positive_mask]).square()).sum() / weights.sum().clamp_min(1e-8)
    else:
        positive_loss = zero
    negatives = similarity[negative_mask]
    if negatives.numel():
        hard_count = min(negatives.numel(), max(1, int(math.ceil(positive_mask.sum().item() * hard_negative_ratio))))
        negatives = torch.topk(negatives, k=hard_count).values
    negative_loss = F.relu(negatives - negative_margin).square().mean() if negatives.numel() else zero

    structure_loss = zero
    if raw_features is not None and structure_weight:
        raw = F.normalize(raw_features.detach(), p=2, dim=1)
        structure_loss = F.mse_loss(similarity[off_diagonal], (raw @ raw.T)[off_diagonal])
    loss = positive_loss + negative_loss + structure_weight * structure_loss
    return loss, {
        "positive": float(positive_loss.detach()),
        "negative": float(negative_loss.detach()),
        "structure": float(structure_loss.detach()),
        "positive_pairs": int(positive_mask.sum().item() // 2),
        "negative_pairs": int(negative_mask.sum().item() // 2),
    }


def train_image_encoder(args, rows: list[dict], support: np.ndarray, device: torch.device):
    if args.processed_dir is None:
        raise ValueError("--processed_dir is required to fine-tune RAD-DINO from images.")
    if device.type != "cuda":
        print("Warning: training on CPU will be extremely slow; use a CUDA GPU for practical training.")
    eligible = support.sum(axis=1) > 0
    train_rows = [row for row, keep in zip(rows, eligible) if keep]
    train_support = support[eligible]
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("The unified pipeline requires transformers from requirements.txt.") from exc

    encoder = AutoModel.from_pretrained(RAD_DINO_MODEL_NAME, token=_hugging_face_token()).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    trainable_blocks = select_last_transformer_blocks(encoder)
    for block in trainable_blocks:
        for parameter in block.parameters():
            parameter.requires_grad = True
    hidden_size = int(encoder.config.hidden_size)
    fusion = ClsPatchFusion(hidden_size).to(device)
    head = PathologyProjectionHead(input_dim=hidden_size).to(device)
    head_parameters = list(head.parameters()) + list(fusion.parameters())
    encoder_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.lr},
            {"params": encoder_parameters, "lr": args.encoder_lr},
        ],
        weight_decay=1e-5,
    )

    dataset = ImageDataset(
        Path(args.processed_dir), train_rows, resize=None, train=False,
        use_augmentation=False, image_transform=ImagePreprocessor(RAD_DINO_MODEL_NAME),
    )
    batches = args.batches_per_epoch or math.ceil(len(dataset) / args.batch_size)
    last_stats = {}
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        fusion.train()
        sampler = BalancedPathologyBatchSampler(train_support, args.batch_size, batches, args.seed + epoch)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_rows)
        losses = []
        for pixels, batch_rows, spatial_sizes in loader:
            batch_support = torch.tensor([row["support"] for row in batch_rows], dtype=torch.float32, device=device)
            hidden_states = encoder(pixel_values=pixels.to(device)).last_hidden_state
            cls_token = hidden_states[:, 0]
            patch_mask = build_patch_mask(spatial_sizes, pixels.shape[-2], pixels.shape[-1], device)
            features = fusion(cls_token, hidden_states[:, 1:], patch_mask)
            projected = head(F.normalize(features, p=2, dim=1))
            loss, last_stats = balanced_graded_loss(
                projected, batch_support, args.structure_weight,
                args.negative_margin, args.graded_power, args.hard_negative_ratio, cls_token,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={sum(losses) / max(1, len(losses)):.5f} " + " ".join(f"{key}={value}" for key, value in last_stats.items()))

    block_ids = {id(block) for block in trainable_blocks}
    block_prefixes = [name for name, module in encoder.named_modules() if id(module) in block_ids]
    encoder_state = {
        name: value.cpu() for name, value in encoder.state_dict().items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in block_prefixes)
    }
    return head.cpu().eval(), fusion.cpu().eval(), encoder_state, int(eligible.sum()), last_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unified pathology-aware RAD-DINO retrieval pipeline.")
    parser.add_argument("--embeddings_path", default="data/embeddings/embeddings.npy")
    parser.add_argument("--metadata_path", default="data/embeddings/metadata.jsonl")
    parser.add_argument("--processed_dir", required=True, help="Processed image directory used for RAD-DINO fine-tuning.")
    parser.add_argument("--output_checkpoint", default="model/pathology_projection_rad_dino.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--batches_per_epoch", type=int, help="Batches per epoch; default covers eligible rows once.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="Learning rate for the final two RAD-DINO blocks.")
    parser.add_argument("--structure_weight", type=float, default=0.20)
    parser.add_argument("--negative_margin", type=float, default=0.10)
    parser.add_argument("--graded_power", type=float, default=2.0)
    parser.add_argument("--hard_negative_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", help="Override base model; defaults to embedding_manifest.json.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings_path)
    embeddings = np.load(embeddings_path).astype(np.float32)
    rows = read_jsonl(Path(args.metadata_path))
    if len(embeddings) != len(rows) or embeddings.ndim != 2:
        raise ValueError("Embeddings and metadata must have matching row counts.")
    manifest_path = embeddings_path.parent / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if manifest.get("projection"):
        raise ValueError("Train from raw, unprojected RAD-DINO embeddings.")
    base_model = args.base_model or manifest.get("embedding_model") or RAD_DINO_MODEL_NAME
    if base_model != RAD_DINO_MODEL_NAME:
        raise ValueError("The unified pipeline requires microsoft/rad-dino-maira-2 raw embeddings.")
    support = np.asarray([row.get("support", []) for row in rows], dtype=np.float32)
    if support.shape != (len(rows), len(CHEXPERT_LABELS)):
        raise ValueError("Each metadata row must contain the compact six-value support list.")
    if int((support.sum(axis=1) > 0).sum()) < 2:
        raise ValueError("At least two rows with pathology support are required.")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    torch.manual_seed(args.seed)
    head, fusion, encoder_state, eligible_rows, stats = train_image_encoder(args, rows, support, device)

    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "base_model": base_model,
            "input_dim": int(embeddings.shape[1]),
            "output_dim": 512,
            "pipeline": "rad_dino_pathology_aware",
            "projection_state_dict": head.state_dict(),
            "feature_fusion_state_dict": fusion.state_dict() if fusion else None,
            "encoder_trainable_state_dict": encoder_state,
            "training": {
                "eligible_rows": eligible_rows, "epochs": args.epochs, "lr": args.lr,
                "encoder_lr": args.encoder_lr,
                "batch_size": args.batch_size, "structure_weight": args.structure_weight,
                "negative_margin": args.negative_margin, "graded_power": args.graded_power,
                "hard_negative_ratio": args.hard_negative_ratio,
                "balanced_sampling": True,
                "hard_negative_mining": True,
                "fine_tuned_blocks": 2,
                "feature_fusion": "cls_mean_patch",
                "last_epoch": stats,
            },
        }, output_path,
    )
    print(f"Saved unified 512-D RAD-DINO checkpoint to {output_path}")


if __name__ == "__main__":
    main()
