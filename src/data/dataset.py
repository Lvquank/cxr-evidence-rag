from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Dataset that optionally crops, resizes, and applies a supplied image transform.

    - Expects metadata: list of dicts with at least `id` and `image_path` (relative to processed_dir)
    - If metadata row contains 'bbox' as "x1,y1,x2,y2" or list/tuple, the bbox will be applied before resize.
    - Set ``resize=None`` when a model-owned processor performs resizing itself.
    """

    def __init__(self, processed_dir: Path, metadata: List[Dict],
                 resize=(320, 320),
                 train: bool = False,
                 use_augmentation: bool = False,
                 image_transform=None,
                 center_crop: bool = False):
        self.processed_dir = Path(processed_dir)
        self.metadata = metadata
        self.resize = resize
        self.train = train
        self.use_augmentation = use_augmentation
        self.image_transform = image_transform
        self.center_crop = center_crop

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        train_ops = []
        if self.use_augmentation and self.train:
            train_ops.extend([
                transforms.RandomAffine(degrees=4, translate=(0.015, 0.015), scale=(0.98, 1.02)),
                transforms.ColorJitter(brightness=0.04, contrast=0.06),
            ])
        train_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.train_transform = transforms.Compose(train_ops)
        self.eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.metadata)

    def _apply_crop(self, img: Image.Image, meta: Dict) -> Image.Image:
        # bbox can be provided as list/tuple or comma-separated string
        bbox = meta.get('bbox') or meta.get('crop_box')
        if bbox:
            if isinstance(bbox, str):
                parts = [float(x) for x in bbox.split(',')]
                if len(parts) == 4:
                    x1, y1, x2, y2 = parts
                else:
                    return img
            else:
                x1, y1, x2, y2 = bbox
            try:
                return img.crop((x1, y1, x2, y2))
            except Exception:
                return img
        if not self.center_crop:
            return img

        # Optional legacy behavior. RAD-DINO uses aspect-ratio-preserving resize.
        w, h = img.size
        if w == h:
            return img
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))

    def __getitem__(self, idx):
        row = self.metadata[idx]
        image_path = row.get('path', row.get('image_path'))
        if not image_path:
            raise KeyError("Metadata row requires 'path' or legacy 'image_path'.")
        img_path = self.processed_dir / image_path
        img = Image.open(img_path).convert('RGB')

        # apply crop (center crop fallback)
        img = self._apply_crop(img, row)

        # Resize before transforms unless a model-provided processor owns resizing.
        if self.resize is not None:
            img = img.resize(self.resize)

        if self.image_transform is not None:
            return self.image_transform(img), row

        transform = self.train_transform if (self.train and self.use_augmentation) else self.eval_transform
        tensor = transform(img)

        return tensor, row
