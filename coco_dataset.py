"""
COCO Dataset Loading and Preprocessing

This module contains utilities for loading and preprocessing COCO2017 dataset:
- Dataset loading functions
- Image transformations (train and validation)
- Custom collate functions for batching
- COCO class definitions and ID mappings
"""

import os

import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader


# COCO class names (80 classes + background)
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Mapping from original COCO IDs to continuous IDs (1-90 -> 1-80)
# COCO has 80 valid classes, but IDs are not continuous (some IDs are missing)
COCO_ID_TO_CONTINUOUS_ID = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20,
    22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
    35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
    46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50,
    56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60,
    67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70,
    80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
}

# Mapping from continuous IDs back to original COCO IDs
CONTINUOUS_ID_TO_COCO_ID = {v: k for k, v in COCO_ID_TO_CONTINUOUS_ID.items()}

# Mapping model output IDs to COCO continuous IDs (91 outputs -> 80 valid classes)
# Model outputs 91 classes (0-90), but only 80 are valid COCO classes
MODEL_OUTPUT_TO_COCO_ID = {}
MODEL_OUTPUT_TO_COCO_ID[0] = 0  # background

# Populate valid COCO classes
for coco_id, continuous_id in COCO_ID_TO_CONTINUOUS_ID.items():
    MODEL_OUTPUT_TO_COCO_ID[coco_id] = continuous_id

# Invalid COCO IDs map to background
# These IDs don't exist in COCO dataset
non_existent_ids = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
for invalid_id in non_existent_ids:
    MODEL_OUTPUT_TO_COCO_ID[invalid_id] = 0

# The 80 category_ids that actually exist in COCO. Used to drop predictions that
# land on one of the 10 gaps in the 1..90 id range.
VALID_COCO_CATEGORY_IDS = frozenset(COCO_ID_TO_CONTINUOUS_ID.keys())

# The one num_classes value this project is allowed to use.
#
# Targets are raw COCO category_ids, so the head must have a column for the
# LARGEST id (90) - hence 90 is the smallest value that avoids an IndexError in
# cross_entropy. But 90 is NOT a safe choice: it produces a 91-column cls_head,
# while model_loader.py and eval_coco.py rebuild the model at 91 (92 columns),
# and load_state_dict then fails with
#     size mismatch for bias: copying a param with shape [91] ...
# Anything other than exactly this value produces a checkpoint some part of the
# project cannot load, so every builder imports it from here rather than
# hardcoding a literal.
COCO_NUM_CLASSES = 91

# Class index used for "no object". The model outputs num_classes+1 logits where
# index 0 is background and indices 1..90 are raw COCO category_ids, so the
# background column is the FIRST one, not the last. Every consumer of the class
# logits must agree with DetectionLoss on this.
BACKGROUND_CLASS_INDEX = 0


# Cache of foreground validity masks, keyed by (num_foreground_columns, device,
# dtype). Rebuilding the mask per call would put a host-to-device copy inside the
# inference loop.
_FOREGROUND_MASK_CACHE = {}


def _foreground_validity_mask(num_fg, device, dtype):
    """Boolean mask over foreground columns: True where the column is a real COCO id."""
    key = (num_fg, device, dtype)
    mask = _FOREGROUND_MASK_CACHE.get(key)
    if mask is None:
        ids = torch.arange(1, num_fg + 1)
        mask = torch.tensor([int(i) in VALID_COCO_CATEGORY_IDS for i in ids],
                            device=device, dtype=torch.bool)
        _FOREGROUND_MASK_CACHE[key] = mask
    return mask


def decode_class_predictions(pred_logits):
    """
    Convert raw classification logits into (scores, COCO category ids).

    Background is index 0, so the foreground candidates are columns 1..N.
    Slicing with [..., :-1] would instead drop the never-trained last column and
    leave background in the argmax, which silently suppresses most detections.

    Columns that do not correspond to a real COCO id are masked out before the
    argmax. With num_classes=91 there are 11 such columns - the 10 gaps in
    1..90 plus column 91, which exists only because the head is sized
    num_classes+1 - and none of them is ever a training target. They are pushed
    down by cross_entropy but can still win an argmax, and a query whose argmax
    lands on one is discarded outright: eval_coco.py filters it against
    VALID_COCO_CATEGORY_IDS and calculate_validation_metrics can never match its
    class. Masking makes that query fall back to its best VALID class instead of
    being thrown away.

    The mask is applied to probabilities rather than logits so the returned score
    stays the true softmax probability of the chosen class - zeroing a column
    after softmax removes it from the argmax without redistributing its mass onto
    the survivors.

    Args:
        pred_logits: Tensor [..., num_classes + 1] of raw (pre-softmax) logits

    Returns:
        scores: Tensor [...] best foreground probability
        labels: Tensor [...] corresponding COCO category_id (1-based)
    """
    prob = pred_logits.softmax(-1)
    fg = prob[..., 1:]
    valid = _foreground_validity_mask(fg.shape[-1], fg.device, fg.dtype)
    fg = fg.masked_fill(~valid, 0.0)
    scores, labels = fg.max(-1)
    return scores, labels + 1


# Default input resolution. 800 -> 50×50 patches with ViT-B/16.
DEFAULT_INPUT_RESOLUTION = 800

# The value is mirrored into an environment variable by set_input_resolution()
# so that DataLoader workers, which re-import this module when Windows spawns
# them, end up with the same resolution as the parent process.
INPUT_RESOLUTION = int(os.environ.get('DINOV3_INPUT_RESOLUTION', DEFAULT_INPUT_RESOLUTION))


def _build_transform(resolution):
    """Geometry pipeline shared by training and validation."""
    # Use BILINEAR interpolation to accelerate resize (2-3x faster than BICUBIC)
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            (resolution, resolution),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])


# Move transform and collate_fn to global scope to solve Windows multiprocessing pickle error
global_train_transform = _build_transform(INPUT_RESOLUTION)
# No data augmentation for validation
global_val_transform = _build_transform(INPUT_RESOLUTION)


def set_input_resolution(resolution):
    """
    Change the input resolution used by every transform in this module.

    Must be called BEFORE the DataLoaders are constructed. Previously the
    resolution was hardcoded here, so the `input_resolution` knob in train.py
    had no effect whatsoever.
    """
    global INPUT_RESOLUTION, global_train_transform, global_val_transform
    INPUT_RESOLUTION = int(resolution)
    os.environ['DINOV3_INPUT_RESOLUTION'] = str(INPUT_RESOLUTION)
    global_train_transform = _build_transform(INPUT_RESOLUTION)
    global_val_transform = _build_transform(INPUT_RESOLUTION)
    return INPUT_RESOLUTION

# Photometric augmentation, training only. Applied to the PIL image before the
# geometry pipeline; it does not move any box so targets are unaffected.
#
# hue is deliberately 0. Measured per 640x480 image:
#   brightness 3.9ms | contrast 4.7ms | saturation 3.7ms | HUE 10.7ms
#   resize 4.3ms | ToTensor+Normalize 8.9ms
# Hue costs more than the entire geometry pipeline because it round-trips
# RGB->HSV->RGB, and it is the weakest of the four for detection. Dropping it
# cuts the whole per-image CPU cost from 29.3ms to 19.4ms (-34%).
global_train_photometric = torchvision.transforms.ColorJitter(
    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0
)

# Minimum side length (in original pixels) for a GT box to be kept.
MIN_BOX_SIZE_PX = 1.0


def collate_with_multiple_targets(batch, is_train=True):
    """
    Collate a batch of (PIL image, COCO annotation list) pairs.

    Training applies random horizontal flip + colour jitter; validation applies
    neither. Annotations are converted to normalized [cx, cy, w, h].

    Crowd regions (iscrowd=1) and degenerate boxes are dropped: they are not
    valid single-instance targets and would corrupt Hungarian matching / CIoU.
    """
    images = []
    targets = []

    # Select correct transform
    transform = global_train_transform if is_train else global_val_transform

    # Batch process images and targets
    for img, target_list in batch:
        orig_w, orig_h = img.size

        # torch's RNG is re-seeded per DataLoader worker, python's `random` is
        # not guaranteed to be, so draw the flip decision from torch.
        do_flip = is_train and bool(torch.rand(()) < 0.5)
        if do_flip:
            img = TF.hflip(img)
        if is_train:
            img = global_train_photometric(img)

        # Apply transform
        img = transform(img)
        images.append(img)

        # Fast process targets
        image_targets = []
        if target_list:
            for target in target_list:
                if target.get('iscrowd', 0):
                    continue

                bbox = target['bbox']  # [x_tl, y_tl, w, h] in original pixels
                if bbox[2] <= MIN_BOX_SIZE_PX or bbox[3] <= MIN_BOX_SIZE_PX:
                    continue

                # Normalize to [0, 1] against the ORIGINAL size; the non
                # aspect-preserving resize below is undone by the same ratio at
                # inference time, so this stays consistent.
                x1, y1 = bbox[0] / orig_w, bbox[1] / orig_h
                x2, y2 = x1 + bbox[2] / orig_w, y1 + bbox[3] / orig_h

                if do_flip:
                    x1, x2 = 1.0 - x2, 1.0 - x1

                # Clip to the image; a few COCO boxes stick out slightly.
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(1.0, x2), min(1.0, y2)
                w, h = x2 - x1, y2 - y1
                if w <= 0.0 or h <= 0.0:
                    continue

                image_targets.append({
                    'bbox': torch.tensor([x1 + w * 0.5, y1 + h * 0.5, w, h], dtype=torch.float32),
                    'category_id': torch.tensor(target['category_id'], dtype=torch.long)
                })

        targets.append(image_targets)

    return torch.stack(images), targets


# Create lambda functions for DataLoader
def collate_train(batch):
    return collate_with_multiple_targets(batch, is_train=True)


def collate_val(batch):
    return collate_with_multiple_targets(batch, is_train=False)


def load_coco_dataset(batch_size=48, input_resolution=None, num_workers=8):
    """
    Load COCO2017 datasets with specified batch size.

    Args:
        batch_size: Batch size for training and validation (default: 48)
        input_resolution: Square input resolution; None keeps the current setting
        num_workers: DataLoader worker processes. Decoding a JPEG and resizing it
            to 800x800 is CPU-bound, so this wants to be close to the physical
            core count or the GPU will sit waiting on data.

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    if input_resolution is not None:
        set_input_resolution(input_resolution)

    # Do not apply transform in CocoDetection, because we need original image size in collate_fn
    train_dataset = CocoDetection(
        root='./data/coco/train2017',
        annFile='./data/coco/annotations/instances_train2017.json',
        transform=None
    )

    val_dataset = CocoDetection(
        root='./data/coco/val2017',
        annFile='./data/coco/annotations/instances_val2017.json',
        transform=None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate_train,
                              persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_val,
                            persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)

    print(f"COCO2017 datasets loaded:")
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset: {len(val_dataset)} samples")
    print(f"  Input resolution: {INPUT_RESOLUTION}×{INPUT_RESOLUTION}")
    print(f"  Batch size: {batch_size}, workers: {num_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader
