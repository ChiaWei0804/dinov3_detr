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


PATCH_SIZE = 16

# ==================== Aspect-ratio buckets ====================
#
# Every image used to be resized to a square, which squashes a 4:3 photo
# vertically by 33% and a 3:2 one by 50%. Nothing downstream ever required a
# square: dinov3_detection_model derives its patch grid from x.shape,
# PositionEmbeddingSine normalizes each axis separately, and the encoder
# objectness target takes (h, w) explicitly. This module was the only thing
# hardcoding it.
#
# Each bucket gets its own canvas and AspectRatioBatchSampler puts only
# same-bucket images in a batch, so there is no padding and therefore no need
# for an attention mask - which matters, because neither TransformerEncoder nor
# TransformerDecoder supports one.
#
# Targets are unaffected. They are normalized against the width and height of
# whatever frame the image was cropped to, and a full-frame resize onto any
# canvas preserves those fractions exactly. The box convention is unchanged.
BUCKET_LANDSCAPE, BUCKET_SQUARE, BUCKET_PORTRAIT = 0, 1, 2

BUCKET_NAMES = {
    BUCKET_LANDSCAPE: 'landscape',
    BUCKET_SQUARE: 'square',
    BUCKET_PORTRAIT: 'portrait',
}

# Patch-grid shape per bucket, as a fraction of the square grid
# (INPUT_RESOLUTION // PATCH_SIZE). At resolution 800 the square grid is 50x50,
# so 1.14/0.86 gives 57x43 = 2451 tokens against the square's 2500: VRAM and
# epoch time stay at the measured baseline no matter which bucket a batch is in.
# 57/43 = 1.326 is within 1% of 4:3.
_BUCKET_GRID_SCALES = {
    BUCKET_LANDSCAPE: (1.14, 0.86),
    BUCKET_SQUARE: (1.00, 1.00),
    BUCKET_PORTRAIT: (0.86, 1.14),
}

# w/h at or above this goes to landscape, at or below its reciprocal to
# portrait. COCO's two dominant shapes, 640x480 (1.333) and 640x427 (1.499),
# both land in landscape, where the residual distortion is 1.01x and 1.13x
# instead of the 1.33x and 1.50x the square canvas imposed.
BUCKET_LANDSCAPE_MIN_AR = 1.2
BUCKET_PORTRAIT_MAX_AR = 1.0 / BUCKET_LANDSCAPE_MIN_AR


def bucket_for_size(width, height):
    """Which aspect bucket an image of this original size belongs to."""
    if height <= 0:
        return BUCKET_SQUARE
    ar = width / height
    if ar >= BUCKET_LANDSCAPE_MIN_AR:
        return BUCKET_LANDSCAPE
    if ar <= BUCKET_PORTRAIT_MAX_AR:
        return BUCKET_PORTRAIT
    return BUCKET_SQUARE


def canvas_for_bucket(bucket, resolution=None):
    """Canvas size (width, height) in pixels for a bucket, aligned to patches."""
    if resolution is None:
        resolution = INPUT_RESOLUTION
    grid = max(1, int(resolution) // PATCH_SIZE)
    scale_w, scale_h = _BUCKET_GRID_SCALES[bucket]
    return (max(1, round(grid * scale_w)) * PATCH_SIZE,
            max(1, round(grid * scale_h)) * PATCH_SIZE)


def canvas_for_size(width, height, resolution=None):
    """Canvas an image of this original size is resized onto."""
    return canvas_for_bucket(bucket_for_size(width, height), resolution)


# Built lazily per canvas and cached: a Compose is stateless, and rebuilding one
# per batch would allocate in the DataLoader worker hot path.
_CANVAS_TRANSFORM_CACHE = {}


def transform_for_canvas(canvas):
    """Geometry pipeline for one canvas, shared by training and validation."""
    transform = _CANVAS_TRANSFORM_CACHE.get(canvas)
    if transform is None:
        width, height = canvas
        # Use BILINEAR interpolation to accelerate resize (2-3x faster than BICUBIC)
        transform = torchvision.transforms.Compose([
            # torchvision's Resize takes (height, width), not (width, height).
            torchvision.transforms.Resize(
                (height, width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        _CANVAS_TRANSFORM_CACHE[canvas] = transform
    return transform


def transform_for_size(width, height, resolution=None):
    """Geometry pipeline for an image of this original size."""
    return transform_for_canvas(canvas_for_size(width, height, resolution))


def set_input_resolution(resolution):
    """
    Change the input resolution used by every transform in this module.

    Must be called BEFORE the DataLoaders are constructed. Previously the
    resolution was hardcoded here, so the `input_resolution` knob in train.py
    had no effect whatsoever.

    The value is the SQUARE-bucket side length; the other two canvases are
    derived from it by canvas_for_bucket().
    """
    global INPUT_RESOLUTION
    INPUT_RESOLUTION = int(resolution)
    os.environ['DINOV3_INPUT_RESOLUTION'] = str(INPUT_RESOLUTION)
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

# Scale jitter + random crop, training only. Until now the only geometric
# augmentation was a horizontal flip, so the model saw every object at exactly
# one scale for its whole training run - which is a plausible share of why
# AP(small) sits at 0.188 while AP(large) is 0.616.
#
# The crop keeps the CANVAS aspect ratio, so resizing it onto the canvas adds no
# distortion at all, and a scale below 1 magnifies the content. That
# magnification is the point: it is the only mechanism in this pipeline that
# ever makes a small object big.
CROP_PROBABILITY = 0.5
CROP_MIN_SCALE = 0.5
CROP_MAX_SCALE = 1.0


def _extract_boxes(target_list):
    """
    COCO annotations -> [(x1, y1, x2, y2, category_id), ...] in original pixels.

    Crowd regions (iscrowd=1) and degenerate boxes are dropped here: they are
    not valid single-instance targets and would corrupt Hungarian matching /
    CIoU.
    """
    boxes = []
    for target in target_list or ():
        if target.get('iscrowd', 0):
            continue
        bbox = target['bbox']  # [x_tl, y_tl, w, h] in original pixels
        if bbox[2] <= MIN_BOX_SIZE_PX or bbox[3] <= MIN_BOX_SIZE_PX:
            continue
        boxes.append((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3],
                      target['category_id']))
    return boxes


def _sample_crop(orig_w, orig_h, canvas_ar):
    """
    A random sub-rectangle with the canvas's aspect ratio.

    Starts from the largest rectangle of that aspect that fits inside the image,
    scales it by a random factor and places it at random.

    Returns (x0, y0, width, height) in original pixels.
    """
    if orig_w / orig_h > canvas_ar:
        max_h = float(orig_h)
        max_w = canvas_ar * max_h
    else:
        max_w = float(orig_w)
        max_h = max_w / canvas_ar

    # torch's RNG is re-seeded per DataLoader worker, python's `random` is not
    # guaranteed to be, so every draw here comes from torch.
    scale = CROP_MIN_SCALE + (CROP_MAX_SCALE - CROP_MIN_SCALE) * float(torch.rand(()))
    crop_w = max(1.0, min(float(orig_w), max_w * scale))
    crop_h = max(1.0, min(float(orig_h), max_h * scale))
    x0 = float(torch.rand(())) * (orig_w - crop_w)
    y0 = float(torch.rand(())) * (orig_h - crop_h)
    return x0, y0, crop_w, crop_h


def _boxes_in_crop(boxes, crop):
    """
    Keep the boxes whose CENTRE falls inside the crop, translated into crop
    coordinates.

    Centre-in-crop is the SSD/YOLO convention. Plain clipping would instead keep
    the sliver of an object that the crop grazed and still label it with the
    whole object's class, which is supervision the model cannot satisfy.
    Boxes are clipped to the frame later, in normalized space.
    """
    x0, y0, crop_w, crop_h = crop
    kept = []
    for x1, y1, x2, y2, category_id in boxes:
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if x0 <= cx <= x0 + crop_w and y0 <= cy <= y0 + crop_h:
            kept.append((x1 - x0, y1 - y0, x2 - x0, y2 - y0, category_id))
    return kept


def collate_with_multiple_targets(batch, is_train=True):
    """
    Collate a batch of (PIL image, COCO annotation list) pairs.

    Training applies scale jitter + random crop, random horizontal flip and
    colour jitter; validation applies none of them. Annotations are converted to
    normalized [cx, cy, w, h].

    Every image in the batch is resized onto ONE canvas, chosen from the first
    sample's aspect bucket. AspectRatioBatchSampler is what makes that correct -
    it only ever groups same-bucket indices. A batch assembled without it still
    works, the minority images just distort the way they did before bucketing.
    """
    images = []
    targets = []

    first_w, first_h = batch[0][0].size
    canvas = canvas_for_size(first_w, first_h)
    transform = transform_for_canvas(canvas)
    canvas_ar = canvas[0] / canvas[1]

    # Batch process images and targets
    for img, target_list in batch:
        orig_w, orig_h = img.size
        boxes = _extract_boxes(target_list)

        # torch's RNG is re-seeded per DataLoader worker, python's `random` is
        # not guaranteed to be, so draw every decision from torch.
        crop = None
        if is_train and bool(torch.rand(()) < CROP_PROBABILITY):
            crop = _sample_crop(orig_w, orig_h, canvas_ar)
            kept = _boxes_in_crop(boxes, crop)
            if boxes and not kept:
                # The crop removed every object, turning a labelled image into a
                # background-only sample. The loss tolerates empty targets, but
                # training on them is not what the augmentation is for.
                crop = None
            else:
                boxes = kept

        if crop is None:
            frame_w, frame_h = float(orig_w), float(orig_h)
        else:
            x0, y0, frame_w, frame_h = crop
            img = img.crop((x0, y0, x0 + frame_w, y0 + frame_h))

        do_flip = is_train and bool(torch.rand(()) < 0.5)
        if do_flip:
            img = TF.hflip(img)
        if is_train:
            img = global_train_photometric(img)

        # Apply transform
        images.append(transform(img))

        # Fast process targets
        image_targets = []
        for x1, y1, x2, y2, category_id in boxes:
            # Normalize to [0, 1] against the frame the image was cropped to -
            # the whole image when there was no crop. Any full-frame resize
            # scales both axes uniformly, so these fractions survive it, and
            # eval_coco.py's multiply-by-orig_size stays correct.
            x1, y1 = x1 / frame_w, y1 / frame_h
            x2, y2 = x2 / frame_w, y2 / frame_h

            if do_flip:
                x1, x2 = 1.0 - x2, 1.0 - x1

            # Clip to the frame; a few COCO boxes stick out slightly, and a
            # cropped one can stick out a lot.
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)
            w, h = x2 - x1, y2 - y1
            if w <= 0.0 or h <= 0.0:
                continue

            image_targets.append({
                'bbox': torch.tensor([x1 + w * 0.5, y1 + h * 0.5, w, h], dtype=torch.float32),
                'category_id': torch.tensor(category_id, dtype=torch.long)
            })

        targets.append(image_targets)

    return torch.stack(images), targets


# Create lambda functions for DataLoader
def collate_train(batch):
    return collate_with_multiple_targets(batch, is_train=True)


def collate_val(batch):
    return collate_with_multiple_targets(batch, is_train=False)


class AspectRatioBatchSampler(torch.utils.data.Sampler):
    """
    Yield batches whose images all share one aspect bucket.

    That homogeneity is what lets collate_with_multiple_targets pick a single
    canvas per batch without padding: a batch never mixes a 912x688 image with a
    688x912 one, so torch.stack always agrees and no attention mask is needed.

    Batch COUNT is fixed across epochs (each bucket is chunked independently and
    the tail batch is kept), so len() is exact and the progress bar does not
    drift between epochs.
    """

    def __init__(self, bucket_ids, batch_size, shuffle):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._by_bucket = {}
        for index, bucket in enumerate(bucket_ids):
            self._by_bucket.setdefault(bucket, []).append(index)
        self._num_batches = sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._by_bucket.values())

    def bucket_counts(self):
        return {bucket: len(indices) for bucket, indices in self._by_bucket.items()}

    def __iter__(self):
        batches = []
        for indices in self._by_bucket.values():
            if self.shuffle:
                order = torch.randperm(len(indices)).tolist()
                indices = [indices[i] for i in order]
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start:start + self.batch_size])
        if self.shuffle:
            # Without this the run would see every landscape batch, then every
            # square one - a bucket-ordered curriculum nobody asked for.
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        return iter(batches)

    def __len__(self):
        return self._num_batches


def bucket_ids_for_dataset(dataset):
    """
    Aspect bucket per dataset index, read from the COCO index rather than by
    decoding images - the sampler needs all 118k of them before epoch 1 starts.
    """
    coco = dataset.coco
    bucket_ids = []
    for image_id in dataset.ids:
        info = coco.loadImgs(image_id)[0]
        bucket_ids.append(bucket_for_size(info['width'], info['height']))
    return bucket_ids


def load_coco_dataset(batch_size=48, input_resolution=None, num_workers=8):
    """
    Load COCO2017 datasets with specified batch size.

    Args:
        batch_size: Batch size for training and validation (default: 48)
        input_resolution: Side of the SQUARE bucket's canvas; the landscape and
            portrait canvases are derived from it. None keeps the current
            setting.
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
    
    train_sampler = AspectRatioBatchSampler(bucket_ids_for_dataset(train_dataset),
                                            batch_size=batch_size, shuffle=True)
    val_sampler = AspectRatioBatchSampler(bucket_ids_for_dataset(val_dataset),
                                          batch_size=batch_size, shuffle=False)

    # batch_sampler is mutually exclusive with batch_size / shuffle / drop_last.
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate_train,
                              persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate_val,
                            persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)

    print(f"COCO2017 datasets loaded:")
    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset: {len(val_dataset)} samples")
    print(f"  Aspect buckets (canvas / train imgs / val imgs):")
    train_counts = train_sampler.bucket_counts()
    val_counts = val_sampler.bucket_counts()
    for bucket in (BUCKET_LANDSCAPE, BUCKET_SQUARE, BUCKET_PORTRAIT):
        canvas_w, canvas_h = canvas_for_bucket(bucket)
        print(f"    {BUCKET_NAMES[bucket]:<9} {canvas_w}×{canvas_h} "
              f"({canvas_w // PATCH_SIZE}×{canvas_h // PATCH_SIZE} patches, "
              f"{(canvas_w // PATCH_SIZE) * (canvas_h // PATCH_SIZE)} tokens)  "
              f"{train_counts.get(bucket, 0)} / {val_counts.get(bucket, 0)}")
    print(f"  Batch size: {batch_size}, workers: {num_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader
