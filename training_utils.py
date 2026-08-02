"""
Training Utilities

This module contains utility functions for training:
- Checkpoint management (finding and loading checkpoints)
- Other training-related helper functions
"""

import os

from tracking import read_manifest

BEST_CHECKPOINT_NAME = 'best_coco_detection_head.pth'
EPOCH_CHECKPOINT_PREFIX = 'coco_detection_head_epoch'


def _best_checkpoint_epoch(best_path):
    """
    1-based epoch of the best checkpoint, read from its sibling JSON manifest.

    Using the manifest avoids torch.load()ing a ~100 MB file just to read one
    integer. Returns None when there is no manifest (checkpoints written before
    tracking existed).
    """
    manifest = read_manifest(best_path)
    if not manifest:
        return None
    epoch = manifest.get('epoch')
    return epoch + 1 if isinstance(epoch, int) else None


def find_latest_checkpoint(directory='runs', best=True):
    """
    Find the checkpoint to resume from.

    Args:
        directory: Directory to search for checkpoints (default: 'runs')
        best: True returns the best model immediately. False prefers the
            periodic epoch checkpoints, but still falls back to the best model -
            periodic saves only happen every 5 epochs, so a run that stopped
            before the first one would otherwise appear to have no checkpoint at
            all and silently restart from scratch.

    Returns:
        checkpoint_path: Path to the checkpoint (or None if not found)
        epoch: 1-based epoch of that checkpoint (0 if unknown, -1 for best model
            when best=True). The caller re-derives the authoritative value from
            the checkpoint's own 'epoch' field after loading.
    """
    if not os.path.exists(directory):
        return None, 0

    best_path = os.path.join(directory, BEST_CHECKPOINT_NAME)
    has_best = os.path.exists(best_path)

    if best and has_best:
        return best_path, -1

    checkpoints = []
    for f in os.listdir(directory):
        if f.startswith(EPOCH_CHECKPOINT_PREFIX) and f.endswith('.pth'):
            try:
                epoch_num = int(f.replace(EPOCH_CHECKPOINT_PREFIX, '').replace('.pth', ''))
                checkpoints.append((epoch_num, f))
            except ValueError:
                pass

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    best_epoch = _best_checkpoint_epoch(best_path) if has_best else None

    if checkpoints:
        latest_epoch, latest_file = checkpoints[0]
        # The best model can be newer than the last periodic save; it carries the
        # same optimizer/scheduler/scaler state, so resuming from it is valid.
        if best_epoch is not None and best_epoch > latest_epoch:
            return best_path, best_epoch
        return os.path.join(directory, latest_file), latest_epoch

    if has_best:
        return best_path, best_epoch or 0

    return None, 0
