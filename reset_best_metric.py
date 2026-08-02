"""
Reset the early-stopping state inside a checkpoint.

Needed when you resume training after changing loss weights. The stored
`best_val_loss` was computed under the OLD weighting, so if the new weighting
produces larger numbers nothing will ever beat it: the run stops saving "best"
checkpoints and early stopping fires a few epochs later for no real reason.

The model weights are untouched - only the bookkeeping fields change.

    python reset_best_metric.py runs/coco_detection_head_epoch10.pth
    python reset_best_metric.py runs/coco_detection_head_epoch10.pth --keep-recall
"""

import argparse
import os
import shutil

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--keep-recall', action='store_true',
                   help="keep best_val_acc (recall is weight-independent, so this "
                        "is usually what you want; default resets it too)")
    p.add_argument('--no-backup', action='store_true')
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    print(f"{args.checkpoint}")
    print(f"  epoch             {ckpt.get('epoch')}")
    print(f"  best_val_loss     {ckpt.get('best_val_loss')}")
    print(f"  best_val_acc      {ckpt.get('best_val_acc')}")
    print(f"  epochs_no_improve {ckpt.get('epochs_no_improve')}")

    if not args.no_backup:
        backup = args.checkpoint + '.bak'
        shutil.copy2(args.checkpoint, backup)
        print(f"\nbackup written to {backup}")

    ckpt['best_val_loss'] = float('inf')
    ckpt['epochs_no_improve'] = 0
    if not args.keep_recall:
        ckpt['best_val_acc'] = 0.0

    # The LR schedule tracked the old loss scale too, so its "best" is stale.
    if 'scheduler' in ckpt and isinstance(ckpt['scheduler'], dict):
        if 'best' in ckpt['scheduler']:
            ckpt['scheduler']['best'] = float('inf')
        ckpt['scheduler']['num_bad_epochs'] = 0
        print("scheduler plateau state reset as well")

    torch.save(ckpt, args.checkpoint)

    print(f"\nreset ->")
    print(f"  best_val_loss     {ckpt['best_val_loss']}")
    print(f"  best_val_acc      {ckpt['best_val_acc']}")
    print(f"  epochs_no_improve {ckpt['epochs_no_improve']}")
    print("\nModel weights unchanged. Resume as usual with `python train.py`.")


if __name__ == '__main__':
    main()
