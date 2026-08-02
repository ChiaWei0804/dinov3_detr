"""
Measure DataLoader throughput in isolation, with no model attached.

Compare the number it prints against the images/sec your training run achieves
(tqdm's it/s x batch_size). If the loader is much faster than training, the GPU
is the bottleneck and tuning workers/augmentation will not help. If the two are
close, the loader is starving the GPU and is worth optimising.

Run it while training is STOPPED - otherwise the two compete for the same cores
and both numbers are meaningless.

    python bench_dataloader.py                  # current settings
    python bench_dataloader.py --workers 12     # try more workers
    python bench_dataloader.py --batches 40
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

import coco_dataset
from coco_dataset import collate_train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='./data/coco/train2017')
    p.add_argument('--ann', default='./data/coco/annotations/instances_train2017.json')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resolution', type=int, default=800)
    p.add_argument('--batches', type=int, default=30, help='timed batches (after warmup)')
    p.add_argument('--warmup', type=int, default=5)
    args = p.parse_args()

    coco_dataset.set_input_resolution(args.resolution)
    ds = CocoDetection(root=args.root, annFile=args.ann, transform=None)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=collate_train,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None)

    print(f"\nresolution={args.resolution}  batch_size={args.batch_size}  workers={args.workers}")
    print(f"warming up ({args.warmup} batches, includes worker startup)...")

    it = iter(loader)
    for _ in range(args.warmup):
        next(it)

    print(f"timing {args.batches} batches...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    n_img = 0
    for _ in range(args.batches):
        images, _ = next(it)
        n_img += images.shape[0]
    dt = time.perf_counter() - t0

    ips = n_img / dt
    bps = args.batches / dt
    steps_per_epoch = -(-len(ds) // args.batch_size)

    print(f"\n  {bps:6.2f} batches/sec")
    print(f"  {ips:6.1f} images/sec")
    print(f"  {dt / args.batches * 1000:6.1f} ms/batch")
    print(f"\n  data-only epoch time: {steps_per_epoch / bps / 60:.1f} min "
          f"({steps_per_epoch} batches over {len(ds)} images)")
    print("\nIf your training epoch takes much longer than that, you are GPU-bound.")
    print("If it is close to that, the loader is the bottleneck.\n")


if __name__ == '__main__':
    main()
