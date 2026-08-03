"""
Measure pure GPU throughput with synthetic data - no DataLoader involved.

bench_dataloader.py already showed the loader can supply ~2x what training
consumes, and the loss costs ~4% of a step, so whatever is left is GPU compute.
This isolates it:

  backbone only    the frozen ViT-B/16 forward, ~89% of the FLOPs
  full forward     + encoder / query selection / decoder / heads
  full train step  + backward + optimizer

Read it like this:
  - "backbone only" close to "full train step" -> the frozen ViT is the floor.
    Nothing in the detector is worth optimising; only a lower input resolution
    or a smaller backbone would move the number.
  - "full train step" much slower than "backbone only" -> something in the
    detector or the training loop is stalling and is worth chasing.

Run with training STOPPED.

    python bench_gpu.py
    python bench_gpu.py --batch-size 48 --resolution 640
"""

import argparse
import time

import torch
from torch.amp import autocast, GradScaler

from dinov3_detection_model import DINOv3DetectionModel
from detection_loss import DetectionLoss
from coco_dataset import COCO_NUM_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--resolution', type=int, default=800)
    p.add_argument('--queries', type=int, default=100)
    p.add_argument('--encoder-layers', type=int, default=2)
    p.add_argument('--decoder-layers', type=int, default=6)
    p.add_argument('--iters', type=int, default=15)
    p.add_argument('--warmup', type=int, default=5)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available - this benchmark needs a GPU.")
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dev = torch.device('cuda')
    print(f"\n{torch.cuda.get_device_name(0)}  |  torch {torch.__version__}  |  CUDA {torch.version.cuda}")
    print(f"batch={args.batch_size}  resolution={args.resolution}  "
          f"enc={args.encoder_layers}  dec={args.decoder_layers}  queries={args.queries}\n")

    model = DINOv3DetectionModel(num_classes=COCO_NUM_CLASSES, num_queries=args.queries,
                                 num_encoder_layers=args.encoder_layers,
                                 num_decoder_layers=args.decoder_layers,
                                 num_aux_layers=0).to(dev)
    model.train()

    loss_fn = DetectionLoss(num_classes=COCO_NUM_CLASSES, num_queries=args.queries,
                            aux_loss_weight=0, enc_loss_weight=1.0).to(dev)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=1e-4)
    scaler = GradScaler('cuda')

    x = torch.randn(args.batch_size, 3, args.resolution, args.resolution, device=dev)
    # COCO averages ~7.3 objects per image
    targets = [[{'bbox': torch.rand(4).clamp(0.15, 0.85),
                 'category_id': torch.tensor(i % 80 + 1)} for i in range(7)]
               for _ in range(args.batch_size)]

    def timed(fn, n):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    def backbone_only():
        with torch.no_grad(), autocast('cuda'):
            model.backbone.forward_features(x)

    def full_forward():
        with torch.no_grad(), autocast('cuda'):
            model(x)

    def _step(loss_builder):
        opt.zero_grad(set_to_none=True)
        with autocast('cuda'):
            out = model(x)
            loss = loss_builder(out)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(opt)
        scaler.update()

    def train_step():
        _step(lambda out: loss_fn(out, targets)[0])

    def train_step_dummy_loss():
        # Same graph, same backward, but a trivial scalar instead of the real
        # criterion. The gap between this and train_step is what Hungarian
        # matching + the loss computation actually cost.
        _step(lambda out: out['pred_boxes'].sum() + out['pred_logits'].sum()
              + out['enc_objectness'].sum())

    rows = []
    for name, fn in [("backbone only (frozen ViT-B/16)", backbone_only),
                     ("full forward", full_forward),
                     ("train step, DUMMY loss", train_step_dummy_loss),
                     ("full train step (fwd+bwd+opt)", train_step)]:
        dt = timed(fn, args.iters)
        rows.append((name, dt))

    print(f"{'stage':<36}{'ms/step':>10}{'img/s':>10}{'min/epoch':>12}")
    print("-" * 68)
    for name, dt in rows:
        ips = args.batch_size / dt
        print(f"{name:<36}{dt*1000:>10.1f}{ips:>10.1f}{118287/ips/60:>12.1f}")
    print("-" * 68)

    bb, fwd, dummy, step = (r[1] for r in rows)
    print(f"\n  backbone                {bb*1000:7.1f} ms  ({bb/step*100:4.1f}% of a training step)")
    print(f"  detector forward        {(fwd-bb)*1000:7.1f} ms")
    print(f"  backward + optimizer    {(dummy-fwd)*1000:7.1f} ms")
    print(f"  loss (matching + calc)  {(step-dummy)*1000:7.1f} ms  <-- the only part worth tuning")
    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"50 epochs at this rate: {118287/(args.batch_size/step)*50/3600:.1f} hours "
          f"(excluding validation)\n")


if __name__ == '__main__':
    main()
