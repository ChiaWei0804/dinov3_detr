"""
train2.py - DINOv3 Detection Training (Plain DETR Architecture)

Plain DETR style detection training pipeline based on DINOv3:

Architecture (Full Plain DETR):
1. Frozen DINOv3 Backbone (ViT-B/16)
2. Lightweight projection layers (Patch Projection + CLS Projection)
3. Transformer Encoder (2 layers - independent module, not fused into backbone)
4. Mixed Query Selection (dynamically select Top-K queries from Encoder output)
5. Transformer Decoder (4 layers with Self-Attention + Cross-Attention + FFN)
6. Detection heads (Classification Head + BBox Head)

Loss Function:
- L1 Loss: coordinate-level precision
- CIoU Loss: Complete IoU (considers overlap, center distance, aspect ratio)
- Classification Loss: CrossEntropy for object classes (background down-weighted
  by eos_coef, background is class index 0)
- Encoder objectness Loss: supervises the Top-K score head of Mixed Query Selection

Configuration:
- Input resolution: 800×800
- Feature map size: 50×50 (2500 patches from ViT-B/16)
- Batch size: 8 (with AMP for memory efficiency)
- Training: COCO 2017, see target_total_epochs in main()

Comparison with official DINOv3 implementation:
- Same: frozen backbone, independent Transformer Encoder, end-to-end training
- Different:
  * Official: Objects365 pretraining + 2048px resolution + 100M params detector
  * Ours: direct COCO training + 800px resolution + ~30M params detector

All checkpoints, best model and final model are saved in runs/ directory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
import json
import math
import os
import sys
import traceback
import numpy as np
from tqdm import tqdm

# Import from custom modules
from dinov3_detection_model import DINOv3DetectionModel, PositionEmbeddingSine
from detection_loss import DetectionLoss, complete_box_iou, hungarian_matching_simple
from coco_dataset import load_coco_dataset, COCO_NUM_CLASSES
from training_utils import find_latest_checkpoint
from validation_utils import validate_epoch
from tracking import ExperimentTracker, setup_file_logging, LOSS_SEMANTICS_VERSION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Windows spawns DataLoader workers by re-importing this module (as
# __mp_main__), so an unguarded print here fires once per worker.
if __name__ == '__main__':
    print(f"device: {device}")


def build_checkpoint(model, optimizer, scheduler, scaler, epoch,
                     best_val_loss, best_val_acc, epochs_no_improve, loss_fn):
    """
    Assemble a resumable checkpoint.

    Besides the weights this stores the scheduler and early-stopping state, so a
    resumed run keeps its LR schedule and does not overwrite the best model with
    a worse one on its first epoch.
    """
    return {
        'patch_proj': model.patch_proj.state_dict(),
        'cls_proj': model.cls_proj.state_dict(),
        'encoder': model.encoder.state_dict(),
        'topk_score_head': model.topk_score_head.state_dict(),
        'decoder': model.decoder.state_dict(),
        'cls_head': model.cls_head.state_dict(),
        'bbox_head': model.bbox_head.state_dict(),
        'num_queries': model.num_queries,
        'num_encoder_layers': model.num_encoder_layers,
        'num_decoder_layers': model.num_decoder_layers,
        'num_aux_layers': model.num_aux_layers,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'epochs_no_improve': epochs_no_improve,
        # Stamped so a resume can tell whether best_val_loss was computed under
        # the same loss definition this run uses. Checkpoints written before this
        # field existed simply have no key, which is handled as "older".
        'loss_semantics_version': LOSS_SEMANTICS_VERSION,
        # The version constant only moves when someone edits it. This snapshot
        # catches the configuration changes it silently misses.
        'loss_weights': loss_fn.loss_scale_signature(),
    }


def train_detection_model(model, train_loader, val_loader, num_epochs=10, start_epoch=0,
                         resume_checkpoint=None, early_stopping_patience=8, save_dir='runs',
                         # Learning rate parameters
                         warmup_epochs=10, encoder_warmup_lr=0.00005, encoder_target_lr=0.0001,
                         backbone_lr=0.0001, decoder_lr=0.0001,
                         # Loss weights
                         bbox_loss_weight=10.0, ciou_loss_weight=5.0, cls_loss_weight=2.0, aux_loss_weight=0.4,
                         eos_coef=0.1, enc_loss_weight=1.0,
                         # Other training parameters
                         gradient_clip_max_norm=5.0, tracker=None):
    """
    Train detection model with configurable hyperparameters.

    Args:
        tracker: ExperimentTracker. When omitted a manifest-only tracker is
            created, so every checkpoint still gets its provenance JSON.
    """
    if tracker is None:
        tracker = ExperimentTracker(save_dir=save_dir, config={}, use_mlflow=False)
    model.to(device)
    num_queries = model.num_queries

    # Store hyperparameters for use in training loop
    WARMUP_EPOCHS = warmup_epochs
    ENCODER_WARMUP_LR = encoder_warmup_lr
    ENCODER_TARGET_LR = encoder_target_lr

    loss_fn = DetectionLoss(
        # Must track the model: empty_weight is sized num_classes+1, and a
        # mismatch with the model's logits blows up inside cross_entropy.
        # Read it off the model rather than hardcoding 91, which silently
        # decoupled the two whenever the caller passed anything else.
        num_classes=model.num_classes,
        num_queries=num_queries,
        bbox_loss_weight=bbox_loss_weight,
        ciou_loss_weight=ciou_loss_weight,
        cls_loss_weight=cls_loss_weight,
        aux_loss_weight=aux_loss_weight,
        eos_coef=eos_coef,
        enc_loss_weight=enc_loss_weight,
        max_targets_per_image=100
    )
    # DetectionLoss holds the per-class weight buffer, so it must follow the model.
    loss_fn.to(device)

    # Three parameter groups with different learning rates
    # Group 0: Encoder (with Warmup) - randomly initialized, needs careful tuning
    # Group 1: Backbone related (patch_proj, cls_proj, etc.)
    # Group 2: Decoder + Detection Heads
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if ("encoder" in n) and p.requires_grad],
            "lr": encoder_warmup_lr,  # Initial warmup LR for Encoder
            "name": "encoder"
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if ("encoder" not in n and "bbox_head" not in n and "cls_head" not in n 
                          and "decoder" not in n and "topk_score_head" not in n) and p.requires_grad],
            "lr": backbone_lr,
            "name": "backbone"
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if ("bbox_head" in n or "cls_head" in n or "decoder" in n 
                          or "topk_score_head" in n) and p.requires_grad],
            "lr": decoder_lr,
            "name": "decoder_heads"
        },
    ]
    
    # Use fused=True to accelerate AdamW (requires PyTorch >= 2.0)
    try:
        optimizer = optim.AdamW(param_dicts, weight_decay=0.01, fused=True)
        print("Using fused AdamW optimizer with 3 parameter groups")
    except:
        optimizer = optim.AdamW(param_dicts, weight_decay=0.01)
        print("Using standard AdamW optimizer with 3 parameter groups")

    # Use AMP mixed precision training (40-50% speedup)
    # Must initialize scaler BEFORE loading checkpoint state
    scaler = GradScaler('cuda')
    print("Using AMP (Automatic Mixed Precision) Training")

    # If continuing training, try to load optimizer, scheduler, scaler states
    if resume_checkpoint is not None:
        # Restore AdamW's moment estimates. Fail gracefully rather than abort:
        # a changed param-group layout makes the state unloadable, and starting
        # the moments fresh costs a few steps, not the run.
        if 'optimizer' in resume_checkpoint:
            try:
                optimizer.load_state_dict(resume_checkpoint['optimizer'])
                print("Optimizer state loaded successfully.")
            except Exception as e:
                print(f"WARNING: Skipping optimizer state (Architecture mismatch): {e}")
                print("   Optimizer will start fresh for compatibility")

        # Deliberately NOT restoring the LR from the optimizer state here. Under
        # ReduceLROnPlateau the checkpoint's LR was the only record of how far
        # the decay had progressed, so it had to be preserved. A cosine schedule
        # is a closed-form function of the epoch index instead: it is rebuilt
        # below with last_epoch=start_epoch-1 and recomputes the correct LR from
        # scratch. Anything written here would just be overwritten.

        # Load AMP scaler state
        if 'scaler' in resume_checkpoint:
            print("Loading AMP scaler state from checkpoint...")
            try:
                scaler.load_state_dict(resume_checkpoint['scaler'])
                print("AMP scaler state loaded.")
            except Exception as e:
                print(f"Failed to load scaler state: {e}")
    
    # Cosine decay over the WHOLE run, replacing ReduceLROnPlateau(patience=3,
    # factor=0.75). Measured over the 50-epoch run that produced AP 0.398: the
    # plateau scheduler fired exactly once in 50 epochs, at epoch 46, because
    # val loss is noisy enough that a new low kept landing before the counter
    # reached 3. Epochs 21-45 sat at the full 1e-4 and moved recall 76.12% ->
    # 76.54% (+0.42pt in 25 epochs); the single 25% cut then moved it to 77.83%
    # in 5 (+1.29pt). The run was LR-limited for its whole middle, and nothing
    # about a plateau rule was going to notice.
    #
    # The lambdas are the ONLY thing that sets a learning rate now. The warmup
    # ramp used to be applied by hand at the top of the epoch loop while the
    # scheduler wrote the same field from its own state - two sources of truth
    # for one number, and the loop's write silently won for WARMUP_EPOCHS.
    total_epochs = start_epoch + num_epochs
    LR_ETA_MIN_RATIO = 0.01  # floor at 1% of base, i.e. 1e-6 from 1e-4

    def _cosine_factor(epoch):
        span = max(1, total_epochs - WARMUP_EPOCHS)
        progress = min(1.0, max(0.0, (epoch - WARMUP_EPOCHS) / span))
        return LR_ETA_MIN_RATIO + (1.0 - LR_ETA_MIN_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _encoder_factor(epoch):
        # Group 0's initial_lr is ENCODER_TARGET_LR, so the warmup ramp has to
        # be expressed as a fraction of it.
        if epoch < WARMUP_EPOCHS:
            warmup_progress = (epoch + 1) / WARMUP_EPOCHS
            lr = ENCODER_WARMUP_LR + (ENCODER_TARGET_LR - ENCODER_WARMUP_LR) * warmup_progress
            return lr / ENCODER_TARGET_LR
        return _cosine_factor(epoch)

    # LambdaLR multiplies initial_lr, which it reads off the param groups. After
    # a resume those carry whatever LR the checkpoint's optimizer state had
    # (7.5e-5 for the 50-epoch run), so the configured bases are written back
    # first - otherwise the cosine would restart from the decayed value and
    # decay it a second time.
    for group, base_lr in zip(optimizer.param_groups,
                              (encoder_target_lr, backbone_lr, decoder_lr)):
        group['initial_lr'] = base_lr

    # last_epoch=start_epoch-1 makes __init__'s implicit first step land on
    # start_epoch, which is the index of the first epoch this run will train.
    scheduler = LambdaLR(optimizer,
                         lr_lambda=[_encoder_factor, _cosine_factor, _cosine_factor],
                         last_epoch=start_epoch - 1)
    print(f"Using cosine LR decay over {total_epochs} epochs "
          f"(warmup {WARMUP_EPOCHS}, floor {LR_ETA_MIN_RATIO:.0%} of base)")

    # Early Stopping parameters
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0

    if resume_checkpoint is not None:
        # The scheduler state in the checkpoint is deliberately IGNORED. Older
        # checkpoints hold a ReduceLROnPlateau state whose keys (best,
        # num_bad_epochs, cooldown_counter) mean nothing here, and LambdaLR's
        # load_state_dict does self.__dict__.update(state) BEFORE it looks for
        # 'lr_lambdas' - so a plateau state overwrites last_epoch with the
        # plateau's own counter and only then raises, leaving the scheduler
        # pointing at the wrong place on the cosine curve. Rebuilding from
        # last_epoch=start_epoch-1 above is exact and needs no stored state.
        #
        # Restore the early-stopping state though: without it the resumed run
        # starts from best_val_loss=inf and overwrites the saved best model with
        # whatever the first epoch produces.
        best_val_loss = resume_checkpoint.get('best_val_loss', best_val_loss)
        best_val_acc = resume_checkpoint.get('best_val_acc', best_val_acc)
        epochs_no_improve = resume_checkpoint.get('epochs_no_improve', epochs_no_improve)
        print(f"   Restored best: val_loss={best_val_loss:.4f}, recall={best_val_acc*100:.2f}%, "
              f"no-improve streak={epochs_no_improve}")

        # A best_val_loss computed under a DIFFERENT loss definition is not a
        # threshold this run can clear. Measured for the v3 -> v4 change: the same
        # predictions score 1.52x higher, so a v3 best of 2.6158 lands near 3.99
        # here - nothing ever beats it, no "best" checkpoint is ever written
        # again, and epochs_no_improve climbs until early stopping fires for no
        # real reason.
        #
        # RESET rather than warn. An earlier version only printed a warning, which
        # was worse than useless: build_checkpoint() stamps the very next save
        # with the CURRENT version while the stale best_val_loss rides along
        # inside it, so from the second resume onwards the versions match, the
        # warning never fires again, and the bad threshold is now invisible.
        # There is also no correct alternative to resetting - a threshold from a
        # different loss definition is not a threshold at all.
        #
        # best_val_acc goes too: the recall metric changed with the same batch of
        # work (one-to-one Hungarian assignment instead of per-GT best match), so
        # the stored value is inflated relative to what this run can produce.
        # Two independent ways the stored best can become incomparable, and
        # neither subsumes the other:
        #   - the version constant, for changes to what the loss MEANS (a new enc
        #     target, matching on CIoU instead of IoU). Hand-maintained.
        #   - the weight signature, for changes to its MAGNITUDE. Automatic.
        # Measured for the second kind: aux_loss_weight 0.4 -> 0 takes the same
        # predictions from 24.63 to 17.75 (0.72x) with the version untouched, and
        # the smaller number reads as an improvement.
        ckpt_semantics = resume_checkpoint.get('loss_semantics_version')
        ckpt_weights = resume_checkpoint.get('loss_weights')
        current_weights = loss_fn.loss_scale_signature()
        version_changed = ckpt_semantics != LOSS_SEMANTICS_VERSION
        weights_changed = ckpt_weights != current_weights

        if version_changed or weights_changed:
            stale = (best_val_loss, best_val_acc, epochs_no_improve)
            best_val_loss = float('inf')
            best_val_acc = 0.0
            epochs_no_improve = 0
            # Nothing to reset on the scheduler side any more: the cosine
            # schedule is a function of the epoch index alone and never looked
            # at the loss. Under ReduceLROnPlateau its `best` tracked the old
            # loss scale and had to be cleared here too.

            print("\n" + "!" * 70)
            print("Loss definition changed since this checkpoint was written.")
            if version_changed:
                print(f"  checkpoint loss_semantics_version : {ckpt_semantics if ckpt_semantics is not None else 'absent (pre-versioning)'}")
                print(f"  this run                          : {LOSS_SEMANTICS_VERSION}")
            if weights_changed:
                if ckpt_weights is None:
                    print("  checkpoint records no loss weights (written before they were stored)")
                else:
                    print("  loss weights differ:")
                    for key in sorted(set(ckpt_weights) | set(current_weights)):
                        was, now = ckpt_weights.get(key), current_weights.get(key)
                        if was != now:
                            print(f"    {key:18s} {was} -> {now}")
            print("")
            print("  Early-stopping bookkeeping has been RESET (model weights and")
            print("  optimizer state are untouched):")
            print(f"    best_val_loss     {stale[0]:.4f} -> inf")
            print(f"    best_val_acc      {stale[1] * 100:.2f}% -> 0.00%")
            print(f"    epochs_no_improve {stale[2]} -> 0")
            print("")
            print("  The next improving epoch will therefore write a new best")
            print("  model, replacing the one saved under the old definition.")
            print("!" * 70 + "\n")

    # Check current LR
    current_lrs = [group['lr'] for group in optimizer.param_groups]
    print(f"Initial LRs: {current_lrs}")

    # Note: Warmup configuration is defined at function start (WARMUP_EPOCHS, ENCODER_WARMUP_LR, ENCODER_TARGET_LR)

    train_losses = []
    val_losses = []
    val_accuracies = []
    # Seeded from the scheduler's opening rates so the return is well-defined
    # even if the loop body never runs; every epoch overwrites it.
    final_lrs = {'encoder': optimizer.param_groups[0]['lr'],
                 'backbone': optimizer.param_groups[1]['lr'],
                 'decoder': optimizer.param_groups[2]['lr']}

    print(f"Training from epoch {start_epoch+1} to {start_epoch+num_epochs}...")
    print(f"Saving checkpoints to: {save_dir}/")
    print(f"Encoder Warmup: {WARMUP_EPOCHS} epochs ({ENCODER_WARMUP_LR} -> {ENCODER_TARGET_LR})")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # No LR assignment here. The scheduler set this epoch's rates when it
        # stepped at the end of the previous one (or at construction, for the
        # first epoch of the run), warmup included.
        if epoch < WARMUP_EPOCHS:
            print(f"Epoch {epoch+1}: Encoder LR Warmup = "
                  f"{optimizer.param_groups[0]['lr']:.6f}")

        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_ciou_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_enc_loss = 0.0

        # disable=None turns the bar off whenever stderr is not a terminal.
        # tqdm redraws in place with \r, which is one line on a console but a
        # full extra line every 0.1s once the stream is a file - hundreds of
        # thousands of lines per epoch on COCO.
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch+num_epochs}',
                         ncols=120, disable=None)

        # A compact progress line roughly 10 times per epoch, always. The tqdm
        # bar only exists on an interactive console and never reaches the log
        # file, so without these the log would show nothing at all between epoch
        # summaries - on COCO that is a 20+ minute silence.
        num_batches = len(train_loader)
        progress_interval = max(1, num_batches // 10)

        for batch_idx, (images, targets) in enumerate(train_bar, start=1):
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # AMP mixed precision forward pass
            with autocast('cuda'):
                outputs = model(images)
                loss, l1_loss, ciou_loss, cls_loss, enc_loss = loss_fn(outputs, targets)

            # AMP mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_l1_loss += l1_loss
            epoch_ciou_loss += ciou_loss
            epoch_cls_loss += cls_loss
            epoch_enc_loss += enc_loss
            train_bar.set_postfix({
                'Loss': f'{loss.item():.2f}',
                'L1': f'{l1_loss:.2f}',
                'CIoU': f'{ciou_loss:.2f}',
                'Cls': f'{cls_loss:.2f}',
                'Enc': f'{enc_loss:.2f}'
            })

            if batch_idx % progress_interval == 0 or batch_idx == num_batches:
                print(f'  [epoch {epoch+1}] {batch_idx}/{num_batches} '
                      f'({100.0 * batch_idx / num_batches:.0f}%) '
                      f'loss={epoch_loss / batch_idx:.4f} '
                      f'l1={epoch_l1_loss / batch_idx:.4f} '
                      f'ciou={epoch_ciou_loss / batch_idx:.4f} '
                      f'cls={epoch_cls_loss / batch_idx:.4f} '
                      f'enc={epoch_enc_loss / batch_idx:.4f}', flush=True)

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_l1_loss = epoch_l1_loss / len(train_loader)
        avg_train_ciou_loss = epoch_ciou_loss / len(train_loader)
        avg_train_cls_loss = epoch_cls_loss / len(train_loader)
        avg_train_enc_loss = epoch_enc_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        val_metrics = validate_epoch(model, val_loader, loss_fn, device, 
                                     confidence_threshold=0.1, iou_threshold=0.5)
        
        avg_val_loss = val_metrics['avg_val_loss']
        avg_val_l1_loss = val_metrics['avg_val_l1_loss']
        avg_val_ciou_loss = val_metrics['avg_val_ciou_loss']
        avg_val_cls_loss = val_metrics['avg_val_cls_loss']
        avg_val_enc_loss = val_metrics['avg_val_enc_loss']
        recall_iou05 = val_metrics['recall_iou05']
        avg_max_iou = val_metrics['avg_max_iou']
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(recall_iou05)

        # The scheduler steps at the very END of the loop body, not here. It no
        # longer consumes val_loss, and stepping before the reporting below
        # would log and checkpoint the NEXT epoch's learning rates.

        print(f'Epoch {epoch+1}/{start_epoch+num_epochs}:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, L1: {avg_train_l1_loss:.4f}, CIoU: {avg_train_ciou_loss:.4f}, Cls: {avg_train_cls_loss:.4f}, Enc: {avg_train_enc_loss:.4f}')
        print(f'  Val   - Loss: {avg_val_loss:.4f}, L1: {avg_val_l1_loss:.4f}, CIoU: {avg_val_ciou_loss:.4f}, Cls: {avg_val_cls_loss:.4f}, Enc: {avg_val_enc_loss:.4f}')
        print(f'  Val   - Recall@IoU0.5: {recall_iou05*100:.2f}%, Avg Max IoU: {avg_max_iou:.4f}')
        
        # Display LR for all parameter groups
        encoder_lr = optimizer.param_groups[0]['lr']
        backbone_lr = optimizer.param_groups[1]['lr']
        decoder_lr = optimizer.param_groups[2]['lr']
        print(f'  LR: Encoder={encoder_lr:.6f}, Backbone={backbone_lr:.6f}, Decoder={decoder_lr:.6f}')

        # Experiment tracking: one point per epoch, never per step
        epoch_train_metrics = {
            'loss': avg_train_loss, 'l1': avg_train_l1_loss, 'ciou': avg_train_ciou_loss,
            'cls': avg_train_cls_loss, 'enc': avg_train_enc_loss,
        }
        epoch_val_metrics = {
            'loss': avg_val_loss, 'l1': avg_val_l1_loss, 'ciou': avg_val_ciou_loss,
            'cls': avg_val_cls_loss, 'enc': avg_val_enc_loss,
            'recall_iou05': recall_iou05, 'avg_max_iou': avg_max_iou,
        }
        epoch_lrs = {'encoder': encoder_lr, 'backbone': backbone_lr, 'decoder': decoder_lr}
        # Captured here, not after the loop: scheduler.step() at the bottom of
        # the body advances to the NEXT epoch's rates, so reading param_groups
        # once training is over reports an epoch that never ran. It also differs
        # between the two ways out of the loop - early stopping breaks before the
        # step, running out of epochs does not.
        final_lrs = dict(epoch_lrs)
        architecture = {
            'num_classes': loss_fn.num_classes,
            'num_queries': model.num_queries,
            'num_encoder_layers': model.num_encoder_layers,
            'num_decoder_layers': model.num_decoder_layers,
            'num_aux_layers': model.num_aux_layers,
        }
        tracker.log_epoch(epoch, epoch_train_metrics, epoch_val_metrics, epoch_lrs)

        # Early Stopping check
        #
        # `is_best` decides which weights land in best_coco_detection_head.pth, so
        # it follows ONE criterion. Letting either metric set it meant the file
        # could hold the best-recall epoch while the manifest beside it reported a
        # best_val_loss from a different epoch. Loss is the criterion because
        # recall here is a coarse single-threshold proxy.
        #
        # `improved` is separate and stays permissive: either metric moving counts
        # as progress for the early-stopping counter.
        is_best = False
        improved = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            is_best = True
            improved = True
        if recall_iou05 > best_val_acc:
            best_val_acc = recall_iou05
            improved = True
        
        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'  WARNING: No improvement for {epochs_no_improve} epoch(s)')
            
        # Every checkpoint gets a sibling .json recording exactly what produced it
        def _save(filename):
            path = os.path.join(save_dir, filename)
            torch.save(build_checkpoint(model, optimizer, scheduler, scaler, epoch,
                                        best_val_loss, best_val_acc, epochs_no_improve,
                                        loss_fn), path)
            tracker.write_manifest(path, epoch, epoch_train_metrics, epoch_val_metrics,
                                   epoch_lrs, best_val_loss, best_val_acc,
                                   epochs_no_improve, architecture=architecture)
            return path

        if is_best:
            _save('best_coco_detection_head.pth')
            print(f'  -> Save best model (val_loss: {best_val_loss:.4f}, recall: {best_val_acc*100:.2f}%)')

        if (epoch + 1) % 5 == 0:
            _save(f'coco_detection_head_epoch{epoch+1}.pth')
            print(f'  -> Save checkpoint: epoch {epoch+1}')

        # Early Stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f'\nEarly Stopping triggered! No improvement for {early_stopping_patience} epochs.')
            print(f'Best Val Loss: {best_val_loss:.4f}, Best Recall: {best_val_acc*100:.2f}%')
            break

        # Advance the cosine curve to the next epoch. Last statement in the body
        # so everything above - logging, manifests, the checkpoint's optimizer
        # state - records the rates this epoch actually trained with.
        scheduler.step()

    print(f'\nTraining completed!')
    print(f'Best Val Loss: {best_val_loss:.4f}, Best Recall: {best_val_acc*100:.2f}%')
    # best_val_loss / best_val_acc are returned rather than recomputed by the
    # caller: they span the whole run (seeded from the checkpoint on resume),
    # whereas val_losses/val_accuracies only cover the epochs of THIS segment.
    #
    # final_lrs likewise: the caller only has the CONFIGURED bases, and under a
    # cosine schedule the last epoch trains at a small fraction of them - 1.03e-6
    # against a configured 1e-4 over 100 epochs - so a manifest built from the
    # config misreports the final state by two orders of magnitude. It is filled
    # in inside the loop, before the scheduler advances.
    return (model, train_losses, val_losses, val_accuracies,
            best_val_loss, best_val_acc, final_lrs)



def main():
    # ==================== TRAINING CONFIGURATION ====================
    # Modify these parameters to customize your training
    
    # === Model Architecture ===
    # Not a free knob: model_loader.py and eval_coco.py rebuild the model with
    # this same constant, and any other value yields a checkpoint they cannot
    # load. DetectionLoss rejects anything else outright.
    num_classes = COCO_NUM_CLASSES  # COCO classes (raw category_ids 1..90 + background at 0)
    # Back to 100. The move to 200 was a bet that per-object centre blocks would
    # let the extra queries find distinct objects; measured on the 50-epoch run,
    # they did not. AR@10=0.508 vs AR@100=0.529 at 200 queries is the same 0.021
    # marginal return as AR@10=0.364 vs AR@100=0.387 was at 100 - and COCO has
    # about 7 instances per image, so the top 10 already cover most of them.
    # 200 queries bought nothing, cost twice the decoder, and doubled the
    # background mass in the query-mean classification loss.
    num_queries = 100              # Number of object queries (100-300)
    num_encoder_layers = 2         # Transformer encoder layers (2-6) 太多層encoder很難訓練，backbone本身的特徵提取能力就很強了，所以不需太多層
    num_decoder_layers = 6         # Transformer decoder layers (4-8)
    num_aux_layers = 2             # Intermediate decoder layers feeding aux loss (0 to disable)

    # === Data & Training ===
    input_resolution = 800         # Input image resolution (384, 512, 640)
    # Measured: batch 8 at 800px used only 2.7GB of the 16GB card, so activations
    # cost roughly 0.2GB/sample on top of ~1GB fixed overhead. 32 should land
    # near 7-8GB, leaving room for fragmentation and the validation pass.
    batch_size = 32                # Batch size (depends on VRAM)
    num_workers = 8                # DataLoader workers; ~= physical CPU cores
    target_total_epochs = 100      # Total training epochs (30-100)
    # Raised from 8. The 50-epoch run never triggered it, but it came within one
    # epoch (a streak of 4 at epochs 42-45) purely from val-loss noise at a flat
    # LR. This run adds a cosine ramp-down plus new canvases and augmentation,
    # each of which buys a few epochs of re-adaptation - 8 would fire on that.
    early_stopping_patience = 20   # Early stopping patience (5-15) 多久沒有提升就停止
    
    # === Learning Rates ===
    # LR is coupled to batch_size: 8 -> 32 means 4x fewer optimizer steps per
    # epoch, so keeping 3e-5 would train strictly slower than before. Scaled to
    # 1e-4, which is also DETR's reference value for its transformer at batch 64.
    # Warmup needs >= 2 epochs to do anything (the ramp is (epoch+1)/N).
    warmup_epochs = 3              # Warmup epochs for encoder; 0 disables
    encoder_warmup_lr = 0.00003    # Encoder starting LR during warmup
    encoder_target_lr = 0.0001     # Encoder target LR after warmup
    backbone_lr = 0.0001           # patch_proj / cls_proj LR (backbone is frozen)
    decoder_lr = 0.0001            # Decoder & heads LR
    
    # === Loss Weights ===
    bbox_loss_weight = 8           # L1 bounding box loss weight (5.0-15.0)
    # L1 on normalized cxcywh is scale-DEPENDENT: the same 50% relative error
    # costs 0.058 on a small box and 0.292 on a large one. CIoU is scale
    # invariant and is the only term counteracting that, so at 8/2 a small
    # object received 46% of the box loss a large one did for the same mistake.
    # Measured small/large loss ratio: 8/2 -> 0.458, 5/2 -> 0.546, 5/5 -> 0.725.
    # Raised to 5.0 to close most of the gap without inverting it.
    ciou_loss_weight = 5.0         # CIoU loss weight (2.0-8.0)
    # NOTE: the classification loss is now a DETR-style weighted mean over all
    # queries instead of a sum divided by the matched-target count, so its
    # magnitude dropped roughly 10x. This weight was re-tuned accordingly.
    cls_loss_weight = 2.0          # Classification loss weight (1.0-5.0)
    # Was 0, which routed through `if aux_loss_weight <= 0: num_aux_layers = 0`
    # below and disabled the auxiliary heads entirely - 6 decoder layers with
    # supervision on the last one only. Re-enabled at the DETR-ish default.
    # NOTE: this makes aux_rematch live for the first time (2 extra Hungarian
    # matchings per step), so expect epoch time to rise above the 40.7 min
    # baseline measured while the aux path was dead.
    aux_loss_weight = 0.4          # Auxiliary loss weight (0.2-0.6, 0 to disable)
    eos_coef = 0.1                 # Background class weight in cls loss (DETR default)
    enc_loss_weight = 1.0          # Encoder objectness loss (trains Mixed Query Selection)

    # === Training Stability ===
    gradient_clip_max_norm = 3.0   # Gradient clipping max norm (1.0-10.0)
    
    # === Paths & Resume ===
    save_dir = 'runs'             # Checkpoint save directory
    resume_training = True        # Resume from latest checkpoint
    weights_path = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    # Checkpoints written before loss_semantics_version existed carry no record
    # of which bbox convention their weights use. Version 2 stored ABSOLUTE
    # coordinates; version 3 onward stores an offset from the query's reference
    # point. The tensor shapes are identical, so loading the wrong one succeeds
    # silently and every predicted box is wrong. Resuming such a checkpoint
    # therefore requires saying out loud which convention it uses.
    #   None -> refuse to resume an unversioned checkpoint (default)
    #   int  -> assert it was written under that loss_semantics_version
    assume_loss_semantics_version = None

    # === Experiment Tracking ===
    use_mlflow = True              # False to skip MLflow (JSON manifests stay on)
    experiment_name = 'dinov3-detr'

    # ================================================================

    # Mirror stdout into runs/train.log so a background run always leaves a log.
    # Do NOT additionally redirect the shell to this same file.
    os.makedirs(save_dir, exist_ok=True)
    log_path = setup_file_logging(save_dir)
    print(f"Logging to: {log_path}")

    # Performance optimization configuration
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    os.makedirs('./data/coco', exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created checkpoint directory: {save_dir}/")

    # Display configuration
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    # Skip the extra head evaluations entirely when the aux loss is switched off.
    if aux_loss_weight <= 0:
        num_aux_layers = 0

    print(f"Model Architecture:")
    print(f"  Classes: {num_classes}, Queries: {num_queries}")
    print(f"  Encoder layers: {num_encoder_layers}, Decoder layers: {num_decoder_layers}, "
          f"Aux layers: {num_aux_layers}")
    print(f"\nData & Training:")
    print(f"  Resolution: {input_resolution}×{input_resolution} ({input_resolution//16}×{input_resolution//16} patches)")
    print(f"  Batch size: {batch_size}, Epochs: {target_total_epochs}")
    print(f"  Early stopping: {early_stopping_patience} epochs")
    print(f"\nLearning Rates:")
    print(f"  Encoder: {encoder_warmup_lr} → {encoder_target_lr} ({warmup_epochs} warmup epochs)")
    print(f"  Backbone: {backbone_lr}, Decoder: {decoder_lr}")
    print(f"\nLoss Weights:")
    print(f"  BBox: {bbox_loss_weight}, CIoU: {ciou_loss_weight}, Cls: {cls_loss_weight}, Aux: {aux_loss_weight}")
    print(f"  EOS coef: {eos_coef}, Encoder objectness: {enc_loss_weight}")
    print(f"\nTraining Stability:")
    print(f"  Gradient clip: {gradient_clip_max_norm}")
    # Calibrated against measured runs on this setup (ViT-B/16 @ 800px, AMP,
    # need_weights=False): batch 8 -> 2.7GB, batch 32 -> 6.0GB. The fixed part is
    # weights + CUDA context; the per-sample part scales with the token count,
    # i.e. with resolution squared. The previous formula was not based on any
    # measurement and reported 34.7GB for a run that actually used 6GB.
    vram_fixed_gb = 1.6
    vram_per_sample_gb = 0.1375 * (input_resolution / 800) ** 2
    print(f"\nEstimated VRAM: ~{vram_fixed_gb + batch_size * vram_per_sample_gb:.1f}GB (with AMP)")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading DINOv3 ViT-B/16 weights: {weights_path}")
    model = DINOv3DetectionModel(
        weights_path=weights_path,
        num_classes=num_classes,
        num_queries=num_queries,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_aux_layers=num_aux_layers
    )

    start_epoch = 0
    checkpoint_path = None
    if resume_training:
        found_path, found_epoch = find_latest_checkpoint(save_dir, False)
        if found_path:
            checkpoint_path = found_path
            if found_epoch > 0:
                start_epoch = found_epoch
                print(f"Found latest checkpoint: {checkpoint_path} (Epoch {start_epoch})")
            else:
                print(f"Found best model checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found, starting from scratch.")
            resume_training = False

    num_epochs = target_total_epochs - start_epoch
    if num_epochs <= 0:
        print(f"Training already reached target epoch {target_total_epochs}. Exiting.")
        return

    checkpoint = None
    if resume_training and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}...")
        try:
            # weights_only=True is the safe path and works for our checkpoints
            # (plain tensors, floats, ints). Catch broadly: older torch raises
            # TypeError for the unknown kwarg, newer torch raises
            # UnpicklingError if anything is not on its allowlist.
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as e:
            print(f"weights_only=True load failed ({type(e).__name__}), retrying: {e}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Establish which bbox convention these weights use BEFORE loading them.
        # bbox_head has the same shape under every version, so a mismatch does
        # not raise - it just makes every box wrong. Falls back to the sibling
        # manifest, which has recorded the version since before the checkpoint
        # itself did.
        ckpt_version = checkpoint.get('loss_semantics_version')
        version_source = 'checkpoint'
        if ckpt_version is None:
            manifest_path = os.path.splitext(checkpoint_path)[0] + '.json'
            try:
                with open(manifest_path) as fh:
                    manifest = json.load(fh)
                # A malformed manifest must fall through to the refusal below,
                # not crash. `except ValueError` covers unparseable JSON, but a
                # well-formed non-object ("[]", "42") parses fine and would then
                # raise AttributeError on .get().
                if isinstance(manifest, dict):
                    ckpt_version = manifest.get('loss_semantics_version')
                    if ckpt_version is not None:
                        version_source = f'manifest ({os.path.basename(manifest_path)})'
                else:
                    print(f"Ignoring {manifest_path}: expected a JSON object, "
                          f"got {type(manifest).__name__}")
            except (OSError, ValueError):
                pass
        # Validate the VALUE, not just the manifest's container type. A manifest
        # reading {"loss_semantics_version": "4"} parses fine and survives the
        # isinstance(dict) check above, then raises
        #   TypeError: '<' not supported between instances of 'str' and 'int'
        # at the `< 3` comparison below - a crash instead of the refusal path
        # that exists precisely for unusable version information. Discarding it
        # here also lets assume_loss_semantics_version rescue the run.
        # bool is excluded explicitly: it passes isinstance(x, int) and True < 3
        # would silently evaluate as 1 < 3.
        if ckpt_version is not None and (not isinstance(ckpt_version, int)
                                         or isinstance(ckpt_version, bool)):
            print(f"Ignoring loss_semantics_version {ckpt_version!r} from "
                  f"{version_source}: expected an int, got "
                  f"{type(ckpt_version).__name__}")
            ckpt_version = None

        if ckpt_version is None and assume_loss_semantics_version is not None:
            ckpt_version = assume_loss_semantics_version
            version_source = 'assume_loss_semantics_version'

        if ckpt_version is None:
            print(f"Error: {checkpoint_path} records no loss_semantics_version, "
                  f"and no manifest was found beside it.")
            print("  Version 2 checkpoints store ABSOLUTE box coordinates; version 3+")
            print("  store an offset from the query's reference point. The shapes match,")
            print("  so loading the wrong one is silent and every box comes out wrong.")
            print("  Set assume_loss_semantics_version in the config to the version this")
            print("  checkpoint was trained under, or start from scratch.")
            # Not `return`: a plain return leaves the process at exit code 0, and
            # a background runner would file "never started training" as success.
            # This is a refusal, unlike the "already at target epoch" paths above.
            sys.exit(1)
        if ckpt_version < 3:
            print(f"Error: {checkpoint_path} is loss_semantics_version {ckpt_version} "
                  f"(source: {version_source}).")
            print("  Its bbox_head predicts absolute coordinates, which this model no")
            print("  longer interprets that way. The weights are not convertible.")
            print("  Train from scratch instead.")
            sys.exit(1)
        print(f"Checkpoint loss_semantics_version: {ckpt_version} (source: {version_source})")

        # Write the resolved version back. train_detection_model() re-reads this
        # key off the checkpoint to decide whether the early-stopping bookkeeping
        # is on the current loss scale, and it has no access to the manifest or
        # to assume_loss_semantics_version. Without this the two fallbacks are
        # dead weight: a checkpoint whose manifest says v4 would still look
        # unversioned there, and the mismatch branch would wipe best_val_loss,
        # best_val_acc and the scheduler state - overwriting a perfectly good
        # best model on the first epoch.
        checkpoint['loss_semantics_version'] = ckpt_version

        try:
            model.patch_proj.load_state_dict(checkpoint['patch_proj'])
            model.cls_proj.load_state_dict(checkpoint['cls_proj'])
            
            # Load Encoder weights (new format only)
            model.encoder.load_state_dict(checkpoint['encoder'])
            print("Loaded Transformer Encoder weights")
            
            # Load topk_score_head
            if 'topk_score_head' in checkpoint:
                model.topk_score_head.load_state_dict(checkpoint['topk_score_head'])
            
            # Load Decoder weights (new format only)
            model.decoder.load_state_dict(checkpoint['decoder'])
            print("Loaded Transformer Decoder weights")

            # Load Head weights
            model.cls_head.load_state_dict(checkpoint['cls_head'])
            model.bbox_head.load_state_dict(checkpoint['bbox_head'])
            
            # The checkpoint's own epoch field is authoritative - the filename is
            # only a hint, and the best-model file has no epoch in its name.
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Updated start_epoch to {start_epoch}")
                num_epochs = target_total_epochs - start_epoch

            print("Checkpoint loaded successfully.")

            # Re-check AFTER the correction above. The earlier guard ran against
            # the filename-derived epoch, so without this a finished run would
            # fall through with num_epochs <= 0 and "train" for zero epochs.
            if num_epochs <= 0:
                print(f"Checkpoint is already at epoch {start_epoch}, "
                      f"target is {target_total_epochs}. Nothing to do.")
                print(f"Raise target_total_epochs to continue training.")
                return
        except Exception as e:
            print(f"Error: Failed to load checkpoint: {e}")
            print("If checkpoint is from old version, please delete it and retrain from scratch.")
            raise

    else:
        print("Starting training from scratch (Mixed Query Selection enabled).")
        start_epoch = 0
        num_epochs = target_total_epochs

    train_loader, val_loader = load_coco_dataset(batch_size=batch_size,
                                                 input_resolution=input_resolution,
                                                 num_workers=num_workers)

    # Everything the manifest and MLflow need to make this run reproducible.
    tracker = ExperimentTracker(
        save_dir=save_dir,
        config={
            'num_classes': num_classes,
            'num_queries': num_queries,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'num_aux_layers': num_aux_layers,
            'input_resolution': input_resolution,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'target_total_epochs': target_total_epochs,
            'early_stopping_patience': early_stopping_patience,
            'warmup_epochs': warmup_epochs,
            'encoder_warmup_lr': encoder_warmup_lr,
            'encoder_target_lr': encoder_target_lr,
            'backbone_lr': backbone_lr,
            'decoder_lr': decoder_lr,
            'bbox_loss_weight': bbox_loss_weight,
            'ciou_loss_weight': ciou_loss_weight,
            'cls_loss_weight': cls_loss_weight,
            'aux_loss_weight': aux_loss_weight,
            'eos_coef': eos_coef,
            'enc_loss_weight': enc_loss_weight,
            'gradient_clip_max_norm': gradient_clip_max_norm,
            'start_epoch': start_epoch,
            'weights_path': weights_path,
        },
        use_mlflow=use_mlflow,
        experiment_name=experiment_name,
        resumed_from=checkpoint_path if resume_training else None,
    )

    (trained_model, train_losses, val_losses, val_accuracies,
     best_val_loss, best_val_acc, final_lrs) = train_detection_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        start_epoch=start_epoch,
        resume_checkpoint=checkpoint,
        early_stopping_patience=early_stopping_patience,
        save_dir=save_dir,
        # Learning rate parameters
        warmup_epochs=warmup_epochs,
        encoder_warmup_lr=encoder_warmup_lr,
        encoder_target_lr=encoder_target_lr,
        backbone_lr=backbone_lr,
        decoder_lr=decoder_lr,
        # Loss weights
        bbox_loss_weight=bbox_loss_weight,
        ciou_loss_weight=ciou_loss_weight,
        cls_loss_weight=cls_loss_weight,
        aux_loss_weight=aux_loss_weight,
        eos_coef=eos_coef,
        enc_loss_weight=enc_loss_weight,
        # Training stability
        gradient_clip_max_norm=gradient_clip_max_norm,
        tracker=tracker
    )
    print("training done")

    final_save_dict = {
        'patch_proj': trained_model.patch_proj.state_dict(),
        'cls_proj': trained_model.cls_proj.state_dict(),
        'encoder': trained_model.encoder.state_dict(),
        'topk_score_head': trained_model.topk_score_head.state_dict(),
        'decoder': trained_model.decoder.state_dict(),
        'cls_head': trained_model.cls_head.state_dict(),
        'bbox_head': trained_model.bbox_head.state_dict(),
        'num_queries': trained_model.num_queries,
        'num_encoder_layers': trained_model.num_encoder_layers,
        'num_decoder_layers': trained_model.num_decoder_layers,
        'num_aux_layers': trained_model.num_aux_layers,
    }
    
    final_path = os.path.join(save_dir, 'final_coco_detection_head_vitb16.pth')
    torch.save(final_save_dict, final_path)
    print(f"Final model saved to: {final_path}")

    # best_val_loss / best_val_acc already track the whole run: they are seeded
    # from the checkpoint on resume and updated each epoch. Recomputing them from
    # val_losses / val_accuracies - which only cover the epochs of THIS segment -
    # made a resumed run report the best of the resumed part as if it were the
    # best overall, hiding a better earlier epoch.
    if val_losses:
        tracker.write_manifest(
            final_path, start_epoch + len(val_losses) - 1,
            {'loss': train_losses[-1]},
            {'loss': val_losses[-1], 'recall_iou05': val_accuracies[-1]},
            final_lrs,
            best_val_loss, best_val_acc, 0,
            architecture={
                'num_classes': num_classes,
                'num_queries': trained_model.num_queries,
                'num_encoder_layers': trained_model.num_encoder_layers,
                'num_decoder_layers': trained_model.num_decoder_layers,
                'num_aux_layers': trained_model.num_aux_layers,
            })

    tracker.close(best_val_loss=best_val_loss, best_val_acc=best_val_acc,
                  final_checkpoint=final_path)

    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(train_losses, label='Train Loss', marker='o')
        ax1.plot(val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch (Relative)')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(val_accuracies, label='Val Recall@IoU0.5', marker='s', color='green')
        ax2.set_xlabel('Epoch (Relative)')
        ax2.set_ylabel('Recall')
        ax2.set_title('Validation Recall')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves_resumed.png'), dpi=300, bbox_inches='tight')
        print(f"Training curve saved: {save_dir}/training_curves_resumed.png")
    except Exception as e:
        print(f"Cannot plot curve: {e}")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # stderr is not teed into the log file, so a crash in a background run
        # would otherwise leave no trace there.
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
