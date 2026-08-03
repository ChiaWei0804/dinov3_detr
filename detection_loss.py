"""
Detection Loss Functions and Classes

This module contains loss functions and utilities for object detection:
- Complete IoU (CIoU) calculation
- Hungarian matching for assignment
- DetectionLoss class combining L1, CIoU, and Classification losses
- Encoder objectness loss that supervises Mixed Query Selection
- Auxiliary loss support (OPTIMIZED for GPU utilization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from coco_dataset import (BACKGROUND_CLASS_INDEX, VALID_COCO_CATEGORY_IDS,
                          COCO_NUM_CLASSES)
from iou_utils import calculate_iou_batch_cxcywh, _safe_iou, _promote


def complete_box_iou(boxes1, boxes2):
    """
    Complete IoU Loss (CIoU)
    Input format: [cx, cy, w, h]
    CIoU = IoU - (ρ²/c²) - α·v
    
    Where:
    - ρ²: squared distance between center points
    - c²: squared diagonal length of smallest enclosing box
    - α: weight coefficient
    - v: aspect ratio consistency
    """
    # 1. Convert to [x1, y1, x2, y2]
    # Widen first: in fp16 the area of a 1e-4 box underflows to zero, so every
    # term below - intersection, union, the aspect-ratio arctans - would be
    # computed from values that have already lost their magnitude.
    boxes1, boxes2 = _promote(boxes1), _promote(boxes2)

    cx1, cy1, w1, h1 = boxes1.unbind(-1)
    x1_1 = cx1 - 0.5 * w1
    y1_1 = cy1 - 0.5 * h1
    x2_1 = cx1 + 0.5 * w1
    y2_1 = cy1 + 0.5 * h1

    cx2, cy2, w2, h2 = boxes2.unbind(-1)
    x1_2 = cx2 - 0.5 * w2
    y1_2 = cy2 - 0.5 * h2
    x2_2 = cx2 + 0.5 * w2
    y2_2 = cy2 + 0.5 * h2

    # 2. Calculate intersection
    x1_i = torch.max(x1_1, x1_2)
    y1_i = torch.max(y1_1, y1_2)
    x2_i = torch.min(x2_1, x2_2)
    y2_i = torch.min(y2_1, y2_2)
    intersection = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)

    # 3. Calculate union and IoU
    # Areas come from the converted corners, not from raw w*h: under AMP the two
    # disagree by enough that identical boxes scored 1.0049, which made the
    # `1 - ciou` loss negative and flipped the sign of alpha's denominator below.
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    iou = _safe_iou(intersection, union)

    # 4. Calculate center distance (ρ²)
    center_distance_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # 5. Calculate diagonal of smallest enclosing box (c²)
    x1_c = torch.min(x1_1, x1_2)
    y1_c = torch.min(y1_1, y1_2)
    x2_c = torch.max(x2_1, x2_2)
    y2_c = torch.max(y2_1, y2_2)
    diagonal_distance_sq = ((x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2).clamp(min=1e-7)

    # 6. Calculate aspect ratio consistency (v)
    arctan1 = torch.atan(w1 / (h1 + 1e-7))
    arctan2 = torch.atan(w2 / (h2 + 1e-7))
    v = (4 / (torch.pi ** 2)) * torch.pow(arctan1 - arctan2, 2)

    # 7. Calculate alpha
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + 1e-7)

    # 8. Calculate CIoU
    ciou = iou - (center_distance_sq / diagonal_distance_sq) - alpha * v
    
    return ciou


def hungarian_matching_simple(cost_matrix):
    """
    Use Scipy's optimized version, extremely fast, no loops needed
    cost_matrix: Tensor [N, M]
    
    Handles NaN/Inf values robustly to prevent linear_sum_assignment errors:
    - NaN values are replaced with a large finite cost (1e6)
    - Inf values are clipped to a large finite cost (1e6)
    """
    C = _sanitize_cost_matrix(cost_matrix).numpy()
    row_ind, col_ind = linear_sum_assignment(C)
    matches = list(zip(row_ind, col_ind))
    return matches


def _sanitize_cost_matrix(cost_matrix):
    """
    Move a cost matrix to the CPU as float32 and strip any NaN/Inf.

    .float() is mandatory, not cosmetic: under AMP the cost matrix arrives as
    fp16, whose max representable value is 65504. Casting the 1e6 sentinel to
    fp16 turns it straight back into inf, so the sanitisation would be a no-op
    and scipy would reject the matrix.
    """
    C = cost_matrix.detach().float().cpu()
    C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)

    # Additional safety check: ensure all values are finite
    if not torch.all(torch.isfinite(C)):
        # This should rarely happen after nan_to_num, but just in case
        C = torch.where(torch.isfinite(C), C, torch.full_like(C, 1e6))

    return C


class DetectionLoss(nn.Module):
    """
    Detection Loss with L1 + CIoU + Classification + Auxiliary Loss
    
    OPTIMIZED for GPU utilization:
    - GT data transferred to GPU only ONCE
    - Hungarian matching done only ONCE
    - All losses computed on GPU with pre-loaded data
    
    This keeps GPU utilization high and avoids CPU bottlenecks.
    """
    def __init__(self, bbox_loss_weight=5.0, ciou_loss_weight=5.0, cls_loss_weight=2.0,
                 num_classes=COCO_NUM_CLASSES, num_queries=100, max_targets_per_image=100,
                 aux_loss_weight=0.4, eos_coef=0.1, enc_loss_weight=1.0,
                 aux_rematch=True, enc_center_radius=1):
        """
        Initialize detection loss function.

        Args:
            bbox_loss_weight: Weight for L1 bbox loss (default: 5.0)
            ciou_loss_weight: Weight for CIoU loss (default: 5.0)
            cls_loss_weight: Weight for classification loss (default: 2.0)
            num_classes: Number of object classes (default: 91 for COCO)
            num_queries: Number of object queries (default: 100)
            max_targets_per_image: Maximum number of targets per image (default: 100)
            aux_loss_weight: Weight for auxiliary losses from intermediate layers (default: 0.4)
            eos_coef: Relative weight of the background class in the classification
                loss (default: 0.1, same as DETR). Most queries are unmatched, so
                without this the background term drowns out the foreground term.
            enc_loss_weight: Weight of the encoder objectness loss that supervises
                Mixed Query Selection (default: 1.0)
            enc_center_radius: Half-width, in patches, of the positive block the
                encoder objectness loss places on each GT centre (default: 1,
                i.e. 3x3). The block is clipped to the GT box, so this is an
                upper bound. Larger values re-introduce the area bias that lets
                one big object monopolise the top-K query budget.
            aux_rematch: Run Hungarian matching separately for every auxiliary
                head (default: True, which is what DETR does). Reusing the final
                layer's assignment is cheaper - one matching per step instead of
                one per decoder layer - but an intermediate layer whose best
                assignment differs then gets gradient pushed onto the wrong
                query. Set False to trade that correctness back for speed.
        """
        super().__init__()
        self.bbox_loss_weight = bbox_loss_weight
        self.ciou_loss_weight = ciou_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.max_targets_per_image = max_targets_per_image
        self.aux_loss_weight = aux_loss_weight
        self.enc_loss_weight = enc_loss_weight
        self.eos_coef = eos_coef
        self.aux_rematch = aux_rematch
        # Validate here rather than letting it fail inside the loss: a bad radius
        # survives construction and only blows up on the first batch that
        # contains a GT - minutes into a run - as `IndexError: tensors used as
        # indices must be long...` (float radius) or a torch.arange RuntimeError
        # (negative radius), neither of which names the parameter at fault.
        if not isinstance(enc_center_radius, int) or isinstance(enc_center_radius, bool):
            raise TypeError(
                f"enc_center_radius must be an int, got "
                f"{type(enc_center_radius).__name__}: {enc_center_radius!r}")
        if enc_center_radius < 0:
            raise ValueError(
                f"enc_center_radius must be >= 0, got {enc_center_radius}. "
                f"0 marks only each GT's centre patch; 1 (default) a 3x3 block.")
        self.enc_center_radius = enc_center_radius

        # cross_entropy indexes the logits with the raw COCO category_id, so the
        # head must have a column for the LARGEST id (90), not for the 80 ids
        # that actually exist. Configuring num_classes=80 looks reasonable and
        # fails at the first annotation with category_id > 80, several minutes
        # into a run, as `IndexError: Target 90 is out of bounds`.
        # Reject on inequality, not on "< 90". 90 avoids the IndexError but still
        # builds a 91-column cls_head that model_loader.py cannot load, and 92+
        # is wrong in the other direction - so the only correct test is equality
        # against the single value every builder shares.
        if num_classes != COCO_NUM_CLASSES:
            max_category_id = max(VALID_COCO_CATEGORY_IDS)
            reason = (f"below {max_category_id}, so cross_entropy would index "
                      f"out of bounds on a raw COCO category_id"
                      if num_classes < max_category_id else
                      f"not the value model_loader.py and eval_coco.py rebuild "
                      f"the model with, so the checkpoint would fail to load")
            raise ValueError(
                f"num_classes={num_classes} is invalid: it is {reason}. Use "
                f"coco_dataset.COCO_NUM_CLASSES ({COCO_NUM_CLASSES}). The "
                f"unused ids in 1..{COCO_NUM_CLASSES} stay as dead columns; "
                f"decode_class_predictions masks them out.")

        # Use L1 Loss, reduction='none'
        self.l1_loss = nn.L1Loss(reduction='none')

        # Per-class weights for the classification loss; background is damped.
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[BACKGROUND_CLASS_INDEX] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_scale_signature(self):
        """
        Every setting that moves the MAGNITUDE of the returned loss.

        LOSS_SEMANTICS_VERSION covers changes to what the loss MEANS, but it is a
        hand-maintained constant: flipping aux_loss_weight from 0.4 to 0 shifts
        total loss to 0.72x of its former value while the version stays put, and
        a resumed run would read the smaller number as an improvement and
        overwrite a genuinely better best model on its first epoch. Comparing
        this signature catches the configuration half automatically; the version
        constant still covers the code half.

        eos_coef is included even though it only reweights the classification
        term, and enc_center_radius because it changes how many positives the
        encoder objectness BCE sees.

        num_queries is here despite being a MODEL parameter rather than a loss
        one: the classification term is a weighted mean over every query, so
        adding queries without adding GT raises the share of background targets
        and moves the mean. Measured at 100 -> 300 queries with everything else
        held constant: total 17.39 -> 19.47 (1.12x). The state dict loads clean
        across that change - the heads do not depend on the query count - so
        nothing else would catch it.
        """
        return {
            'num_queries': int(self.num_queries),
            'bbox_loss_weight': float(self.bbox_loss_weight),
            'ciou_loss_weight': float(self.ciou_loss_weight),
            'cls_loss_weight': float(self.cls_loss_weight),
            'aux_loss_weight': float(self.aux_loss_weight),
            'enc_loss_weight': float(self.enc_loss_weight),
            'eos_coef': float(self.eos_coef),
            'enc_center_radius': int(self.enc_center_radius),
        }

    def calculate_iou_batch(self, boxes1, boxes2):
        """Calculate IoU batch matrix using unified iou_utils module"""
        return calculate_iou_batch_cxcywh(boxes1, boxes2)

    def _prepare_targets_gpu(self, targets, device):
        """
        Move every GT in the batch to the device with exactly TWO transfers.

        Per-image entries expose `gt_bboxes` / `gt_classes` as slices of the flat
        tensors, so they are free views rather than separate allocations, and
        `offset` lets the loss address a target by its position in the flat
        tensor. Transferring per image instead costs 2*batch_size pageable
        host-to-device copies, each of which is synchronous.

        Returns:
            batch_gt_data: per-image dicts (views into the flat tensors)
            flat_boxes:   [total_gt, 4] every GT box in the batch
            flat_classes: [total_gt] matching class ids
            gt_batch_ids: [total_gt] which image each GT belongs to
        """
        kept_per_image = []
        all_boxes, all_classes, batch_ids = [], [], []

        for i, image_targets in enumerate(targets):
            num_targets = min(len(image_targets), self.max_targets_per_image)
            kept = image_targets[:num_targets]
            kept_per_image.append(num_targets)
            for t in kept:
                all_boxes.append(t['bbox'])
                all_classes.append(t['category_id'])
                batch_ids.append(i)

        if all_boxes:
            flat_boxes = torch.stack(all_boxes).to(device, non_blocking=True)
            flat_classes = torch.stack(all_classes).to(device, dtype=torch.long,
                                                       non_blocking=True)
            gt_batch_ids = torch.as_tensor(batch_ids, dtype=torch.long, device=device)
        else:
            flat_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            flat_classes = torch.zeros((0,), dtype=torch.long, device=device)
            gt_batch_ids = torch.zeros((0,), dtype=torch.long, device=device)

        batch_gt_data = []
        offset = 0
        for num_targets in kept_per_image:
            batch_gt_data.append({
                'gt_bboxes': flat_boxes[offset:offset + num_targets] if num_targets else None,
                'gt_classes': flat_classes[offset:offset + num_targets] if num_targets else None,
                'num_targets': num_targets,
                'offset': offset,
            })
            offset += num_targets

        return batch_gt_data, flat_boxes, flat_classes, gt_batch_ids

    def forward(self, outputs, targets):
        """
        Calculate total detection loss with auxiliary losses (FULLY OPTIMIZED).

        Optimizations:
        1. GT data loaded to GPU ONCE (not 6 times)
        2. All computations on GPU (minimize CPU involvement)

        Matching runs once per head by default (aux_rematch=True), matching
        DETR. Set aux_rematch=False to reuse the final layer's assignment for
        every auxiliary head, which is faster but assigns gradient to queries
        the intermediate layer did not choose.

        This keeps GPU utilization high (>80%) and training fast.

        Args:
            outputs: Model predictions dict with 'pred_logits', 'pred_boxes', and
                optionally 'aux_outputs' / 'enc_objectness'
            targets: Ground truth targets (list of lists of dicts)

        Returns:
            total_loss: Weighted sum of all losses (main + auxiliary + encoder)
            l1_loss: Average L1 loss (from final layer)
            ciou_loss: Average CIoU loss (from final layer)
            cls_loss: Average classification loss (from final layer)
            enc_loss: Encoder objectness loss (Mixed Query Selection supervision)
        """
        # Get final predictions
        if isinstance(outputs, dict):
            final_logits = outputs['pred_logits']
            final_boxes = outputs['pred_boxes']
        else:
            # Backward compatibility
            final_logits = outputs[:, :, :self.num_classes + 1]
            final_boxes = outputs[:, :, self.num_classes + 1:]

        device = final_logits.device

        # === OPTIMIZATION 1: Pre-load GT data to GPU ONCE ===
        batch_gt_data, flat_boxes, flat_classes, gt_batch_ids = \
            self._prepare_targets_gpu(targets, device)

        # === OPTIMIZATION 2: Hungarian matching ONCE on final predictions ===
        all_matches = self._do_hungarian_matching_fast(
            final_logits, final_boxes, batch_gt_data, flat_boxes, flat_classes, gt_batch_ids
        )

        # Flatten the matches into index tensors ONCE, with three host-to-device
        # copies for the whole batch. The main head and every auxiliary head then
        # reuse them, so adding aux layers costs no extra transfers.
        match_index = self._flatten_matches(all_matches, batch_gt_data, device)

        # Box losses are normalised by the number of matched targets in the whole
        # batch, so the same value has to be used for the main and the auxiliary
        # heads (otherwise their magnitudes are not comparable).
        num_boxes = max(sum(len(m) for m in all_matches), 1)

        # === Compute main loss ===
        main_loss, l1_loss, ciou_loss, cls_loss = self._compute_loss_fast(
            final_logits, final_boxes, match_index, flat_boxes, flat_classes, num_boxes
        )

        # === Compute auxiliary losses (reuse GT data and matches) ===
        aux_loss = torch.zeros((), device=device)
        aux_outputs = outputs.get('aux_outputs') if isinstance(outputs, dict) else None
        if aux_outputs:
            for aux_output in aux_outputs:
                if self.aux_rematch:
                    # DETR matches each decoder layer independently. Reusing the
                    # final layer's assignment sends this layer's gradient to
                    # whichever query the LAST layer picked, which is not
                    # necessarily the one this layer predicts best.
                    aux_matches = self._do_hungarian_matching_fast(
                        aux_output['pred_logits'], aux_output['pred_boxes'],
                        batch_gt_data, flat_boxes, flat_classes, gt_batch_ids
                    )
                    aux_match_index = self._flatten_matches(
                        aux_matches, batch_gt_data, device)
                else:
                    aux_match_index = match_index

                aux_l, _, _, _ = self._compute_loss_fast(
                    aux_output['pred_logits'],
                    aux_output['pred_boxes'],
                    aux_match_index,
                    flat_boxes,     # ← REUSE pre-loaded GT data!
                    flat_classes,
                    num_boxes       # matching is complete on the GT side, so the
                                    # count is the same for every layer
                )
                aux_loss = aux_loss + aux_l

            # Average auxiliary losses
            aux_loss = aux_loss / len(aux_outputs)

        # === Encoder objectness loss (supervises Mixed Query Selection) ===
        enc_loss = torch.zeros((), device=device)
        if isinstance(outputs, dict) and outputs.get('enc_objectness') is not None:
            enc_loss = self._encoder_objectness_loss(
                outputs['enc_objectness'], outputs['enc_grid'], flat_boxes, gt_batch_ids
            )

        # Total loss
        total_loss = (main_loss
                      + self.aux_loss_weight * aux_loss
                      + self.enc_loss_weight * enc_loss)

        return total_loss, l1_loss.item(), ciou_loss.item(), cls_loss.item(), enc_loss.item()

    @staticmethod
    def _flatten_matches(all_matches, batch_gt_data, device):
        """
        Turn the per-image match lists into three flat index tensors.

        `gt_ids` are absolute positions in the flat GT tensors, obtained by
        adding each image's offset to its local GT index.

        Returns (batch_ids, pred_ids, gt_ids) or None if nothing matched.
        """
        batch_ids, pred_ids, gt_ids = [], [], []
        for i, matches in enumerate(all_matches):
            if not matches:
                continue
            offset = batch_gt_data[i]['offset']
            for pred_idx, gt_idx in matches:
                batch_ids.append(i)
                pred_ids.append(pred_idx)
                gt_ids.append(offset + gt_idx)

        if not batch_ids:
            return None

        return (torch.as_tensor(batch_ids, dtype=torch.long, device=device),
                torch.as_tensor(pred_ids, dtype=torch.long, device=device),
                torch.as_tensor(gt_ids, dtype=torch.long, device=device))

    def _encoder_objectness_loss(self, enc_logits, grid_hw, flat_boxes, gt_batch_ids):
        """
        Supervise the Top-K score head used by Mixed Query Selection.

        torch.topk only returns indices, which carry no gradient, and the scores
        themselves appear in no other term - so without this loss
        `topk_score_head` never receives a gradient and query selection stays
        frozen at its random initialisation.

        Target: a small block of patches centred on each GT's centre patch,
        clipped to the GT box.

        NOT "every patch whose centre falls inside a GT box". That coverage-mask
        formulation makes the number of positives proportional to object AREA,
        which is what Mixed Query Selection then spends its query budget on. On a
        50x50 grid one object of 0.5x0.6 supplies 780 of the 2500 patches, so
        torch.topk(100) can legitimately place every single query inside it,
        while eight small objects supply 58 patches between them. Measured
        symptom: AR@10=0.364 -> AR@100=0.387, i.e. 90 extra detections buy 0.023
        because they are duplicates of the same few large objects.

        Capping each GT at a fixed-size block makes the budget per object roughly
        equal regardless of size, so top-K has to spread out to keep scoring.

        Args:
            enc_logits: Tensor [B, num_patches] raw objectness logits
            grid_hw: (h, w) patch grid size
            flat_boxes: [total_gt, 4] every GT box in the batch, normalized cxcywh
            gt_batch_ids: [total_gt] image index each GT belongs to

        Returns:
            Scalar BCE loss
        """
        device = enc_logits.device
        h, w = grid_hw

        # Vectorised over the whole batch: every GT contributes its block in one
        # set of ops, then the blocks are scattered back to their image. Looping
        # per image instead costs a dozen kernel launches per image on tiny
        # tensors, which on Windows dominates the actual arithmetic.
        targets = torch.zeros(enc_logits.shape, device=device, dtype=torch.float32)
        if flat_boxes.numel() > 0:
            boxes = flat_boxes.float()                            # [G, 4] cxcywh
            x1 = (boxes[:, 0] - boxes[:, 2] * 0.5).unsqueeze(1)   # [G, 1]
            y1 = (boxes[:, 1] - boxes[:, 3] * 0.5).unsqueeze(1)
            x2 = (boxes[:, 0] + boxes[:, 2] * 0.5).unsqueeze(1)
            y2 = (boxes[:, 1] + boxes[:, 3] * 0.5).unsqueeze(1)

            # Patch containing each GT centre. clamp guards centres that land
            # exactly on the far edge (cx == 1.0 would index w).
            gx = (boxes[:, 0] * w).long().clamp(0, w - 1)         # [G]
            gy = (boxes[:, 1] * h).long().clamp(0, h - 1)         # [G]

            r = self.enc_center_radius
            offs = torch.arange(-r, r + 1, device=device)
            dy, dx = torch.meshgrid(offs, offs, indexing='ij')
            dy, dx = dy.reshape(1, -1), dx.reshape(1, -1)         # [1, K]

            ny = (gy.unsqueeze(1) + dy).clamp(0, h - 1)           # [G, K]
            nx = (gx.unsqueeze(1) + dx).clamp(0, w - 1)           # [G, K]
            flat_idx = ny * w + nx                                # [G, K] row-major

            # Clip the block to the box so a small object does not claim patches
            # outside itself. The clamps above can also fold two offsets onto the
            # same patch at the grid edge; assigning 1.0 is idempotent so the
            # duplicates are harmless.
            pc_x = (nx.float() + 0.5) / w
            pc_y = (ny.float() + 0.5) / h
            inside = ((pc_x >= x1) & (pc_x <= x2) &
                      (pc_y >= y1) & (pc_y <= y2))                # [G, K]

            # A box smaller than one patch can have NO patch centre inside it,
            # which would leave that object with zero positives and no way to
            # ever be selected. Force its own centre patch on.
            inside[:, offs.numel() ** 2 // 2] = True

            img_idx = gt_batch_ids.unsqueeze(1).expand_as(flat_idx)  # [G, K]
            targets[img_idx[inside], flat_idx[inside]] = 1.0

        # Object patches are a small minority; rebalance so they are not ignored.
        num_pos = targets.sum().clamp(min=1.0)
        num_neg = (targets.numel() - targets.sum()).clamp(min=0.0)
        # The lower clamp matters: with no negatives at all (a GT box covering the
        # whole grid) the ratio is 0, which zeroes every positive term and leaves
        # topk_score_head with no gradient for that step. Fall back to unweighted.
        #
        # The upper clamp was 100, chosen when the target was a coverage mask and
        # positives were plentiful (hundreds per image). Centre blocks make them
        # ~10x sparser - a single-object image now sits near 2491/9 = 277 - so 100
        # would clip the rebalancing it exists to provide on exactly the sparse
        # images that need it most.
        pos_weight = (num_neg / num_pos).clamp(min=1.0, max=400.0)

        return F.binary_cross_entropy_with_logits(
            enc_logits.float(), targets, pos_weight=pos_weight
        )

    def _do_hungarian_matching_fast(self, pred_logits, pred_boxes, batch_gt_data,
                                    flat_boxes, flat_classes, gt_batch_ids):
        """
        Hungarian matching, with the cost of the whole batch built in one shot.

        Two things matter for speed here:

        1. The cost of every GT in the batch is computed with a single set of
           broadcast ops. Looping per image issues ~70 kernels per image on
           tensors of ~7 rows, where launch overhead dwarfs the arithmetic.
        2. Exactly ONE device->host transfer. Copying per image interleaves a
           synchronisation with every scipy call, stalling the GPU batch_size
           times per step.

        Because the flat GT tensors are laid out image-major, the [Q, G] cost
        matrix splits directly into the per-image matrices scipy needs.
        """
        all_matches = [[] for _ in batch_gt_data]
        if flat_boxes.shape[0] == 0:
            return all_matches

        # Clamp bbox predictions to valid range to prevent degenerate boxes
        pred_bboxes = pred_boxes.clamp(min=1e-6, max=1.0)          # [B, Q, 4]

        # Classification cost: -log p(class of GT g) for every query of g's image.
        # Add epsilon to prevent log(0).
        log_probs = torch.log_softmax(pred_logits + 1e-8, dim=-1)  # [B, Q, C+1]
        cls_cost = -log_probs[gt_batch_ids, :, flat_classes]       # [G, Q]

        # Box costs against the queries of the image each GT belongs to
        pred_per_gt = pred_bboxes[gt_batch_ids]                    # [G, Q, 4]
        gt_expanded = flat_boxes.unsqueeze(1)                      # [G, 1, 4]

        l1_cost = torch.abs(pred_per_gt - gt_expanded).sum(dim=-1)             # [G, Q]

        # CIoU, not plain IoU: this term is weighted by ciou_loss_weight and must
        # therefore BE the quantity that weight trains. The two disagree in
        # practice - for GT [0.5,0.5,0.4,0.4], predictions [0.3,0.3,0.3,0.7] and
        # [0.3,0.5,0.2,0.8] rank 0.165 > 0.143 under IoU but 0.069 < 0.084 under
        # CIoU - so matching on IoU hands the GT to the query the loss then has
        # to correct hardest.
        ciou_cost = 1.0 - complete_box_iou(pred_per_gt, gt_expanded)           # [G, Q]

        # The matching criterion must be the loss it feeds. The hardcoded 5.0 on
        # both box terms meant the assignment was chosen under cls:L1:IoU =
        # 1:5:5 while _compute_loss_fast then trained it under 1:4:1 (the
        # configured 2.0/8.0/2.0) - so IoU drove which query owned a GT but
        # barely drove how that query was corrected afterwards. DETR keeps the
        # two identical for exactly this reason.
        cost = (self.cls_loss_weight * cls_cost
                + self.bbox_loss_weight * l1_cost
                + self.ciou_loss_weight * ciou_cost)               # [G, Q]

        # ONE transfer; transpose to [Q, G] so each image is a contiguous slice
        merged = _sanitize_cost_matrix(cost.transpose(0, 1))

        # Pure CPU from here, no further device interaction
        widths = [d['num_targets'] for d in batch_gt_data]
        chunks = torch.split(merged, widths, dim=1)
        for i, chunk in enumerate(chunks):
            if widths[i] == 0:
                continue
            row_ind, col_ind = linear_sum_assignment(chunk.numpy())
            all_matches[i] = list(zip(row_ind, col_ind))

        return all_matches

    def _compute_loss_fast(self, pred_logits, pred_boxes, match_index, flat_boxes,
                           flat_classes, num_boxes):
        """
        Compute loss from flat match indices (fully vectorised).

        Classification follows DETR: every query gets a target (background for the
        unmatched ones), the loss is a mean over all queries, and background is
        down-weighted by eos_coef. Box losses are summed over matched pairs and
        normalised by `num_boxes`.

        The batch dimension is NOT looped over. Iterating per image issues a few
        dozen kernels per image on tensors of ~7 rows, so the launch overhead
        dwarfs the arithmetic - measured at 28% of a whole training step, with
        the GPU idle throughout. Here every matched pair in the batch is handled
        by one set of ops regardless of batch size.

        Args:
            match_index: (batch_ids, pred_ids, gt_ids) flat LongTensors, or None
                when nothing matched anywhere in the batch

        Returns tensors (not floats) so the caller decides when to synchronise.
        """
        device = pred_logits.device
        batch_size, num_queries = pred_logits.shape[0], pred_logits.shape[1]

        target_classes = torch.full((batch_size, num_queries), BACKGROUND_CLASS_INDEX,
                                    dtype=torch.long, device=device)

        if match_index is None:
            total_l1_loss = pred_logits.new_zeros(())
            total_ciou_loss = pred_logits.new_zeros(())
        else:
            batch_ids, pred_ids, gt_ids = match_index

            target_classes[batch_ids, pred_ids] = flat_classes[gt_ids]

            matched_pred_bboxes = pred_boxes[batch_ids, pred_ids]   # [M, 4]
            matched_gt_bboxes = flat_boxes[gt_ids]                  # [M, 4]

            # L1 Loss
            total_l1_loss = self.l1_loss(matched_pred_bboxes, matched_gt_bboxes).sum()

            # CIoU Loss
            ciou = complete_box_iou(matched_pred_bboxes, matched_gt_bboxes)
            total_ciou_loss = (1 - ciou).sum()

        # cross_entropy expects [B, C, Q]; .float() keeps it stable under AMP and
        # matches the dtype of empty_weight.
        avg_cls_loss = F.cross_entropy(
            pred_logits.transpose(1, 2).float(),
            target_classes,
            weight=self.empty_weight.to(device),
        )

        avg_l1_loss = total_l1_loss / num_boxes
        avg_ciou_loss = total_ciou_loss / num_boxes

        # Calculate weighted loss
        total_loss = (self.bbox_loss_weight * avg_l1_loss +
                      self.ciou_loss_weight * avg_ciou_loss +
                      self.cls_loss_weight * avg_cls_loss)

        return total_loss, avg_l1_loss, avg_ciou_loss, avg_cls_loss
