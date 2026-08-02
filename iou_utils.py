"""
IoU (Intersection over Union) Calculation Utilities

This module provides comprehensive IoU calculation functions for object detection:
- Support for different box formats: [cx, cy, w, h] and [x1, y1, x2, y2]
- Single box pair and batch matrix calculations
- Both Python native and PyTorch tensor implementations
- Numerical stability optimizations
"""

import torch


def _promote(boxes):
    """
    Widen fp16/bf16 boxes to fp32 BEFORE any area arithmetic happens.

    Promoting at the division is too late: in fp16 a box of w=h=1e-4 has an area
    of 1e-8, which is below the smallest representable subnormal, so the
    multiplication underflows to exactly 0 and the IoU comes out 0 instead of 1.
    No care taken afterwards can recover a value that is already gone.

    fp64 is left alone so callers doing double-precision geometry keep it.
    """
    return boxes if boxes.dtype in (torch.float32, torch.float64) else boxes.float()


def _safe_iou(intersection, union):
    """
    intersection / union, guarded against zero without distorting small boxes.

    The previous `union.clamp(min=1e-7)` was a floor on the union itself, large
    enough to swallow legitimate boxes: two identical 1e-4 boxes have a union of
    1e-8 and came out at IoU 0.1 instead of 1.0. finfo.tiny sits far below any
    representable area, so only a genuinely zero union is affected.

    The clamp to [0, 1] absorbs the rounding that otherwise lets identical boxes
    score slightly above 1 (measured 1.0049 under AMP), which turned `1 - IoU`
    into a negative loss. Inputs are expected to have been through _promote.
    """
    uni = union.clamp(min=torch.finfo(union.dtype).tiny)
    return (intersection / uni).clamp(0.0, 1.0)


def _is_single_box(boxes):
    """
    True when `boxes` is one box rather than a sequence of them.

    `len(boxes) == 4` alone misreads a (4, 4) NumPy array of four boxes as a
    single box, because its first element is an ndarray rather than a
    list/tuple, so the four rows get unpacked as x1/y1/x2/y2. Anything that
    exposes .ndim answers the question directly; the length test is only a
    fallback for plain Python sequences.
    """
    ndim = getattr(boxes, 'ndim', None)
    if ndim is not None:
        return ndim == 1
    return len(boxes) == 4 and not isinstance(boxes[0], (list, tuple))


def cxcywh_to_xyxy(boxes):
    """
    Convert boxes from [cx, cy, w, h] format to [x1, y1, x2, y2] format.
    
    Args:
        boxes: Tensor or list/array of shape [N, 4] or [4] in [cx, cy, w, h] format
    
    Returns:
        Tensor or list/array of shape [N, 4] or [4] in [x1, y1, x2, y2] format
    """
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        # Python native. .shape is not available on a list, which is what the
        # docstring says this branch accepts, so go through _is_single_box.
        if _is_single_box(boxes):
            cx, cy, w, h = boxes
            return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        else:
            result = []
            for box in boxes:
                cx, cy, w, h = box
                result.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            return result


def xyxy_to_cxcywh(boxes):
    """
    Convert boxes from [x1, y1, x2, y2] format to [cx, cy, w, h] format.
    
    Args:
        boxes: Tensor or list/array of shape [N, 4] or [4] in [x1, y1, x2, y2] format
    
    Returns:
        Tensor or list/array of shape [N, 4] or [4] in [cx, cy, w, h] format
    """
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)
    else:
        # Python native
        if _is_single_box(boxes):
            x1, y1, x2, y2 = boxes
            return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        else:
            result = []
            for box in boxes:
                x1, y1, x2, y2 = box
                result.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            return result


def calculate_iou_xyxy(box1, box2):
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.
    Supports both Python native types and PyTorch tensors.
    
    Args:
        box1: Box in [x1, y1, x2, y2] format (list, tuple, or tensor)
        box2: Box in [x1, y1, x2, y2] format (list, tuple, or tensor)
    
    Returns:
        IoU value (float or tensor)
    """
    if isinstance(box1, torch.Tensor) or isinstance(box2, torch.Tensor):
        return _calculate_iou_xyxy_tensor(box1, box2)
    else:
        return _calculate_iou_xyxy_native(box1, box2)


def _calculate_iou_xyxy_native(box1, box2):
    """Python native implementation for [x1, y1, x2, y2] format"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def _calculate_iou_xyxy_tensor(box1, box2):
    """PyTorch tensor implementation for [x1, y1, x2, y2] format"""
    # Ensure tensors. The caller dispatches here as soon as EITHER argument is a
    # tensor, so build the other one on that tensor's device - defaulting to CPU
    # makes torch.max fail on a device mismatch when the first is on CUDA.
    ref = box1 if isinstance(box1, torch.Tensor) else box2
    if not isinstance(box1, torch.Tensor):
        box1 = torch.as_tensor(box1, dtype=torch.float32, device=ref.device)
    if not isinstance(box2, torch.Tensor):
        box2 = torch.as_tensor(box2, dtype=torch.float32, device=ref.device)

    box1, box2 = _promote(box1), _promote(box2)

    x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[2], box1[3]
    x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[2], box2[3]
    
    x1_i = torch.max(x1_1, x1_2)
    y1_i = torch.max(y1_1, y1_2)
    x2_i = torch.min(x2_1, x2_2)
    y2_i = torch.min(y2_1, y2_2)
    
    intersection = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return _safe_iou(intersection, union)


def calculate_iou_cxcywh(box1, box2):
    """
    Calculate IoU between two boxes in [cx, cy, w, h] format.
    Supports both Python native types and PyTorch tensors.
    
    Args:
        box1: Box in [cx, cy, w, h] format (list, tuple, or tensor)
        box2: Box in [cx, cy, w, h] format (list, tuple, or tensor)
    
    Returns:
        IoU value (float or tensor)
    """
    # Widen BEFORE the conversion, not after. cxcywh_to_xyxy computes cx +/- w/2,
    # and fp16 near 0.5 has a spacing of ~4.9e-4, so a box of w=1e-4 collapses to
    # zero width there - the corners are already identical by the time
    # _calculate_iou_xyxy_tensor gets a chance to promote.
    if isinstance(box1, torch.Tensor):
        box1 = _promote(box1)
    if isinstance(box2, torch.Tensor):
        box2 = _promote(box2)

    # Convert to xyxy format and calculate
    box1_xyxy = cxcywh_to_xyxy(box1)
    box2_xyxy = cxcywh_to_xyxy(box2)
    return calculate_iou_xyxy(box1_xyxy, box2_xyxy)


def calculate_iou_batch_xyxy(boxes1, boxes2):
    """
    Calculate IoU matrix between two sets of boxes in [x1, y1, x2, y2] format.
    Returns a matrix where [i, j] is the IoU between boxes1[i] and boxes2[j].
    
    Args:
        boxes1: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
        boxes2: Tensor of shape [M, 4] in [x1, y1, x2, y2] format
    
    Returns:
        Tensor of shape [N, M] containing IoU values
    """
    # Ensure tensors
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.tensor(boxes1, dtype=torch.float32)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.tensor(boxes2, dtype=torch.float32)
    
    boxes1, boxes2 = _promote(boxes1), _promote(boxes2)

    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    # Expand dimensions for broadcasting: [N, 1] and [1, M]
    x1_1, y1_1, x2_1, y2_1 = x1_1.unsqueeze(1), y1_1.unsqueeze(1), x2_1.unsqueeze(1), y2_1.unsqueeze(1)
    x1_2, y1_2, x2_2, y2_2 = x1_2.unsqueeze(0), y1_2.unsqueeze(0), x2_2.unsqueeze(0), y2_2.unsqueeze(0)
    
    # Calculate intersection
    x1_i = torch.max(x1_1, x1_2)
    y1_i = torch.max(y1_1, y1_2)
    x2_i = torch.min(x2_1, x2_2)
    y2_i = torch.min(y2_1, y2_2)
    
    intersection = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)
    
    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return _safe_iou(intersection, union)


def calculate_iou_batch_cxcywh(boxes1, boxes2):
    """
    Calculate IoU matrix between two sets of boxes in [cx, cy, w, h] format.
    Returns a matrix where [i, j] is the IoU between boxes1[i] and boxes2[j].
    
    Args:
        boxes1: Tensor of shape [N, 4] in [cx, cy, w, h] format
        boxes2: Tensor of shape [M, 4] in [cx, cy, w, h] format
    
    Returns:
        Tensor of shape [N, M] containing IoU values
    """
    # Ensure tensors
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.tensor(boxes1, dtype=torch.float32)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.tensor(boxes2, dtype=torch.float32)
    
    boxes1, boxes2 = _promote(boxes1), _promote(boxes2)

    cx1, cy1, w1, h1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    cx2, cy2, w2, h2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    # Convert to xyxy format
    x1_1 = cx1 - w1 / 2
    y1_1 = cy1 - h1 / 2
    x2_1 = cx1 + w1 / 2
    y2_1 = cy1 + h1 / 2
    
    x1_2 = cx2 - w2 / 2
    y1_2 = cy2 - h2 / 2
    x2_2 = cx2 + w2 / 2
    y2_2 = cy2 + h2 / 2
    
    # Expand dimensions for broadcasting: [N, 1] and [1, M]
    x1_1, y1_1, x2_1, y2_1, w1, h1 = [t.unsqueeze(1) for t in [x1_1, y1_1, x2_1, y2_1, w1, h1]]
    x1_2, y1_2, x2_2, y2_2, w2, h2 = [t.unsqueeze(0) for t in [x1_2, y1_2, x2_2, y2_2, w2, h2]]
    
    # Calculate intersection
    x1_i = torch.max(x1_1, x1_2)
    y1_i = torch.max(y1_1, y1_2)
    x2_i = torch.min(x2_1, x2_2)
    y2_i = torch.min(y2_1, y2_2)
    
    intersection = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)
    
    # Areas must come from the SAME converted corners the intersection uses.
    # Deriving them from the raw w*h instead lets fp16 rounding make the
    # intersection exceed the area, so identical boxes score above 1.
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return _safe_iou(intersection, union)


def calculate_iou_pairwise_cxcywh(boxes1, boxes2):
    """
    IoU between broadcastable box tensors in [cx, cy, w, h] format.

    calculate_iou_batch_cxcywh builds the full [N, M] cross product from [N, 4]
    and [M, 4]. This one instead broadcasts the leading dimensions normally and
    returns one IoU per aligned pair, for when the pairing is already known.

    Args:
        boxes1: Tensor [..., 4] in [cx, cy, w, h] format
        boxes2: Tensor [..., 4], broadcastable against boxes1

    Returns:
        Tensor of the broadcast shape (without the trailing 4)
    """
    cx1, cy1, w1, h1 = _promote(boxes1).unbind(-1)
    cx2, cy2, w2, h2 = _promote(boxes2).unbind(-1)

    x1_1, y1_1 = cx1 - w1 / 2, cy1 - h1 / 2
    x2_1, y2_1 = cx1 + w1 / 2, cy1 + h1 / 2
    x1_2, y1_2 = cx2 - w2 / 2, cy2 - h2 / 2
    x2_2, y2_2 = cx2 + w2 / 2, cy2 + h2 / 2

    intersection = ((torch.min(x2_1, x2_2) - torch.max(x1_1, x1_2)).clamp(min=0) *
                    (torch.min(y2_1, y2_2) - torch.max(y1_1, y1_2)).clamp(min=0))
    # Same reason as the batch version: areas from the converted corners, not
    # from raw w*h, or fp16 lets identical boxes come out above 1.
    union = ((x2_1 - x1_1) * (y2_1 - y1_1) +
             (x2_2 - x1_2) * (y2_2 - y1_2) - intersection)

    return _safe_iou(intersection, union)


# Convenience aliases for backward compatibility
calculate_iou = calculate_iou_xyxy  # Default to xyxy format for single box pairs

