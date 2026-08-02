"""
IoU (Intersection over Union) Calculation Utilities

This module provides comprehensive IoU calculation functions for object detection:
- Support for different box formats: [cx, cy, w, h] and [x1, y1, x2, y2]
- Single box pair and batch matrix calculations
- Both Python native and PyTorch tensor implementations
- Numerical stability optimizations
"""

import torch


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
        # Python native
        if len(boxes.shape) == 1:
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
        if len(boxes) == 4 and not isinstance(boxes[0], (list, tuple)):
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
    # Ensure tensors
    if not isinstance(box1, torch.Tensor):
        box1 = torch.tensor(box1, dtype=torch.float32)
    if not isinstance(box2, torch.Tensor):
        box2 = torch.tensor(box2, dtype=torch.float32)
    
    x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[2], box1[3]
    x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[2], box2[3]
    
    x1_i = torch.max(x1_1, x1_2)
    y1_i = torch.max(y1_1, y1_2)
    x2_i = torch.min(x2_1, x2_2)
    y2_i = torch.min(y2_1, y2_2)
    
    intersection = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = (area1 + area2 - intersection).clamp(min=1e-7)
    
    return intersection / union


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
    union = (area1 + area2 - intersection).clamp(min=1e-7)
    
    return intersection / union


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
    
    # Calculate areas
    area1 = w1 * h1
    area2 = w2 * h2
    union = (area1 + area2 - intersection).clamp(min=1e-7)
    
    return intersection / union


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
    cx1, cy1, w1, h1 = boxes1.unbind(-1)
    cx2, cy2, w2, h2 = boxes2.unbind(-1)

    x1_1, y1_1 = cx1 - w1 / 2, cy1 - h1 / 2
    x2_1, y2_1 = cx1 + w1 / 2, cy1 + h1 / 2
    x1_2, y1_2 = cx2 - w2 / 2, cy2 - h2 / 2
    x2_2, y2_2 = cx2 + w2 / 2, cy2 + h2 / 2

    intersection = ((torch.min(x2_1, x2_2) - torch.max(x1_1, x1_2)).clamp(min=0) *
                    (torch.min(y2_1, y2_2) - torch.max(y1_1, y1_2)).clamp(min=0))
    union = (w1 * h1 + w2 * h2 - intersection).clamp(min=1e-7)

    return intersection / union


# Convenience aliases for backward compatibility
calculate_iou = calculate_iou_xyxy  # Default to xyxy format for single box pairs

