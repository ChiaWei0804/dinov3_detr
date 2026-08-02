"""
Validation Utilities for Object Detection

This module contains functions for validating detection models:
- Validation loop with loss computation
- Recall@IoU0.5 calculation
- Average IoU calculation
- Performance metrics collection
"""

import torch
from torch.amp import autocast
from coco_dataset import decode_class_predictions
from iou_utils import calculate_iou_batch_cxcywh


def validate_epoch(model, val_loader, loss_fn, device, confidence_threshold=0.1, iou_threshold=0.5):
    """
    Validate model for one epoch.
    
    Args:
        model: Detection model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run validation on
        confidence_threshold: Confidence threshold for filtering predictions (default: 0.1)
        iou_threshold: IoU threshold for recall calculation (default: 0.5)
    
    Returns:
        dict: Dictionary containing validation metrics:
            - avg_val_loss: Average validation loss
            - avg_val_l1_loss: Average L1 loss
            - avg_val_ciou_loss: Average CIoU loss
            - avg_val_cls_loss: Average classification loss
            - recall_iou05: Recall@IoU0.5
            - avg_max_iou: Average maximum IoU
    """
    model.eval()
    
    val_loss = 0.0
    val_l1_loss = 0.0
    val_ciou_loss = 0.0
    val_cls_loss = 0.0
    val_enc_loss = 0.0
    total_predictions = 0
    total_iou = 0.0
    correct_detections = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)

            # AMP mixed precision validation
            with autocast('cuda'):
                outputs = model(images)
                loss, l1_loss, ciou_loss, cls_loss, enc_loss = loss_fn(outputs, targets)

            val_loss += loss.item()
            val_l1_loss += l1_loss
            val_ciou_loss += ciou_loss
            val_cls_loss += cls_loss
            val_enc_loss += enc_loss

            # Calculate metrics on predictions
            metrics = calculate_validation_metrics(
                outputs, targets, device, confidence_threshold, iou_threshold
            )
            
            total_predictions += metrics['total_predictions']
            total_iou += metrics['total_iou']
            correct_detections += metrics['correct_detections']
    
    # Calculate averages
    num_batches = len(val_loader)
    avg_val_loss = val_loss / num_batches
    avg_val_l1_loss = val_l1_loss / num_batches
    avg_val_ciou_loss = val_ciou_loss / num_batches
    avg_val_cls_loss = val_cls_loss / num_batches
    avg_val_enc_loss = val_enc_loss / num_batches

    recall_iou05 = correct_detections / total_predictions if total_predictions > 0 else 0.0
    avg_max_iou = total_iou / total_predictions if total_predictions > 0 else 0.0

    return {
        'avg_val_loss': avg_val_loss,
        'avg_val_l1_loss': avg_val_l1_loss,
        'avg_val_ciou_loss': avg_val_ciou_loss,
        'avg_val_cls_loss': avg_val_cls_loss,
        'avg_val_enc_loss': avg_val_enc_loss,
        'recall_iou05': recall_iou05,
        'avg_max_iou': avg_max_iou
    }


def calculate_validation_metrics(outputs, targets, device, confidence_threshold=0.1, iou_threshold=0.5):
    """
    Calculate validation metrics for a batch of predictions.
    
    Args:
        outputs: Model outputs (dict with 'pred_logits' and 'pred_boxes' or tensor)
        targets: Ground truth targets (list of lists of dicts)
        device: Device to run computation on
        confidence_threshold: Confidence threshold for filtering predictions
        iou_threshold: IoU threshold for recall calculation
    
    Returns:
        dict: Dictionary containing:
            - total_predictions: Total number of ground truth objects
            - total_iou: Sum of maximum IoUs
            - correct_detections: Number of detections with IoU >= threshold
    """
    # Handle different output formats
    if isinstance(outputs, dict):
        cls_logits = outputs['pred_logits']
        bbox_pred = outputs['pred_boxes']
    else:
        batch_size = outputs.shape[0]
        cls_logits = outputs[:, :, :92]
        bbox_pred = outputs[:, :, 92:]
    
    batch_size_val = cls_logits.shape[0]
    total_predictions = 0
    total_iou = 0.0
    correct_detections = 0
    
    for i in range(batch_size_val):
        if i >= len(targets):
            break
        
        image_targets = targets[i]
        if len(image_targets) == 0:
            continue
        
        pred_bboxes_tensor = bbox_pred[i]

        # Filter low score predictions. Background is index 0, so the helper
        # slices it off and shifts the labels back to COCO category_ids.
        pred_max_scores, pred_labels = decode_class_predictions(cls_logits[i])

        keep_mask = pred_max_scores > confidence_threshold
        if keep_mask.sum() == 0:
            total_predictions += len(image_targets)
            continue
        
        keep_pred_bboxes = pred_bboxes_tensor[keep_mask]
        keep_pred_labels = pred_labels[keep_mask]
        
        # Collect GT (keep on GPU)
        gt_bboxes = torch.stack([t['bbox'].to(device) for t in image_targets])
        gt_labels = torch.tensor([t['category_id'].item() for t in image_targets], 
                                device=device, dtype=torch.long)
        
        total_predictions += len(image_targets)
        
        # Calculate IoU for each GT object
        for gt_idx, (gt_bbox, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
            # Find predictions with matching class
            matching_mask = (keep_pred_labels == gt_label)
            if matching_mask.sum() == 0:
                continue
            
            matching_bboxes = keep_pred_bboxes[matching_mask]
            
            # Calculate IoU for each matching prediction against the GT bbox
            # Use unsqueeze(0) to make gt_bbox [1, 4] for batch calculation
            # This returns [N, 1] IoU values, one for each matching bbox
            ious = calculate_iou_batch_cxcywh(matching_bboxes, gt_bbox.unsqueeze(0))
            best_iou = ious.max().item()
            
            total_iou += best_iou
            if best_iou >= iou_threshold:
                correct_detections += 1
    
    return {
        'total_predictions': total_predictions,
        'total_iou': total_iou,
        'correct_detections': correct_detections
    }

