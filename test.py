import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import math
from pycocotools.coco import COCO
import random
import cv2
import time

from iou_utils import calculate_iou_xyxy
import coco_dataset
from coco_dataset import (COCO_CLASSES, MODEL_OUTPUT_TO_COCO_ID, decode_class_predictions,
                          INPUT_RESOLUTION)
from model_loader import load_detection_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# GPU Memory tracking utilities
def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def get_gpu_memory_reserved_mb():
    """Get reserved GPU memory in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0

def reset_peak_memory():
    """Reset peak memory stats"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak_memory_mb():
    """Get peak GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def print_memory_summary(stage=""):
    """Print GPU memory summary"""
    if torch.cuda.is_available():
        allocated = get_gpu_memory_mb()
        reserved = get_gpu_memory_reserved_mb()
        peak = get_peak_memory_mb()
        print(f"\n{'='*60}")
        print(f"GPU Memory Usage {stage}")
        print(f"{'='*60}")
        print(f"  Allocated: {allocated:.2f} MB")
        print(f"  Reserved: {reserved:.2f} MB")
        print(f"  Peak: {peak:.2f} MB")
        print(f"{'='*60}\n")
        return allocated, reserved, peak
    else:
        print(f"\n[Note] {stage} - CPU mode, no GPU memory stats")
        return 0, 0, 0

# Default weights path for model loading
DEFAULT_WEIGHTS_PATH = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'


# Canvas comes from coco_dataset so training and inference cannot drift apart.
# Training no longer resizes to a square: an image goes to one of three
# aspect-ratio buckets, so inference has to pick the same canvas per image or it
# feeds the model a geometry it was never trained on. Kept as the square canvas
# only for postprocess_detections' signature, which ignores the value.
MODEL_INPUT_SIZE = (INPUT_RESOLUTION, INPUT_RESOLUTION)


def preprocess_image(image_path):
    """Preprocess the input image onto its aspect bucket's canvas"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    canvas = coco_dataset.canvas_for_size(*original_size)
    processed_image = coco_dataset.transform_for_canvas(canvas)(image).unsqueeze(0)
    return processed_image, image, original_size, canvas


def preprocess_frame(frame):
    """Preprocess a cv2 frame (BGR numpy array) for model input"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_h, original_w = frame.shape[:2]
    original_size = (original_w, original_h)

    canvas = coco_dataset.canvas_for_size(original_w, original_h)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        coco_dataset.transform_for_canvas(canvas),
    ])

    processed_image = transform(frame_rgb).unsqueeze(0)
    return processed_image, original_size

def nms(detections, iou_threshold=0.5):
    """
    Class-specific Non-Maximum Suppression
    Only suppresses boxes of the same class (class_id)
    """
    if len(detections) == 0:
        return []
    
    # Group detections by class_id
    detections_by_class = {}
    for det in detections:
        class_id = det['class_id']
        if class_id not in detections_by_class:
            detections_by_class[class_id] = []
        detections_by_class[class_id].append(det)
    
    # Apply NMS separately for each class
    keep = []
    for class_id, class_detections in detections_by_class.items():
        # Sort by confidence (highest first)
        class_detections = sorted(class_detections, key=lambda x: x['confidence'], reverse=True)
        
        while len(class_detections) > 0:
            current = class_detections.pop(0)
            keep.append(current)
            # Remove boxes with high IoU (same class only)
            class_detections = [
                det for det in class_detections
                if calculate_iou_xyxy(current['bbox'], det['bbox']) < iou_threshold
            ]
    
    return keep


def postprocess_detections(predictions, original_size, model_input_size=MODEL_INPUT_SIZE,
                          confidence_threshold=0.5, nms_threshold=0.5):
    """
    Post-process model output to extract detection results.

    Boxes are predicted normalized to [0, 1], so they are scaled straight back to
    the ORIGINAL image size; model_input_size is kept only for signature
    compatibility and does not enter the computation.
    """
    if isinstance(predictions, dict):
        cls_logits = predictions['pred_logits'][0]
        bbox_pred = predictions['pred_boxes'][0]
    else:
        num_classes_plus_bg = predictions.shape[2] - 4
        cls_logits = predictions[0, :, :num_classes_plus_bg]
        bbox_pred = predictions[0, :, num_classes_plus_bg:]
    
    # Background is column 0, so the helper skips it and returns COCO category_ids
    # directly. Taking argmax over [:-1] instead would leave background in the
    # running and silently drop most detections.
    query_scores, query_labels = decode_class_predictions(cls_logits)  # [num_queries]
    num_queries = cls_logits.shape[0]

    detections = []
    orig_width, orig_height = original_size

    for query_idx in range(num_queries):
        confidence = query_scores[query_idx].item()

        if confidence > confidence_threshold:
            # Model output index directly corresponds to COCO category_id (1-90)
            coco_category_id = int(query_labels[query_idx].item())

            # Get continuous_id for class name lookup
            continuous_id = MODEL_OUTPUT_TO_COCO_ID.get(coco_category_id, 0)
            if continuous_id == 0:
                # One of the 10 unused ids in the 1..90 range, skip
                continue

            cx_norm, cy_norm, w_norm, h_norm = bbox_pred[query_idx].cpu().detach().numpy()
            
            cx = cx_norm * orig_width
            cy = cy_norm * orig_height
            w = w_norm * orig_width
            h = h_norm * orig_height
            
            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            x2 = min(orig_width, cx + w / 2)
            y2 = min(orig_height, cy + h / 2)
            
            if x2 <= x1 or y2 <= y1 or w < 1 or h < 1:
                continue
            
            detection = {
                'class_id': coco_category_id,  # COCO category_id (1-90) for matching with GT
                'class_name': COCO_CLASSES[continuous_id] if continuous_id < len(COCO_CLASSES) else f"Class_{coco_category_id}",
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'query_idx': query_idx
            }
            detections.append(detection)

    detections = nms(detections, iou_threshold=nms_threshold)
    return detections


def visualize_detections(image, detections, output_path=None, ground_truth=None, show_plot=True, 
                          linewidth=1.5, fontsize=10):
    """
    Visualize detection results with optional Ground Truth comparison
    
    Args:
        image: PIL Image or numpy array
        detections: List of detection results
        output_path: Path to save the visualization
        ground_truth: Optional Ground Truth annotations
        show_plot: Whether to display the plot (default: True). 
                   Set to False for batch processing to avoid blocking.
        linewidth: Width of bbox lines (default: 1.5)
        fontsize: Font size of labels (default: 10)
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if ground_truth is not None and len(ground_truth) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        ax1.imshow(image)
        ax1.set_title(f'Ground Truth ({len(ground_truth)} objects)', fontsize=16, fontweight='bold')
        
        for gt in ground_truth:
            bbox = gt['bbox']
            category_id = gt.get('category_id', 0)
            continuous_id = MODEL_OUTPUT_TO_COCO_ID.get(category_id, 0)
            class_name = COCO_CLASSES[continuous_id] if continuous_id < len(COCO_CLASSES) else f"Class_{category_id}"
            
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=linewidth,
                edgecolor='green',
                facecolor='none'
            )
            ax1.add_patch(rect)
            
            label = f'{class_name}'
            ax1.text(
                bbox[0],
                bbox[1] - 10,
                label,
                color='white',
                fontsize=fontsize,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.8)
            )
        
        ax1.set_axis_off()
        
        ax2.imshow(image)
        ax2.set_title(f'Model Predictions ({len(detections)} objects)', fontsize=16, fontweight='bold')
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=linewidth,
                edgecolor='red',
                facecolor='none'
            )
            ax2.add_patch(rect)

            label = f'{class_name}: {confidence:.2f}'
            ax2.text(
                bbox[0],
                bbox[1] - 10,
                label,
                color='white',
                fontsize=fontsize,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8)
            )
        
        ax2.set_axis_off()
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=linewidth, label='Ground Truth'),
            Line2D([0], [0], color='red', lw=linewidth, label='Predictions')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=14)
        
    else:
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f'Detection Results ({len(detections)} objects)', fontsize=16, fontweight='bold')

        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=linewidth,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            label = f'{class_name}: {confidence:.2f}'
            ax.text(
                bbox[0],
                bbox[1] - 10,
                label,
                color='white',
                fontsize=fontsize,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8)
            )

        ax.set_axis_off()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200, pad_inches=0)
        print(f"Visualization saved: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()  # Close figure without displaying (for batch processing)


def calculate_recall_for_image(detections, ground_truth, iou_threshold=0.5):
    """
    Calculate Recall for a single image
    Recall = (IoU >= 0.5 and correct class matched GT count) / (total GT count)
    """
    if not ground_truth:
        return 0, 0
    
    hits = 0
    total_gt = len(ground_truth)
    matched_detection_indices = set()
    
    for gt in ground_truth:
        gt_bbox = gt['bbox']
        gt_cat = gt['category_id']
        
        gt_x1, gt_y1 = gt_bbox[0], gt_bbox[1]
        gt_x2, gt_y2 = gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]
        gt_xyxy = [gt_x1, gt_y1, gt_x2, gt_y2]
        
        best_iou = 0.0
        best_det_idx = -1
        
        for i, det in enumerate(detections):
            if i in matched_detection_indices:
                continue
            
            if det['class_id'] != gt_cat:
                continue
                
            iou = calculate_iou_xyxy(gt_xyxy, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_det_idx = i
        
        if best_iou >= iou_threshold and best_det_idx != -1:
            hits += 1
            matched_detection_indices.add(best_det_idx)
            
    return hits, total_gt


def detect_objects(image_path, model, confidence_threshold=0.3, nms_threshold=0.5, 
                   output_path=None, show_ground_truth=True, coco_api=None, image_id=None, 
                   verbose=True, show_plot=True, linewidth=1.5, fontsize=10):
    """
    Complete object detection pipeline
    Returns hits and total_gt for batch statistics
    
    Args:
        show_plot: Whether to display the plot (default: True). 
                   Set to False for batch processing.
        linewidth: Width of bbox lines (default: 1.5)
        fontsize: Font size of labels (default: 10)
    """
    if verbose:
        print(f"\nLoading image: {image_path}")
    
    processed_image, original_image, original_size, model_input_size = preprocess_image(image_path)
    processed_image = processed_image.to(device)
    
    # Load Ground Truth
    ground_truth = []
    if coco_api is not None and image_id is not None:
        try:
            ann_ids = coco_api.getAnnIds(imgIds=image_id)
            anns = coco_api.loadAnns(ann_ids)
            for ann in anns:
                ground_truth.append({
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id']
                })
        except Exception:
            pass

    # Inference
    with torch.no_grad():
        predictions = model(processed_image)
    
    # Post-process
    detections = postprocess_detections(
        predictions, original_size, model_input_size, 
        confidence_threshold, nms_threshold
    )
    
    # Calculate Recall
    hits, total_gt = calculate_recall_for_image(detections, ground_truth, iou_threshold=0.5)
    image_recall = (hits / total_gt * 100) if total_gt > 0 else 0.0
    
    if verbose:
        print(f"Detections: {len(detections)} predictions")
        print(f"Recall stats: hits {hits} / total {total_gt} (Recall: {image_recall:.1f}%)")
        
        visualize_detections(
            original_image, detections, output_path, 
            ground_truth=ground_truth if show_ground_truth else None,
            show_plot=show_plot,
            linewidth=linewidth,
            fontsize=fontsize
        )
    
    return hits, total_gt


def draw_detections_on_frame(frame, detections, show_fps=False, fps=0, inference_time=0):
    """
    Draw detection boxes on a cv2 frame (in-place)
    Returns the modified frame
    """
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw label background
        label = f'{class_name}: {confidence:.2f}'
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 0, 255), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if show_fps:
        # Display FPS
        fps_text = f'FPS: {fps:.1f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display Inference Time
        inference_text = f'Inference: {inference_time:.1f}ms'
        cv2.putText(frame, inference_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame


def detect_video(video_path, model, confidence_threshold=0.3, nms_threshold=0.5,
                 output_path=None, save_video=False, display=True):
    """
    Detect objects in a video file
    
    Args:
        video_path: Path to video file
        model: Detection model
        confidence_threshold: Confidence threshold
        nms_threshold: NMS threshold
        output_path: Output video path (if save_video=True)
        save_video: Whether to save output video
        display: Whether to display video in real-time
    """
    print(f"\nProcessing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}FPS, Total frames: {total_frames}")
    
    # Track GPU memory before inference
    if torch.cuda.is_available():
        reset_peak_memory()
        initial_memory = get_gpu_memory_mb()
        print(f"\nInitial GPU memory: {initial_memory:.2f} MB")
    
    # Setup video writer if saving
    writer = None
    if save_video:
        if output_path is None:
            output_path = 'output_' + os.path.basename(video_path)
        
        # Auto-correct output format to video extension if needed
        if not output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Replace image extension with .mp4
            base_name = os.path.splitext(output_path)[0]
            output_path = os.path.join("test", base_name + '.mp4')
            print(f"WARNING: Output format auto-corrected to video format: {output_path}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving results to: {output_path}")
    
    frame_count = 0
    start_time = time.time()
    
    print("\nStarting detection... (Press 'q' to quit)\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Preprocess and detect
        processed_frame, original_size = preprocess_frame(frame)
        processed_frame = processed_frame.to(device)
        
        # Measure inference time
        inference_start = time.time()
        with torch.no_grad():
            predictions = model(processed_frame)
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        
        detections = postprocess_detections(
            predictions, original_size, MODEL_INPUT_SIZE,
            confidence_threshold, nms_threshold
        )
        
        # Calculate FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Draw detections
        frame = draw_detections_on_frame(frame, detections, show_fps=True, fps=current_fps, inference_time=inference_time)
        
        # Display
        if display:
            cv2.imshow('Video Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser interrupted")
                break
        
        # Save frame
        if writer is not None:
            writer.write(frame)
        
        # Progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processing: {frame_count}/{total_frames} ({progress:.1f}%) - FPS: {current_fps:.1f} - Detections: {len(detections)}")
    
    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nVideo processing completed!")
    print(f"Total frames: {frame_count}")
    print(f"Processing time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    if save_video:
        print(f"Output saved to: {output_path}")
    
    # Print final GPU memory summary
    if torch.cuda.is_available():
        final_memory = get_gpu_memory_mb()
        peak_memory = get_peak_memory_mb()
        print(f"\n{'='*60}")
        print(f"GPU Memory Summary")
        print(f"{'='*60}")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Current memory: {final_memory:.2f} MB")
        print(f"  Peak memory: {peak_memory:.2f} MB")
        print(f"  Model usage: {peak_memory - initial_memory:.2f} MB")
        print(f"  Avg per frame: {(peak_memory - initial_memory) / max(frame_count, 1):.2f} MB")
        print(f"{'='*60}\n")


def detect_webcam(camera_id=0, model=None, confidence_threshold=0.3, nms_threshold=0.5):
    """
    Real-time object detection from webcam
    
    Args:
        camera_id: Camera device ID (default: 0)
        model: Detection model
        confidence_threshold: Confidence threshold
        nms_threshold: NMS threshold
    """
    print(f"\nStarting webcam {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam {camera_id}")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Webcam started! Press 'q' to exit\n")
    
    # Track GPU memory
    if torch.cuda.is_available():
        reset_peak_memory()
        initial_memory = get_gpu_memory_mb()
        print(f"Initial GPU memory: {initial_memory:.2f} MB\n")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read webcam frame")
            break
        
        frame_count += 1
        
        # Preprocess and detect
        processed_frame, original_size = preprocess_frame(frame)
        processed_frame = processed_frame.to(device)
        
        # Measure inference time
        inference_start = time.time()
        with torch.no_grad():
            predictions = model(processed_frame)
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        
        detections = postprocess_detections(
            predictions, original_size, MODEL_INPUT_SIZE,
            confidence_threshold, nms_threshold
        )
        
        # Calculate FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Draw detections
        frame = draw_detections_on_frame(frame, detections, show_fps=True, fps=current_fps, inference_time=inference_time)
        
        # Display
        cv2.imshow('Webcam Detection', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nWebcam detection ended")
    print(f"Total frames: {frame_count}")
    print(f"Runtime: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    
    # Print GPU memory summary
    if torch.cuda.is_available():
        final_memory = get_gpu_memory_mb()
        peak_memory = get_peak_memory_mb()
        print(f"\n{'='*60}")
        print(f"GPU Memory Summary")
        print(f"{'='*60}")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Current memory: {final_memory:.2f} MB")
        print(f"  Peak memory: {peak_memory:.2f} MB")
        print(f"  Model usage: {peak_memory - initial_memory:.2f} MB")
        print(f"{'='*60}\n")


def test_on_coco_images(model_path, num_images=10, confidence_threshold=0.3, show_ground_truth=True,
                        linewidth=1.5, fontsize=10):
    """
    Test model on COCO validation set with Ground Truth comparison
    
    Args:
        linewidth: Width of bbox lines (default: 1.5)
        fontsize: Font size of labels (default: 10)
    """
    print("\n" + "=" * 80)
    print(f"在 COCO 驗證集上測試（前 {num_images} 張圖）")
    print(f"Ground Truth 對比: {'開啟' if show_ground_truth else '關閉'}")
    print("=" * 80)
    
    coco = COCO('./data/coco/annotations/instances_val2017.json')
    all_img_ids = coco.getImgIds()
    selected_ids = random.sample(all_img_ids, min(num_images, len(all_img_ids)))
    
    model = load_detection_model(
        model_path=model_path,
        device=device,
        weights_path=DEFAULT_WEIGHTS_PATH,
        num_queries=100,
        verbose=True
    )
    
    total_hits = 0
    total_objects = 0
    
    print(f"\n開始測試 {len(selected_ids)} 張圖片...\n")
    
    for i, img_id in enumerate(selected_ids):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        full_path = os.path.join('./data/coco/val2017', file_name)
        
        print(f"{'='*40}")
        print(f"圖片 [{i+1}/{len(selected_ids)}]: {file_name}")
        
        if os.path.exists(full_path):
            hits, num_gt = detect_objects(
                image_path=full_path,
                model=model,
                confidence_threshold=confidence_threshold,
                nms_threshold=0.9,
                output_path=f'test/test_result_{i+1}.jpg',
                show_ground_truth=show_ground_truth,
                coco_api=coco,
                image_id=img_id,
                verbose=True,
                show_plot=False,  # 批量測試時不顯示圖片，只保存
                linewidth=linewidth,
                fontsize=fontsize
            )
            
            total_hits += hits
            total_objects += num_gt
        else:
            print(f"圖片不存在: {full_path}")

    print("\n" + "="*60)
    print("測試總結報告")
    print("="*60)
    print(f"測試圖片數量: {len(selected_ids)}")
    print(f"總 Ground Truth 物體數: {total_objects}")
    print(f"總正確命中數 (IoU>=0.5): {total_hits}")
    
    final_recall = (total_hits / total_objects * 100) if total_objects > 0 else 0.0
    print(f"平均 Recall: {final_recall:.2f}%")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DINOv3 物件檢測測試 - 支持多種輸入源')

    parser.add_argument('--model', type=str, default='runs/best_coco_detection_head.pth',
                       help='模型 checkpoint 路徑 (default: runs/best_coco_detection_head.pth，與 train.py 的 save_dir 一致)')
    
    # 統一的 source 參數
    parser.add_argument('--source', type=str, default="people_walking.mp4",
                       help='輸入源: 0=webcam, video.mp4=video file, image.jpg=image, coco=COCO dataset')
    
    # 保留舊參數以向後兼容
    parser.add_argument('--image', type=str, default=None,
                       help='(舊參數) 測試圖片路徑，建議使用 --source')
    parser.add_argument('--test-coco', action='store_true',
                       help='(舊參數) 在 COCO 驗證集上測試，建議使用 --source coco')
    
    # 檢測參數
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='置信度閾值 (default: 0.1)')
    # DETR 類模型靠一對一匹配抑制重複框，NMS 只是保險，因此閾值故意放寬。
    parser.add_argument('--nms', type=float, default=0.9,
                       help='NMS 閾值 (default: 0.9)')
    
    # 輸出參數
    parser.add_argument('--output', type=str, default='test_detection_result.jpg',
                       help='輸出文件路徑 (圖片/視頻)')
    parser.add_argument('--save-video', action='store_true', default=True,
                       help='保存視頻檢測結果')
    parser.add_argument('--no-display', action='store_true', default=True,
                       help='不顯示實時檢測窗口 (僅用於視頻/攝像頭)')
    
    # COCO 測試參數
    parser.add_argument('--num-coco-images', type=int, default=60,
                       help='測試 COCO 圖片數量')
    parser.add_argument('--no-gt', action='store_true',
                       help='不顯示 Ground Truth 對比')
    
    # 可視化參數
    parser.add_argument('--linewidth', type=float, default=1,
                       help='BBox 線條寬度 (default: 1)')
    parser.add_argument('--fontsize', type=float, default=6,
                       help='Label 字體大小 (default: 6)')
    # 與 eval_coco.py 一致。沒有這個旗標時，另開一個程序跑 test.py 只會拿到
    # coco_dataset 的模組預設值，訓練若用了別的解析度就會靜默錯配：訓練
    # input_resolution=640 的 landscape 畫布是 736×544，推論卻送 912×688。
    parser.add_argument('--resolution', type=int,
                       default=coco_dataset.DEFAULT_INPUT_RESOLUTION,
                       help='輸入解析度，必須與訓練時一致 (default: 800)')


    args = parser.parse_args()

    # 必須在任何 preprocess / transform 之前設定
    coco_dataset.set_input_resolution(args.resolution)

    print("=" * 80)
    print("DINOv3 物件檢測測試工具 - 多源支持")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  模型路徑: {args.model}")
    print(f"  輸入解析度: {args.resolution} (square bucket 邊長)")
    print(f"  置信度閾值: {args.threshold}")
    print(f"  NMS 閾值: {args.nms}")
    print(f"  BBox 線寬: {args.linewidth}")
    print(f"  Label 字體大小: {args.fontsize}")
    
    if not os.path.exists(args.model):
        print(f"\n錯誤：找不到模型文件 {args.model}")
        exit(1)
    
    # 確定輸入源（優先使用 --source，向後兼容 --image 和 --test-coco）
    source = args.source
    
    # 向後兼容性處理
    if source is None:
        if args.test_coco:
            source = 'coco'
        elif args.image:
            source = args.image
        else:
            # 默認使用 COCO 測試
            source = 'coco'
            print("\n未指定輸入源，使用默認 COCO 測試模式")
            print("提示: 使用 --source 指定輸入源")
            print("  --source 0        (攝像頭)")
            print("  --source video.mp4 (視頻文件)")
            print("  --source image.jpg (單張圖片)")
            print("  --source coco      (COCO 數據集)\n")
    
    print(f"  輸入源: {source}\n")
    
    # 根據 source 類型執行不同的檢測
    if source == 'coco':
        # COCO 批量測試
        test_on_coco_images(
            model_path=args.model,
            num_images=args.num_coco_images,
            confidence_threshold=args.threshold,
            show_ground_truth=not args.no_gt,
            linewidth=args.linewidth,
            fontsize=args.fontsize
        )
    
    elif str(source).isdigit():
        # 攝像頭檢測
        camera_id = int(source)
        print(f"模式: 攝像頭檢測 (Camera ID: {camera_id})\n")
        
        model = load_detection_model(
            model_path=args.model,
            device=device,
            weights_path=DEFAULT_WEIGHTS_PATH,
            num_queries=100,
            verbose=True
        )
        
        detect_webcam(
            camera_id=camera_id,
            model=model,
            confidence_threshold=args.threshold,
            nms_threshold=args.nms
        )
    
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
        # 視頻文件檢測
        if not os.path.exists(source):
            print(f"\n錯誤：找不到視頻文件 {source}")
            exit(1)
        
        print(f"模式: 視頻檢測\n")
        
        model = load_detection_model(
            model_path=args.model,
            device=device,
            weights_path=DEFAULT_WEIGHTS_PATH,
            num_queries=100,
            verbose=True
        )
        
        detect_video(
            video_path=source,
            model=model,
            confidence_threshold=args.threshold,
            nms_threshold=args.nms,
            output_path=args.output if args.save_video else None,
            save_video=args.save_video,
            display=not args.no_display
        )
    
    elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        # 單張圖片檢測
        if not os.path.exists(source):
            print(f"\n錯誤：找不到圖片文件 {source}")
            exit(1)
        
        print(f"模式: 單張圖片檢測\n")
        
        model = load_detection_model(
            model_path=args.model,
            device=device,
            weights_path=DEFAULT_WEIGHTS_PATH,
            num_queries=100,
            verbose=True
        )
        
        detect_objects(
            image_path=source,
            model=model,
            confidence_threshold=args.threshold,
            nms_threshold=args.nms,
            output_path=args.output,
            show_ground_truth=False,
            linewidth=args.linewidth,
            fontsize=args.fontsize
        )
    
    else:
        print(f"\n錯誤：不支持的輸入源格式: {source}")
        print("\n支持的格式:")
        print("  - 攝像頭: 0, 1, 2, ...")
        print("  - 視頻: .mp4, .avi, .mov, .mkv, .flv, .wmv")
        print("  - 圖片: .jpg, .jpeg, .png, .bmp, .webp")
        print("  - COCO: coco")
        exit(1)

