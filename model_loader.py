"""
Model Loading Utilities

This module contains functions for loading DINOv3 detection models:
- Loading DINOv3 backbone
- Loading detection model from checkpoint
- Supporting both new and old checkpoint formats
"""

import torch
import os
from dinov3_detection_model import DINOv3DetectionModel


def load_dinov3_backbone(weights_path='weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', 
                         repo_dir=None, freeze=True):
    """
    Load DINOv3 backbone model.
    
    Args:
        weights_path: Path to backbone weights file
        repo_dir: Repository directory (default: current working directory)
        freeze: Whether to freeze backbone parameters (default: True)
    
    Returns:
        backbone: Loaded DINOv3 backbone model
    """
    if repo_dir is None:
        # hubconf.py and the default relative weights path both sit beside this
        # module, so anchor to it. Falling back to os.getcwd() made the loader
        # work only when the caller happened to be in the project root.
        repo_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Repo directory: {repo_dir}")
    
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(repo_dir, weights_path)
    
    if not os.path.exists(weights_path):
        print(f"Warning: Backbone weights not found at {weights_path}")
    else:
        print(f"Loading DINOv3 backbone from: {weights_path}")
    
    backbone = torch.hub.load(
        repo_or_dir=repo_dir,
        model="dinov3_vitb16",
        source="local",
        weights=weights_path,
        check_hash=False,
    )
    
    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        print("Backbone parameters frozen")
    
    return backbone


def load_detection_model(model_path, device=None, weights_path='weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                        num_queries=100, verbose=True, allow_partial=False):
    """
    Load detection model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint file
        device: Device to load model on (default: auto-detect)
        weights_path: Path to backbone weights file
        num_queries: Default number of queries if not in checkpoint
        verbose: Whether to print loading information (default: True)
    
    Returns:
        model: Loaded detection model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print("=" * 80)
        print("Load model")
        print("=" * 80)
        print(f"\nLoad checkpoint: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get architecture parameters from checkpoint (with defaults for backward compatibility)
    num_encoder_layers = checkpoint.get('num_encoder_layers', 2)
    num_decoder_layers = checkpoint.get('num_decoder_layers', 4)
    num_queries = checkpoint.get('num_queries', num_queries)
    
    if verbose:
        print(f"Model architecture from checkpoint: Encoder={num_encoder_layers} layers, "
              f"Decoder={num_decoder_layers} layers, Queries={num_queries}")
        
        if 'epoch' in checkpoint:
            print(f"Model trained to Epoch: {checkpoint['epoch'] + 1}")
        
        if 'val_loss' in checkpoint:
            print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # Create model with correct architecture
    if verbose:
        print(f"\nCreate model architecture...")
    
    # num_aux_layers=0: inference never uses the intermediate-layer predictions,
    # so skip those extra head evaluations.
    model = DINOv3DetectionModel(
        weights_path=weights_path,
        num_classes=91,
        num_queries=num_queries,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_aux_layers=0
    )
    
    # Load weights (support both new and old checkpoint formats)
    if verbose:
        print(f"\nLoad model weights...")

    # Components whose weights could not be loaded. Every branch below used to
    # print a warning (only when verbose) and carry on, so an old checkpoint
    # produced a model with a randomly initialised encoder, decoder or bbox head
    # while the function still reported success - and evaluation then measured
    # untrained weights. Collected here and turned into a hard error unless the
    # caller explicitly asks for partial loading.
    skipped = []

    try:
        model.patch_proj.load_state_dict(checkpoint['patch_proj'])
        model.cls_proj.load_state_dict(checkpoint['cls_proj'])
        if verbose:
            print("Loaded projection layers")
        
        # Load Encoder weights (support both new and old format)
        if 'encoder' in checkpoint:
            # New format: unified encoder module
            model.encoder.load_state_dict(checkpoint['encoder'])
            if verbose:
                print("Loaded Transformer Encoder weights (new format)")
        elif 'encoder_self_attns' in checkpoint:
            # Old format: separate components (backward compatibility)
            skipped.append('encoder (old encoder_self_attns format)')
        else:
            skipped.append('encoder (no encoder key in checkpoint)')

        # Load TopK score head
        if 'topk_score_head' in checkpoint:
            model.topk_score_head.load_state_dict(checkpoint['topk_score_head'])
            if verbose:
                print("Loaded TopK score head weights")
        else:
            skipped.append('topk_score_head')
        
        # Load Decoder weights (support both new and old format)
        if 'decoder' in checkpoint:
            # New format: unified decoder module
            model.decoder.load_state_dict(checkpoint['decoder'])
            if verbose:
                print("Loaded Transformer Decoder weights (new format)")
        elif 'decoder_self_attns' in checkpoint:
            # Old format: separate components (backward compatibility)
            skipped.append('decoder (old decoder_self_attns format)')
        else:
            skipped.append('decoder (no decoder key in checkpoint)')
        
        # Load detection heads
        model.cls_head.load_state_dict(checkpoint['cls_head'])
        
        # Handle bbox_head format compatibility (old vs new)
        bbox_head_state = checkpoint['bbox_head']
        # Check if checkpoint uses old format (single Linear layer) or new format (Sequential)
        if 'weight' in bbox_head_state and 'bias' in bbox_head_state:
            # Old format: single Linear layer with keys "weight", "bias"
            # New format: Sequential with keys "0.weight", "0.bias", "2.weight", etc.
            skipped.append('bbox_head (old single-Linear format)')
        else:
            # New format: Sequential layers
            try:
                model.bbox_head.load_state_dict(bbox_head_state)
                if verbose:
                    print("Loaded bbox_head weights (Sequential format)")
            except Exception as e:
                skipped.append(f'bbox_head ({e})')
        
        if verbose:
            print("Loaded detection head weights")
            print("Model weights loaded")
            
    except Exception as e:
        error_msg = f"Load weights failed: {e}"
        if verbose:
            print(error_msg)
            print("\nPossible causes:")
            print("1. Checkpoint format mismatch (old vs new architecture)")
            print("2. Architecture parameters don't match checkpoint")
            print("3. Missing required checkpoint keys")
        raise RuntimeError(error_msg) from e

    if skipped:
        summary = "\n".join(f"  - {s}" for s in skipped)
        if not allow_partial:
            raise RuntimeError(
                "Checkpoint did not supply weights for these components, so they "
                "would stay randomly initialised:\n" + summary +
                "\n\nConvert the checkpoint, or pass allow_partial=True if you "
                "genuinely want an untrained module here.")
        print("WARNING: continuing with randomly initialised components:\n" + summary)

    model.to(device)
    model.eval()
    
    if verbose:
        print("\n" + "=" * 80)
        print("Model ready")
        print("=" * 80)
    
    return model

