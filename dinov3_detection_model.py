import torch
import torch.nn as nn
import os
import math
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder
# Delay import to avoid circular dependency
# load_dinov3_backbone is imported inside __init__ method

def inverse_sigmoid(x, eps=1e-5):
    """
    Inverse of torch.sigmoid, with clamping for numerical safety.

    Patch centres always land in (0, 1) - for a 50x50 grid the extremes are
    0.01 and 0.99, i.e. about +/-4.6 in logit space - so the clamp never
    actually engages here; it is there to keep the function total.
    """
    x = x.clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, b, h, w, device):
        y_embed = torch.arange(1, h + 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(1, w + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(h, 1)
        
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (h + eps) * self.scale
            x_embed = (x_embed - 0.5) / (w + eps) * self.scale
            
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos = torch.cat((pos_y, pos_x), dim=2)
        pos = pos.flatten(0, 1).unsqueeze(0).repeat(b, 1, 1)
        return pos


class DINOv3DetectionModel(nn.Module):
    """
DETR architecture based on DINOv3 Backbone.
    """
    def __init__(self, weights_path='weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                 num_classes=91, num_queries=100,
                 num_encoder_layers=2, num_decoder_layers=4, num_aux_layers=2):
        """
        Args:
            num_classes: Kept as a parameter for readability, but in practice the
                only workable value is coco_dataset.COCO_NUM_CLASSES (91).
                Targets are raw COCO category_ids, model_loader.py and
                eval_coco.py rebuild at 91, and DetectionLoss rejects anything
                else outright. This class does not enforce it - doing so would
                pull a COCO-specific import into an otherwise dataset-agnostic
                module - so the check lives in DetectionLoss, which is built
                before the first batch either way.
            num_aux_layers: How many intermediate decoder layers produce auxiliary
                predictions (the final layer is excluded, it is the main output).
                Set to 0 to skip the extra head evaluations entirely.
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_aux_layers = max(0, min(num_aux_layers, num_decoder_layers - 1))

        # Import here to avoid circular dependency
        from model_loader import load_dinov3_backbone
        
        # repo_dir=None lets the loader anchor to its own directory. Passing
        # os.getcwd() here meant hubconf.py and the relative weights path were
        # resolved against wherever the caller happened to be running from.
        self.backbone = load_dinov3_backbone(
            weights_path=weights_path,
            repo_dir=None,
            freeze=True
        )

        self.patch_proj = nn.Linear(768, 256)
        self.cls_proj = nn.Linear(768, 256)
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
        
        # Transformer Encoder (independent module - Plain DETR key component)
        self.encoder = TransformerEncoder(
            embed_dim=256,
            num_heads=8,
            num_layers=num_encoder_layers,
            dim_feedforward=1024,
            dropout=0.1
        )
        
        # Mixed Query Selection
        self.topk_score_head = nn.Linear(256, 1)
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(
            embed_dim=256,
            num_heads=8,
            num_layers=num_decoder_layers,
            dim_feedforward=1024,
            dropout=0.0
        )
        
        # Classification head: Enhanced multi-layer structure for better classification
        # Similar to bbox_head, use deeper network for feature transformation
        # 256→256(+LayerNorm+ReLU)→num_classes
        self.cls_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),      # Stabilize gradients, critical for fast convergence
            nn.ReLU(inplace=True),  # Or use nn.SiLU() for more modern activation
            nn.Linear(256, num_classes + 1)  # Output layer
        )
        
        # BBox head: Multi-layer structure for better coordinate regression
        # 256→256→128→4 with ReLU activations
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

        # The centre is anchored to the source patch (see _predict_boxes) but w/h
        # are free, so at default init they come out of sigmoid(~0) = 0.5: every
        # query proposes a box covering ~25% of the image. Measured over a random
        # init: w mean 0.513, h mean 0.481, area fraction 0.247.
        #
        # That start is close to a large object (w~0.25, logit -1.1) and far from
        # a small one (w~0.05, logit -2.9), and sigmoid' at 0.05 is 0.19x its
        # value at 0.5 - so small boxes have three times further to travel with a
        # fifth of the gradient. sigmoid(-2.0) = 0.12 sits near the median COCO
        # box instead of near the largest.
        nn.init.constant_(self.bbox_head[-1].bias[2:], -2.0)

    def _predict_boxes(self, features, reference_logits):
        """
        Box head with the centre anchored to the query's source patch.

        bbox_head predicts an OFFSET from the reference point instead of an
        absolute coordinate, so at initialisation every query already proposes a
        box centred on the patch it was selected from.

        Without this, every query starts at sigmoid(~0) = 0.5: one box in the
        middle of the image repeated num_queries times (measured pairwise IoU
        0.97 at init), which leaves Hungarian matching no spatial signal and
        forces the model to learn query-to-region specialisation from scratch.

        Only the centre is anchored - the selected patch tells you where an
        object is, not how big it is, so w/h stay unanchored.

        Args:
            features: [B, K, embed_dim] decoder output
            reference_logits: [B, K, 2] inverse_sigmoid of the patch centres

        Returns:
            [B, K, 4] boxes as [cx, cy, w, h] normalized to [0, 1]
        """
        delta = self.bbox_head(features)                    # [B, K, 4]
        cxcy = delta[..., :2] + reference_logits            # anchored centre
        wh = delta[..., 2:]                                 # free scale
        return torch.cat([cxcy, wh], dim=-1).sigmoid()

    def train(self, mode=True):
        """
        Keep the frozen backbone in eval mode at all times.

        nn.Module.train() recurses into every child, which would switch the
        backbone's stochastic layers (drop-path / dropout) back on even though it
        is frozen and wrapped in torch.no_grad().
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        # Input: x [B, 3, H, W] (e.g., 800×800)
        batch_size = x.shape[0]
        device = x.device
        
        # ===== 1. Backbone feature extraction =====
        # DINOv3 ViT-B/16 processing
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
            cls_token = feats["x_norm_clstoken"]      # [B, 768]
            patch_tokens = feats["x_norm_patchtokens"]  # [B, num_patches, 768] (e.g., 50×50 patches for 800×800 input)
        
        # ===== 2. Feature projection =====
        # 768-dim → 256-dim
        patch_feat = self.patch_proj(patch_tokens)  # [B, num_patches, 768] → [B, num_patches, 256]
        cls_feat = self.cls_proj(cls_token)         # [B, 768] → [B, 256]
        
        # Calculate spatial grid size from actual patch tokens (dynamic based on input resolution)
        # DINOv3 ViT-B/16: patch_size=16, so for 800×800 input → 50×50 grid
        num_patches = patch_tokens.shape[1]
        # Derive the grid from the actual input instead of sqrt(num_patches).
        # A non-square input - 640x800 gives 40x50 = 2000 patches - rounds to
        # 44x44 = 1936 under the square assumption, which both mismatches on the
        # positional-encoding add and corrupts the reference points computed from
        # `w` below. The dataset resizes to a square today, so this is a guard
        # against that changing rather than a live failure.
        patch_size = getattr(self.backbone, 'patch_size', 16)
        h, w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
        if h * w != num_patches:
            raise RuntimeError(
                f"Patch grid {h}x{w} ({h * w}) does not match the {num_patches} "
                f"tokens returned for input {tuple(x.shape[-2:])} "
                f"(patch_size={patch_size})")
        pos_encoding = self.pos_embed(batch_size, h, w, device)  # [B, num_patches, 256]
        
        # ===== 3. Transformer Encoder (independent module) =====
        # Input: patch_feat [B, num_patches, 256] + pos_encoding [B, num_patches, 256]
        memory = self.encoder(patch_feat, pos_encoding)  # [B, num_patches, 256]
        memory_pos = pos_encoding  # [B, num_patches, 256]

        # ===== 4. Mixed Query Selection =====
        # Select top-K patches (e.g., top-100 from 2500 patches)
        # The raw logits are returned as 'enc_objectness' so DetectionLoss can
        # supervise them: torch.topk yields indices only, so without that loss
        # this head would never receive a gradient and the selection would stay
        # stuck at its random initialisation.
        # topk on the logits is equivalent to topk on sigmoid(logits) since
        # sigmoid is monotonic, so no activation is needed here.
        topk = min(self.num_queries, num_patches)
        enc_objectness = self.topk_score_head(memory).squeeze(-1)  # [B, num_patches, 256] → [B, num_patches]
        topk_indices = torch.topk(enc_objectness, topk, dim=1)[1]  # [B, 100] (selected patch indices)

        # Reference points: the normalized centre of each selected patch.
        # ViT tokens are flattened row-major (PatchEmbed does flatten(2) on
        # [B, C, H, W], so index = row * w + col), which matches both
        # PositionEmbeddingSine and the encoder objectness target grid.
        ref_x = ((topk_indices % w).float() + 0.5) / w                                  # [B, K]
        ref_y = (torch.div(topk_indices, w, rounding_mode='floor').float() + 0.5) / h   # [B, K]
        reference_points = torch.stack([ref_x, ref_y], dim=-1)  # [B, K, 2] in [0,1]
        reference_logits = inverse_sigmoid(reference_points)    # [B, K, 2]

        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, 256)  # [B, 100, 256]
        query_content = torch.gather(memory, 1, topk_indices_expanded)      # [B, 100, 256]
        query_pos = torch.gather(memory_pos, 1, topk_indices_expanded)      # [B, 100, 256]
        
        # ===== 5. Transformer Decoder =====
        # Prepare Key-Value memory for Cross-Attention
        cls_pos = torch.zeros(batch_size, 1, 256, device=device)  # [B, 1, 256]
        kv_memory = torch.cat([cls_feat.unsqueeze(1), memory], dim=1)  # [B, 1, 256] + [B, num_patches, 256] → [B, 1+num_patches, 256]
        kv_pos = torch.cat([cls_pos, memory_pos], dim=1)               # [B, 1, 256] + [B, num_patches, 256] → [B, 1+num_patches, 256]
        
        image_features_key = kv_memory + kv_pos    # [B, 1+num_patches, 256]
        image_features_value = kv_memory           # [B, 1+num_patches, 256]
        
        # Decoder processing
        # Input: query_content [B, 100, 256] + query_pos [B, 100, 256]
        # Key-Value: image_features_key/value [B, 1+num_patches, 256]
        query_features, intermediate_outputs = self.decoder(
            query=query_content,                    # [B, 100, 256]
            query_pos=query_pos,                    # [B, 100, 256]
            image_features_key=image_features_key,  # [B, 1025, 256]
            image_features_value=image_features_value  # [B, 1025, 256]
        )  # Output: [B, 100, 256], List of [B, 100, 256]
        
        # ===== 6. Final predictions =====
        # Classification head
        cls_logits = self.cls_head(query_features)  # [B, 100, 256] → [B, 100, 92] (91 classes + background)
        
        # BBox head (multi-layer, centre anchored to the source patch)
        # Layer 1: [B, 100, 256] → [B, 100, 256] → ReLU
        # Layer 2: [B, 100, 256] → [B, 100, 128] → ReLU
        # Layer 3: [B, 100, 128] → [B, 100, 4], then offset by reference_logits
        bbox_pred = self._predict_boxes(query_features, reference_logits)  # [B, 100, 4]
        
        # ===== 7. Auxiliary predictions from intermediate decoder layers =====
        # OPTIMIZED: Only the last `num_aux_layers` intermediate layers feed the
        # auxiliary loss. This reduces overhead while retaining most of the
        # benefit. The final layer is excluded - it is the main output above.
        aux_outputs = []
        if self.num_aux_layers > 0:
            for intermediate_feat in intermediate_outputs[-(self.num_aux_layers + 1):-1]:
                aux_cls = self.cls_head(intermediate_feat)  # [B, 100, 92]
                # Same reference points as the final layer - the anchor belongs
                # to the query, not to the decoder depth.
                aux_bbox = self._predict_boxes(intermediate_feat, reference_logits)
                aux_outputs.append({
                    'pred_logits': aux_cls,
                    'pred_boxes': aux_bbox
                })

        return {
            'pred_logits': cls_logits,        # [B, 100, 92], index 0 == background
            'pred_boxes': bbox_pred,          # [B, 100, 4] in [cx, cy, w, h] format
            'aux_outputs': aux_outputs,       # List of predictions from intermediate layers
            'enc_objectness': enc_objectness, # [B, num_patches] Mixed Query Selection logits
            'enc_grid': (h, w),               # Patch grid size behind enc_objectness
            'reference_points': reference_points,  # [B, 100, 2] patch centres the boxes anchor to
        }
