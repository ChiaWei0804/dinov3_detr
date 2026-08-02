import os
import sys

# 若遇到 Conda DLL 載入問題，設定環境變數 DINOV3_EXTRA_DLL_DIR 指向
# <env>\Library\bin 即可，不要把個別機器的絕對路徑寫死在程式裡。
#
# 這段必須排在 import torch 之前：它存在的目的就是讓 torch 找得到 DLL，
# 而 torch 一旦先 import 完，PATH 再怎麼改都來不及了。
_extra_dll_dir = os.environ.get('DINOV3_EXTRA_DLL_DIR')
if _extra_dll_dir and os.path.isdir(_extra_dll_dir):
    os.environ['PATH'] = _extra_dll_dir + os.pathsep + os.environ['PATH']
    if hasattr(os, 'add_dll_directory'):  # Windows: PATH alone is not enough
        # The handle must stay referenced - dropping it removes the directory
        # from the search path again, which would undo this on the next GC.
        _dll_handle = os.add_dll_directory(_extra_dll_dir)
    print(f"Fixed DLL path: Added {_extra_dll_dir} to PATH")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 引入您專案中的模組
from dinov3_detection_model import DINOv3DetectionModel
import coco_dataset
from coco_dataset import decode_class_predictions, VALID_COCO_CATEGORY_IDS

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class COCOEvalDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        # [修正 1] 傳遞 transform=None 給父類別
        # 這樣 super().__getitem__ 才會回傳原始的 PIL 圖片，而不是處理後的 Tensor
        super().__init__(root, annFile, transform=None)
        
        # [修正 2] 將 transform 存在自己的屬性中，稍後手動呼叫
        #
        # transform=None 時 __getitem__ 會原樣回傳 PIL Image，而 collate_fn_eval
        # 直接對它做 torch.stack，會在建立 DataLoader 之後才以 TypeError 爆掉。
        # 與其讓它拖到那時候，不如在這裡就講清楚。
        if transform is None:
            raise ValueError(
                "COCOEvalDataset 需要一個 transform（例如 coco_dataset 的 "
                "global_val_transform）。transform=None 會讓 __getitem__ 回傳 "
                "PIL Image，collate 階段的 torch.stack 必然失敗。")
        self.eval_transform = transform

    def __getitem__(self, index):
        # 1. 取得原始 PIL 圖片 (因為父類別 transform 為 None)
        img, target = super().__getitem__(index)
        
        # 2. 正確獲取原始尺寸 (PIL Image 的 .size 是屬性，回傳 (w, h))
        orig_w, orig_h = img.size
        
        image_id = self.ids[index]

        # 3. 手動執行預處理 (Resize 512 + Normalize + ToTensor)
        if self.eval_transform is not None:
            img = self.eval_transform(img)

        return img, {
            "image_id": image_id,
            "orig_size": torch.tensor([orig_w, orig_h])
        }


def collate_fn_eval(batch):
    """簡單的 Collate Function，用於堆疊圖片"""
    images = torch.stack([b[0] for b in batch], dim=0)
    meta_info = [b[1] for b in batch]
    return images, meta_info

def evaluate(model, data_loader, coco_gt, device, output_file='prediction_results.json'):
    model.eval()
    results = []
    
    print(f"\n開始評估 {len(data_loader.dataset)} 張圖片...")
    
    with torch.no_grad():
        for images, meta_info in tqdm(data_loader, desc="Inference"):
            images = images.to(device)
            
            # 模型推理
            # output 包含: 'pred_logits', 'pred_boxes' (歸一化的 [cx, cy, w, h])
            outputs = model(images)
            
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

            # 背景是第 0 類（訓練時 target 用 0 代表 no-object），不是最後一類。
            # 用 [..., :-1] 會把「從未被訓練到的最後一欄」切掉、卻把背景留在
            # argmax 裡，導致絕大多數偵測被當成背景丟棄。
            scores, labels = decode_class_predictions(pred_logits)

            # 處理 Batch 中的每一張圖
            for i in range(len(images)):
                image_id = meta_info[i]['image_id']
                orig_w, orig_h = meta_info[i]['orig_size']
                
                # 取得該圖的預測
                img_scores = scores[i]
                img_labels = labels[i]
                img_boxes = pred_boxes[i] # [cx, cy, w, h] normalized (0-1)
                
                # 篩選低置信度預測以加速評估 (通常 0.05 或 0.01)
                keep = img_scores > 0.05
                
                if keep.sum() == 0:
                    continue
                
                keep_scores = img_scores[keep]
                keep_labels = img_labels[keep]
                keep_boxes = img_boxes[keep]
                
                # === 座標轉換 ===
                # 1. 將歸一化的 [cx, cy, w, h] 還原回 "原始圖片尺寸"
                # 注意：不是還原回 512，而是還原回原圖大小，這樣才能跟 GT 比對
                cx = keep_boxes[:, 0] * orig_w.to(device)
                cy = keep_boxes[:, 1] * orig_h.to(device)
                w  = keep_boxes[:, 2] * orig_w.to(device)
                h  = keep_boxes[:, 3] * orig_h.to(device)
                
                # 2. 轉換為 COCO 要求的 [x_top_left, y_top_left, w, h]，並裁切到圖內
                img_w = orig_w.to(device).float()
                img_h = orig_h.to(device).float()
                x1 = (cx - 0.5 * w).clamp(min=0)
                y1 = (cy - 0.5 * h).clamp(min=0)
                x2 = (cx + 0.5 * w).clamp(max=img_w)
                y2 = (cy + 0.5 * h).clamp(max=img_h)

                # 轉換回 CPU 列表
                x = x1.cpu().tolist()
                y = y1.cpu().tolist()
                w = (x2 - x1).cpu().tolist()
                h = (y2 - y1).cpu().tolist()
                s = keep_scores.cpu().tolist()
                l = keep_labels.cpu().tolist()

                for box_id in range(len(x)):
                    # 訓練時 GT 直接使用原始 COCO category_id (1-90)，所以模型的
                    # 類別索引即為 category_id，不需要再做映射。
                    category_id = int(l[box_id])

                    # 1-90 之間有 10 個編號在 COCO 並不存在，直接丟棄
                    if category_id not in VALID_COCO_CATEGORY_IDS:
                        continue

                    # 裁切後退化的框對 COCOeval 沒有意義
                    if w[box_id] <= 0 or h[box_id] <= 0:
                        continue

                    res = {
                        "image_id": image_id,
                        "category_id": category_id, 
                        "bbox": [x[box_id], y[box_id], w[box_id], h[box_id]],
                        "score": s[box_id]
                    }
                    results.append(res)
    
    if not results:
        print("未檢測到任何目標，無法進行評估。")
        return

    # 寫入 JSON
    print(f"寫入預測結果至 {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f)
        
    # 執行 COCO Eval
    print("正在執行 COCO Evaluation...")
    coco_dt = coco_gt.loadRes(output_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0] # mAP@0.5:0.95

def main():
    parser = argparse.ArgumentParser(description='COCO mAP Evaluation for DINOv3-Detector')
    # parser.add_argument('--model', type=str, default='runs/coco_detection_head_epoch80.pth', help='Path to .pth checkpoint') #512*512
    parser.add_argument('--model', type=str, default='runs/best_coco_detection_head.pth', help='Path to .pth checkpoint') #800*800
    parser.add_argument('--val-root', type=str, default='./data/coco/val2017', help='Path to val images')
    parser.add_argument('--ann-file', type=str, default='./data/coco/annotations/instances_val2017.json', help='Path to annotations')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--layers', type=int, default=6,
                        help='Decoder layers, 僅在 checkpoint 未記錄時作為後備 (default: 6)')
    parser.add_argument('--resolution', type=int, default=coco_dataset.DEFAULT_INPUT_RESOLUTION,
                        help='輸入解析度，必須與訓練時一致 (default: 800)')
    args = parser.parse_args()

    # 必須在建立 transform / DataLoader 之前設定
    coco_dataset.set_input_resolution(args.resolution)
    print(f"輸入解析度: {args.resolution}×{args.resolution}")

    # === 自動偵測模型配置 ===
    checkpoint = torch.load(args.model, map_location=device)
    num_enc = checkpoint.get('num_encoder_layers', 2)
    num_dec = checkpoint.get('num_decoder_layers', args.layers)
    num_q = checkpoint.get('num_queries', 100)

    print(f"偵測到模型配置: Encoder={num_enc}, Decoder={num_dec}, Queries={num_q}")

    # 根據偵測到的配置初始化模型
    # num_aux_layers=0：推論不需要中間層的輔助預測，省去多餘的 head 計算
    model = DINOv3DetectionModel(
        num_classes=91,
        num_queries=num_q,
        num_encoder_layers=num_enc,
        num_decoder_layers=num_dec,
        num_aux_layers=0
    )

    # 載入權重 (處理 state_dict 的不同儲存方式)
    if 'patch_proj' in checkpoint:
        # 如果是我們自定義的 save_dict 格式 (train.py 儲存的格式)
        # 我們需要手動載入各個部分，因為 save_dict 不是直接的 model state_dict
        print("載入訓練腳本儲存的完整 Checkpoint 格式...")
        try:
            model.patch_proj.load_state_dict(checkpoint['patch_proj'])
            model.cls_proj.load_state_dict(checkpoint['cls_proj'])
            model.encoder.load_state_dict(checkpoint['encoder'])
            model.topk_score_head.load_state_dict(checkpoint['topk_score_head'])
            model.decoder.load_state_dict(checkpoint['decoder'])
            model.cls_head.load_state_dict(checkpoint['cls_head'])
            model.bbox_head.load_state_dict(checkpoint['bbox_head'])
            print("權重載入成功 (分塊載入)。")
        except KeyError as e:
            # 舊的 fallback 是 model.load_state_dict(checkpoint, strict=False)，
            # 但這個巢狀字典的鍵是「元件名稱」而非參數名稱，所以一個權重都載不進去，
            # strict=False 又把它壓下來 —— 評估就在隨機初始化的模型上跑完了。
            raise RuntimeError(
                f"Checkpoint 缺少元件 {e}，無法以分塊格式載入。"
                f"請改用完整的 checkpoint，不要讓評估在未載入權重的模型上進行。"
            ) from e

    else:
        # 可能是標準的 state_dict 格式 (例如 final_xxx.pth)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # DataParallel / DDP 會在每個鍵前面加 "module."，不剝掉的話下面的過濾
        # 會得到空字典，而 strict=False 會讓它看起來像是載入成功。
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k[len('module.'):] if k.startswith('module.') else k: v
                          for k, v in state_dict.items()}
        model_keys = model.state_dict().keys()
        # 過濾掉不匹配的鍵值 (例如凍結的 backbone 部分如果沒存)
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        # 只檢查「非空」不夠：checkpoint 只含 cls_head.* 時 filtered_dict 也非空，
        # strict=False 會讓 decoder、bbox_head 保持隨機初始化然後照樣跑完評估。
        # 逐一比對每個頂層元件，缺哪個就講哪個。
        # 逐一比對「參數」而非只看元件前綴。只確認每個元件至少有一個 key 是不夠的：
        # decoder 只剩一個參數時它仍會被視為存在，strict=False 讓其餘參數靜靜地
        # 保持隨機初始化。backbone 是凍結的、通常不存在 checkpoint 裡，所以只對
        # 這 7 個偵測頭元件要求完整。
        components = ('patch_proj', 'cls_proj', 'encoder', 'topk_score_head',
                      'decoder', 'cls_head', 'bbox_head')
        expected = {k for k in model.state_dict()
                    if k.split('.')[0] in components}
        missing = sorted(expected - set(filtered_dict))
        if missing:
            preview = ', '.join(missing[:6]) + (' ...' if len(missing) > 6 else '')
            raise RuntimeError(
                f"Checkpoint 缺少 {len(missing)}/{len(expected)} 個偵測頭參數，"
                f"它們會維持隨機初始化：{preview}"
                f"（checkpoint 共 {len(state_dict)} 個鍵，對上 {len(filtered_dict)} 個）。"
                f"用這種模型做評估得到的 mAP 沒有意義。")
        model.load_state_dict(filtered_dict, strict=False)
        print(f"權重載入成功 (標準格式)：{len(filtered_dict)}/{len(model_keys)} 個鍵，"
              f"{len(components)} 個元件齊全")
    
    model.to(device)
    
    # 2. 準備數據集 (使用 coco_dataset.py 中的 global_val_transform)
    print("Preparing dataset...")
    dataset = COCOEvalDataset(
        root=args.val_root,
        annFile=args.ann_file,
        # 關鍵：使用專案定義的預處理。以模組屬性讀取，才能取到
        # set_input_resolution() 之後重建的 transform。
        transform=coco_dataset.global_val_transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=collate_fn_eval,
        pin_memory=True
    )
    
    # 3. 載入 GT 用於評估
    coco_gt = COCO(args.ann_file)
    
    # 4. 執行評估
    evaluate(model, dataloader, coco_gt, device)

if __name__ == '__main__':
    main()