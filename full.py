#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR + Symbol pipeline
─────────────────────
Multi-Resolution EAST  →  Tiled YOLO (symbols)  →  Combine
              ↓ mask-out-EAST
          Tiled PaddleOCR
              ↓ mask-out-Paddle
          Tiled EasyOCR
              ↓
      Merge + Cleanup  (text only)
Final image:  red=text   blue=symbol+label
"""

import os, sys, cv2, argparse, numpy as np
from typing import List, Tuple
from PIL import Image
import torch
from paddleocr import PaddleOCR
from ultralytics import YOLO
import easyocr
from imutils.object_detection import non_max_suppression

# ───────────────── 0. Global model loading ─────────────────
USE_GPU = torch.cuda.is_available()

YOLO_MODEL_PATH = "weights/icon_detect/best.pt"          # ← your custom symbol model
yolo_model = YOLO(YOLO_MODEL_PATH)
if USE_GPU: yolo_model.to("cuda")

paddle_ocr = PaddleOCR(det=True, rec=False, lang='en',
                       show_log=False, use_angle_cls=False,
                       use_gpu=USE_GPU)
easyocr_reader = easyocr.Reader(['en'], gpu=USE_GPU)

# ───────────────── 1. Multi-Resolution EAST ─────────────────
class MultiResolutionEAST:
    def __init__(self, pb: str, resolutions: List[Tuple[int,int]]):
        self.net = cv2.dnn.readNet(pb)
        self.res = resolutions
        self.layers = ["feature_fusion/Conv_7/Sigmoid",
                       "feature_fusion/concat_3"]

    def _pass(self, img, tgt, conf):
        H,W = img.shape[:2]; newW,newH = tgt
        rW,rH = W/newW, H/newH
        blob = cv2.dnn.blobFromImage(cv2.resize(img,tgt),1.0,tgt,
                                     (123.68,116.78,103.94),
                                     swapRB=True,crop=False)
        self.net.setInput(blob)
        scores, geo = self.net.forward(self.layers)
        boxes, score_list = [], []
        for y in range(scores.shape[2]):
            sd = scores[0,0,y]
            x0,x1,x2,x3,ang = geo[0,0,y], geo[0,1,y], geo[0,2,y], geo[0,3,y], geo[0,4,y]
            for x in range(scores.shape[3]):
                if sd[x] < conf: continue
                offX,offY = x*4.0, y*4.0
                cos,sin = np.cos(ang[x]), np.sin(ang[x])
                h,w = x0[x]+x2[x], x1[x]+x3[x]
                eX = int(offX + cos*x1[x] + sin*x2[x])
                eY = int(offY - sin*x1[x] + cos*x2[x])
                sX,sY = int(eX - w), int(eY - h)
                boxes.append((int(sX*rW), int(sY*rH),
                              int(eX*rW), int(eY*rH)))
                score_list.append(float(sd[x]))
        return non_max_suppression(np.array(boxes), probs=score_list).tolist() if boxes else []

    def detect(self, img, conf=0.5):
        all_boxes = [self._pass(img,r,conf) for r in self.res]
        flat = [b for sub in all_boxes for b in sub]
        return (non_max_suppression(np.array(flat), overlapThresh=0.7).tolist()
                if flat else [])

multi_east = MultiResolutionEAST(
    "frozen_east_text_detection.pb",
    resolutions=[(960,960),(1600,1600),(3776,3776),(8192,8192)]
)

# ───────────────── 2. Tile helpers ─────────────────
def generate_tiles(img, tile=1024, overlap=100):
    h,w = img.shape[:2]; step=max(tile-overlap,1)
    for y in range(0,h,step):
        for x in range(0,w,step):
            yield img[y:y+tile, x:x+tile], x, y

# ───────────────── 3. Tile-based YOLO symbols ─────────────────
def detect_symbols(img, tile, ov, conf, iou_thr):
    out=[]
    for tile_img, ox, oy in generate_tiles(img,tile,ov):
        res = yolo_model.predict(tile_img, conf=conf, iou=iou_thr,
                                 device="cuda" if USE_GPU else "cpu",
                                 half=USE_GPU, verbose=False)
        if res and len(res[0].boxes):
            for b in res[0].boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                cls = int(b.cls[0])
                out.append((x1+ox, y1+oy, x2+ox, y2+oy, cls))
    return out

# ───────────────── 4. Tile-based Paddle / Easy ─────────────────
def detect_paddle(img, tile, ov, box_thr, det_thr, unclip):
    paddle_ocr.det_db_box_thresh = box_thr
    paddle_ocr.det_db_thresh     = det_thr
    paddle_ocr.det_db_unclip_ratio = unclip
    out=[]
    for tile_img,ox,oy in generate_tiles(img,tile,ov):
        for page in paddle_ocr.ocr(tile_img, cls=False):
            for pts,_ in page:
                xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                out.append((int(min(xs)+ox),int(min(ys)+oy),
                            int(max(xs)+ox),int(max(ys)+oy)))
    return out

def detect_easy(img, tile, ov):
    out=[]
    for tile_img,ox,oy in generate_tiles(img,tile,ov):
        for box,_,_ in easyocr_reader.readtext(
                cv2.cvtColor(tile_img,cv2.COLOR_BGR2RGB), detail=1) or []:
            xs=[p[0] for p in box]; ys=[p[1] for p in box]
            out.append((int(min(xs)+ox),int(min(ys)+oy),
                        int(max(xs)+ox),int(max(ys)+oy)))
    return out

# ───────────────── 5. merge / IoU / cleanup ─────────────────
def compute_iou(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    if xB<xA or yB<yA: return 0
    inter=(xB-xA)*(yB-yA)
    return inter/(((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)+1e-9)

def merge_boxes(boxes, thr=0.3):
    if not boxes: return []
    boxes=sorted(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True)
    keep=[True]*len(boxes)
    for i,b in enumerate(boxes):
        if not keep[i]: continue
        for j in range(i+1,len(boxes)):
            if keep[j] and compute_iou(b,boxes[j])>thr:
                keep[j]=False
    return [b for k,b in zip(keep,boxes) if k]

def cleanup_boxes(boxes, coverage_thr=0.8):
    if not boxes: return []
    area=lambda b:(b[2]-b[0])*(b[3]-b[1])
    boxes=sorted(boxes,key=area,reverse=True)
    keep=[True]*len(boxes)
    for i,big in enumerate(boxes):
        if not keep[i]: continue
        big_a=area(big); cov=0; cnt=0
        for j,sm in enumerate(boxes[i+1:],i+1):
            if keep[j] and sm[0]>=big[0] and sm[1]>=big[1] and sm[2]<=big[2] and sm[3]<=big[3]:
                cov+=area(sm); cnt+=1
        if cnt>1 and cov/big_a>=coverage_thr: keep[i]=False
    return [b for k,b in zip(keep,boxes) if k]

# ───────────────── 6. drawing helpers ─────────────────
def draw_boxes(img, boxes, col, thick=2):
    out=img.copy()
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(out,(x1,y1),(x2,y2),col,thick)
    return out

def draw_symbol_boxes(img, boxes, names):
    out=img.copy()
    for x1,y1,x2,y2,cls in boxes:
        cv2.rectangle(out,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(out, names.get(cls,str(cls)), (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
    return out

# ───────────────── 7. Full per-image pipeline ─────────────────
def process_image(pil_img: Image.Image, *,
                  tile_size=1024, overlap=100,
                  east_conf=0.5, yolo_conf=0.3,
                  iou_merge_thr=0.3, coverage_thr=0.8,
                  paddle_box=0.3, paddle_det=0.3, paddle_unclip=2.0):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 1) EAST (text)
    east_boxes = multi_east.detect(cv_img, conf=east_conf)

    # 2) YOLO (symbols)
    sym_raw  = detect_symbols(cv_img, tile_size, overlap, yolo_conf, 0.45)
    sym_boxes = merge_boxes(sym_raw, thr=iou_merge_thr)

    # 3) Combine EAST+SYMBOL only for subsequent steps (original flow)
    combined = merge_boxes(east_boxes + [b[:4] for b in sym_boxes],
                           thr=iou_merge_thr)

    # 4) PaddleOCR on EAST-masked image (text refinement)
    masked_east = cv_img.copy()
    for x1,y1,x2,y2 in east_boxes:
        cv2.rectangle(masked_east,(x1,y1),(x2,y2),(0,0,0),-1)
    paddle_boxes = detect_paddle(masked_east, tile_size, overlap,
                                 paddle_box, paddle_det, paddle_unclip)

    # 5) EasyOCR on Paddle-masked
    masked_paddle = masked_east.copy()
    for x1,y1,x2,y2 in paddle_boxes:
        cv2.rectangle(masked_paddle,(x1,y1),(x2,y2),(0,0,0),-1)
    easy_boxes = detect_easy(masked_paddle, tile_size, overlap)

    # 6) Merge text detections, cleanup
    all_text = east_boxes + paddle_boxes + easy_boxes
    text_merge = merge_boxes(all_text, thr=iou_merge_thr)
    text_final = cleanup_boxes(text_merge, coverage_thr)

    # 7) Draw
    out = draw_boxes(cv_img, text_final, (0,0,255), 2)   # red text
    out = draw_symbol_boxes(out, sym_boxes, yolo_model.names)  # blue symbols
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

# ───────────────── 8. Batch CLI ─────────────────
def main():
    pa=argparse.ArgumentParser(description="Detect text (red) + symbols (blue) batch")
    pa.add_argument("input_dir"); pa.add_argument("output_dir")
    pa.add_argument("--tile_size", type=int, default=1024)
    pa.add_argument("--overlap",   type=int, default=100)
    pa.add_argument("--east_conf", type=float, default=0.5)
    pa.add_argument("--yolo_conf", type=float, default=0.3)
    pa.add_argument("--iou_merge_thr", type=float, default=0.3)
    pa.add_argument("--coverage_thr",  type=float, default=0.8)
    pa.add_argument("--paddle_box_thr", type=float, default=0.3)
    pa.add_argument("--paddle_det_thr", type=float, default=0.3)
    pa.add_argument("--paddle_unclip",  type=float, default=2.0)
    args=pa.parse_args()

    exts=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    os.makedirs(args.output_dir, exist_ok=True)

    for fn in os.listdir(args.input_dir):
        if not fn.lower().endswith(exts): continue
        try:
            pil=Image.open(os.path.join(args.input_dir,fn)).convert("RGB")
        except Exception as e:
            print(f"Skipping '{fn}': {e}"); continue
        print("Processing", fn)
        out=process_image(pil,
                          tile_size=args.tile_size,
                          overlap=args.overlap,
                          east_conf=args.east_conf,
                          yolo_conf=args.yolo_conf,
                          iou_merge_thr=args.iou_merge_thr,
                          coverage_thr=args.coverage_thr,
                          paddle_box=args.paddle_box_thr,
                          paddle_det=args.paddle_det_thr,
                          paddle_unclip=args.paddle_unclip)
        out.save(os.path.join(args.output_dir, f"{Path(fn).stem}_final.png"))
    print("Batch processing complete.")

if __name__=="__main__":
    main()

