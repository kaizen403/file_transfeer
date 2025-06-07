#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_ocr_symbol_pipeline_verbose.py
───────────────────────────────────
Red = text, Blue = symbols   •   verbose logging optional.

Typical:
    python3 full_ocr_symbol_pipeline_verbose.py IN  OUT

Verbose + tuned symbol merge:
    python3 full_ocr_symbol_pipeline_verbose.py -v \
            --yolo_conf 0.05 --sym_iou 0.05 --sym_min_h 8 \
            IN  OUT
"""
import os, sys, cv2, argparse, numpy as np, logging, time
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR
import easyocr
from imutils.object_detection import non_max_suppression

# ───────────────────── logging config ─────────────────────
LOG = logging.getLogger("OCR-SYM")
logging.basicConfig(format="%(levelname)s: %(message)s",
                    level=logging.INFO)

# ───────────────── early CLI to read weights path ─────────
early = argparse.ArgumentParser(add_help=False)
early.add_argument("--weights", default="weights/icon_detect/best.pt",
                   help="YOLOv8 symbol model path")
early.add_argument("--east_pb", default="frozen_east_text_detection.pb",
                   help="EAST .pb path")
args0, _ = early.parse_known_args()
SYMBOL_WEIGHTS = Path(args0.weights)
EAST_PB        = Path(args0.east_pb)

if not torch.cuda.is_available():
    LOG.error("CUDA GPU not detected."); sys.exit(1)
if not SYMBOL_WEIGHTS.is_file():
    LOG.error("YOLO weights '%s' not found.", SYMBOL_WEIGHTS); sys.exit(1)
if not EAST_PB.is_file():
    LOG.error("EAST model '%s' not found.", EAST_PB); sys.exit(1)

# ───────────────────── model loading ──────────────────────
LOG.info("Loading symbol model  : %s", SYMBOL_WEIGHTS)
sym_model = YOLO(str(SYMBOL_WEIGHTS)).to("cuda")
try: sym_model.model.half(); LOG.info("YOLO FP16 ✔")
except AttributeError: LOG.info("YOLO FP16 not supported (FP32)")

LOG.info("Loading PaddleOCR det : first run may download weights …")
paddle_det = PaddleOCR(det=True, rec=False, lang="en",
                       show_log=False, use_angle_cls=False, use_gpu=True)

LOG.info("Loading EasyOCR       : first run may download weights …")
easy_reader = easyocr.Reader(["en"], gpu=True, verbose=False)

# ───────────────────── Multi-Res EAST ─────────────────────
class MultiResEAST:
    def __init__(self, pb: str, res: List[Tuple[int,int]]):
        self.net  = cv2.dnn.readNet(pb)
        self.res  = res
        self.layers = ["feature_fusion/Conv_7/Sigmoid",
                       "feature_fusion/concat_3"]
        LOG.info("EAST model loaded    : %s", pb)

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
            x0,x1,x2,x3,ang = geo[0,0,y],geo[0,1,y],geo[0,2,y],geo[0,3,y],geo[0,4,y]
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
        return (non_max_suppression(np.array(boxes), probs=score_list).tolist()
                if boxes else [])

    def detect(self, img, conf=0.5):
        flat=[b for r in self.res for b in self._pass(img,r,conf)]
        return (non_max_suppression(np.array(flat), overlapThresh=0.7).tolist()
                if flat else [])

east = MultiResEAST(str(EAST_PB),
                    [(960,960),(1600,1600),(3776,3776),(8192,8192)])

# ───────────────────── helpers ─────────────────────
def tiles(img, size, ov):
    h,w = img.shape[:2]; step = max(size-ov, 1)
    for y in range(0,h,step):
        for x in range(0,w,step):
            yield img[y:y+size, x:x+size], x, y

def iou(a,b):
    xA,yA = max(a[0],b[0]), max(a[1],b[1])
    xB,yB = min(a[2],b[2]), min(a[3],b[3])
    if xB<xA or yB<yA: return 0.
    inter = (xB-xA)*(yB-yA)
    return inter / (((a[2]-a[0])*(a[3]-a[1]) +
                     (b[2]-b[0])*(b[3]-b[1]) - inter)+1e-9)

def draw(img, boxes, col, th=2):
    out = img.copy()
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(out,(x1,y1),(x2,y2),col,th)
    return out

def draw_text(img, boxes):  return draw(img, boxes, (0,0,255), 2)

def draw_symbols(img, boxes, names):
    out = img.copy()
    for x1,y1,x2,y2,cls in boxes:
        cv2.rectangle(out,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(out, names.get(cls,str(cls)), (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),1,cv2.LINE_AA)
    return out

def mask(img, boxes):
    out = img.copy()
    for x1,y1,x2,y2 in boxes:
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,0,0),-1)
    return out

# ───────────────────── symbol detection ───────────────────
def detect_symbols(img, tile, ov, conf, iou_thr):
    out=[]
    for tile_img, ox, oy in tiles(img, tile, ov):
        res = sym_model.predict(tile_img,conf=conf,iou=iou_thr,
                                device="cuda",half=True,verbose=False)
        for b in (res[0].boxes if res else []):
            x1,y1,x2,y2 = map(int,b.xyxy[0].tolist())
            cls = int(b.cls[0])
            out.append((x1+ox,y1+oy,x2+ox,y2+oy,cls))
    return out

def merge_sym(boxes, thr, min_h):
    boxes=[b for b in boxes if (b[3]-b[1])>=min_h]
    if not boxes: return []
    boxes=sorted(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True)
    keep=[True]*len(boxes)
    for i,b in enumerate(boxes):
        if not keep[i]: continue
        for j in range(i+1,len(boxes)):
            if keep[j] and b[4]==boxes[j][4] and iou(b,boxes[j])>thr:
                keep[j]=False
    return [b for k,b in zip(keep,boxes) if k]

# ───────────────────── OCR sub-detectors ──────────
# ───────── detect_paddle (patched) ─────────
def detect_paddle(img, tile, ov):
    out = []
    for tile_img, ox, oy in tiles(img, tile, ov):
        pages = paddle_det.ocr(tile_img, cls=False) or []
        for page in pages:
            if not page:                       # <── NEW
                continue
            for pts, _ in page:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                out.append((int(min(xs)+ox), int(min(ys)+oy),
                            int(max(xs)+ox), int(max(ys)+oy)))
    return out

# ───────── detect_easy (patched) ───────────
def detect_easy(img, tile, ov):
    out = []
    for tile_img, ox, oy in tiles(img, tile, ov):
        res = easy_reader.readtext(
            cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB), detail=1) or []
        for item in res:
            if not item:                       # <── NEW
                continue
            pts, _, _ = item
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            out.append((int(min(xs)+ox), int(min(ys)+oy),
                        int(max(xs)+ox), int(max(ys)+oy)))
    return out

def cleanup(boxes,thr):
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
        if cnt>1 and cov/big_a>=thr: keep[i]=False
    return [b for k,b in zip(keep,boxes) if k]

# ───────────────────── CLI & main ────────────────
def main():
    p = argparse.ArgumentParser(parents=[early],
        description="Full OCR+symbol pipeline (verbose optional)")
    p.add_argument("input_dir"); p.add_argument("output_dir")
    p.add_argument("--tile",type=int,default=1024)
    p.add_argument("--overlap",type=int,default=128)
    p.add_argument("--east_conf",type=float,default=0.5)
    p.add_argument("--yolo_conf",type=float,default=0.25)
    p.add_argument("--merge_iou",type=float,default=0.3)
    p.add_argument("--sym_iou",type=float,default=0.05,
                   help="IoU to merge symbol boxes (default 0.05)")
    p.add_argument("--sym_min_h",type=int,default=6,
                   help="Ignore symbol boxes shorter than this (px)")
    p.add_argument("-v","--verbose",action="store_true")
    args = p.parse_args()
    if args.verbose: LOG.setLevel(logging.DEBUG)

    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    valid=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")

    for fn in sorted(os.listdir(args.input_dir)):
        if not fn.lower().endswith(valid): continue
        img=cv2.cvtColor(np.array(
            Image.open(Path(args.input_dir)/fn).convert("RGB")),
            cv2.COLOR_RGB2BGR)

        t0=time.time()
        e_boxes=east.detect(img,args.east_conf)
        LOG.debug("%s EAST      %d", fn, len(e_boxes))

        s_raw=detect_symbols(img,args.tile,args.overlap,
                             args.yolo_conf,0.45)
        s_boxes=merge_sym(s_raw,args.sym_iou,args.sym_min_h)
        LOG.debug("%s YOLO sym  %d", fn, len(s_boxes))

        masked_e=mask(img,e_boxes)
        p_boxes=detect_paddle(masked_e,args.tile,args.overlap)
        LOG.debug("%s Paddle    %d", fn, len(p_boxes))

        masked_p=mask(masked_e,p_boxes)
        ez_boxes=detect_easy(masked_p,args.tile,args.overlap)
        LOG.debug("%s EasyOCR   %d", fn, len(ez_boxes))

        txt_all=e_boxes+p_boxes+ez_boxes
        txt_merge=(non_max_suppression(np.array(txt_all),overlapThresh=0.3)
                   .tolist() if txt_all else [])
        txt_final=cleanup(txt_merge,thr=0.8)
        LOG.debug("%s FinalTxt  %d", fn, len(txt_final))

        out=draw_text(img,txt_final)
        out=draw_symbols(out,s_boxes,sym_model.names)
        out_path=Path(args.output_dir)/f"{Path(fn).stem}_final.png"
        cv2.imwrite(str(out_path),out)
        LOG.info("%s  text:%d  sym:%d  %.2fs → %s",
                 fn,len(txt_final),len(s_boxes),time.time()-t0,out_path)

    LOG.info("Done → %s", args.output_dir)

if __name__=="__main__":
    main()




    python3 full_ocr_symbol_pipeline_verbose.py  \
        --yolo_conf 0.05 --sym_iou 0.05 --sym_min_h 8 \
        -v  input_dir  output_dir

