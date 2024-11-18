import os
import cv2
import numpy as np
import torch
import argparse
from default_settings import GeneralSettings
from tracker.boost_track import BoostTrack
import utils
from ultralytics import YOLO
from tqdm import tqdm
from natsort import natsorted
import random


id = {}

def get_id_color(id):
    color = [random.randint(0, 255) for _ in range(3)]
    return color

def process_yolo_detection(results, img_width, img_height):
    """YOLO 검출 결과를 BoostTrack 형식으로 변환"""
    dets = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = boxes.cls[i].cpu().numpy()
            
            # person class(0)만 처리
            if cls == 0:
                dets.append([x1, y1, x2, y2, conf])
                
    return np.array(dets) if dets else None

def main():
    parser = argparse.ArgumentParser("BoostTrack for image sequence")
    parser.add_argument("--yolo_model", type=str, default="yolo11x.pt")
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--img_path", type=str, default="cam2")
    args = parser.parse_args()

    # 설정
    GeneralSettings.values['dataset'] = 'mot17'
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = True

    model = YOLO(args.yolo_model)
    tracker = BoostTrack()

    img_list = natsorted([f for f in os.listdir(args.img_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    for img_name in tqdm(img_list):
        frame_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(args.img_path, img_name)
        
        # BGR 이미지 읽기
        np_img = cv2.imread(img_path)
        if np_img is None:
            continue
            
        # YOLO 검출
        results = model.predict(np_img, device='cuda', classes=[0])
        dets = process_yolo_detection(results, np_img.shape[1], np_img.shape[0])
        
        if dets is None or len(dets) == 0:
            continue
            
        # RGB로 변환 및 CUDA 텐서로 변환
        img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().cuda()
        # HWC -> BCHW 형태로 변환
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [H,W,C] -> [1,C,H,W]
        
        # 추적 수행
        targets = tracker.update(dets, img_tensor, np_img, str(frame_id))
        
        # 결과 필터링
        tlwhs, ids, confs = utils.filter_targets(targets, 
                                               GeneralSettings['aspect_ratio_thresh'],
                                               GeneralSettings['min_box_area'])
        
        # 시각화
        if args.visualize:
            vis_img = np_img.copy()
            for tlwh, track_id in zip(tlwhs, ids):
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                if track_id not in id:
                    color = get_id_color(track_id)
                    id[track_id] = color
                else:
                    color = id[track_id]
                    
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(vis_img, f"ID: {track_id}", (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow('Tracking', vis_img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()