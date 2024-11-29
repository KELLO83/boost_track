import os
import cv2
import numpy as np
import torch
import argparse
from default_settings import GeneralSettings, BoostTrackConfig
from tracker.boost_track import BoostTrack
import utils
from ultralytics import YOLO
from tqdm import tqdm
import random
from natsort import natsorted


id = {}

def get_id_color(id):
    color = [random.randint(0, 255) for _ in range(3)]
    return color

def process_yolo_detection(results, img_width, img_height):
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
    parser.add_argument("--img_path", type=str, default="plane/cam2")
    args = parser.parse_args()

    # 설정
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = True

    model = YOLO(args.yolo_model)
    tracker = BoostTrack(BoostTrackConfig(
        reid_model_path='external/weights/vit_transreid_market.pth',
        device='cuda',
        max_age=100,
        min_hits=2,
        det_thresh=0.6,
        iou_threshold=0.3,
        lambda_iou=0.7,
        lambda_mhd=0.25,
        lambda_shape=0.25,
        use_dlo_boost=True,
        use_duo_boost=True,
        dlo_boost_coef=0.65,
        use_rich_s=True, # boost Track ++
        use_sb=True, # Soft Boost
        use_vt=True, # Varying threshold
        s_sim_corr=True, # Corrected shape similarity
        use_reid=True,
        use_cmc=False,
        dataset='Mot17'
    ))

    img_list = natsorted([f for f in os.listdir(args.img_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    for img_name in tqdm(img_list):
        frame_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(args.img_path, img_name)
        
        np_img = cv2.imread(img_path)
        if np_img is None:
            continue

        results = model.predict(np_img, device='cuda', classes=[0] , augment = True)
        dets = process_yolo_detection(results, np_img.shape[1], np_img.shape[0])
        
        if dets is None or len(dets) == 0:
            continue
            

        img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().cuda()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) 
        

        targets = tracker.update(dets, img_tensor, np_img, str(frame_id))
        tlwhs, ids, confs = utils.filter_targets(targets, 
                                               GeneralSettings['aspect_ratio_thresh'],
                                               GeneralSettings['min_box_area'])

        if args.visualize or args.save_video:
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
                           cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        
                
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.imshow('Tracking', vis_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()