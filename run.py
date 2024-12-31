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
from collections import deque
import os
id = {}







MODEL_WEIGHTS = {
    'swinv2': 'Micrsorft_swinv2_large_patch4_window12_192_22k.pth',
    'convNext': 'convnext_xlarge_22k_1k_384_ema.pth',
    'CLIP': 'CLIPReID_MSMT17_clipreid_12x12sie_ViT-B-16_60.pth',
    'CLIP_RGB': 'CLIPReID_MSMT17_clipreid_12x12sie_ViT-B-16_60.pth',
    'La_Transformer': 'LaTransformer.pth',
    'CTL': 'CTL.pth'
}

def get_id_color(id):
    color = [random.randint(150, 255) for _ in range(3)]
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
    global id
    parser = argparse.ArgumentParser("BoostTrack for image sequence")
    parser.add_argument("--yolo_model", type=str, default="yolo11x.pt")
    parser.add_argument("--img_path", type=str, default="plane/cam0")
    parser.add_argument("--model_name", type=str , choices=['convNext', 'dinov2', 'swinv2','CLIP','CLIP_RGB','La_Transformer','CTL'],
                        default='convNext',
                        help="""
                        Select model type:
                        - convNext : ConvNext-B
                        - dinov2 : Dinov2-B
                        - swinv2 : Swin-B
                        - CLIP : CLIP + RGB AVERAGE DIVIATION
                        - CLIP_RGB : CLIP + RGB AVERAGE DIVIATION
                        --La_Transformer
                        --CTL
                        """)
    
    parser.add_argument("--reid_model", type=str, default=None)
    
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize')
    parser.add_argument('--save_video', action='store_true', default=True, help='Save video')
    parser.add_argument('--save_frame' ,action='store_true', default=True, help='Save frame')
    args = parser.parse_args()

    if args.reid_model is None:
        args.reid_model = MODEL_WEIGHTS[args.model_name]
        print("Weights : ",args.reid_model)

    # 설정
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = False # 카메라 움직임 보정 

    model = YOLO(args.yolo_model)
    tracker = BoostTrack(BoostTrackConfig(
        reid_model_path=f'external/weights/{args.reid_model}',
        device='cuda',
        max_age=100, 
        min_hits=5, # 3 -> 5
        det_thresh=0.4, # up # re id에서 사용하는 객체 임계도
        iou_threshold=0.4, # up
        lambda_iou=0.7,
        lambda_mhd=0.25,
        lambda_shape=0.35, # up
        use_dlo_boost=True, # (Detection - by - Localization) # 이전프레임의 추적정보를 활용하여 검출객체의 신뢰도 향상
        use_duo_boost=True,# (Detection - by - Union) # 겹치는 검출영역 통합 신뢰도향상
        dlo_boost_coef=0.75, # up
        use_rich_s=True, # boost Track ++
        use_sb=True, # Soft Boost
        use_vt=True, # Varying threshold
        s_sim_corr=True, # Corrected shape similarity
        use_reid=True,
        use_cmc=False,
        local_feature=True,
        feature_avg = True,
        model_name = args.model_name
    ))
    
    
    deque_list = deque(maxlen=3)
    
    img_list = natsorted([f for f in os.listdir(args.img_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    
    # stop_frame_ids = [145, 573 , 600 , 650 , 673]
    # stop_frame_ids = [i for i in range(10, len(img_list), 10)]
    stop_frame_ids = [142,198,562,648,656,680]
    # print(stop_frame_ids)
    video_writer = None
    model_name = os.path.splitext(args.reid_model)[0]
    save_dir = f'{args.model_name}_res'
    print("save_dir : ", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for idx , img_name in enumerate(tqdm(img_list)):
        frame_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(args.img_path, img_name)
        
        np_img = cv2.imread(img_path)
        if np_img is None:
            continue

        results = model.predict(np_img, device='cuda', classes=[0] , augment = True , 
                                iou = 0.45 , conf = 0.4 )
        
        for result in results:
            yolo_plot = result.plot()
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
        track_id_list = []

        vis_img = np_img.copy()
        for tlwh, track_id in zip(tlwhs, ids):
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            if track_id not in id:
                color = get_id_color(track_id)
                id[track_id] = color
            else:
                color = id[track_id]
            
            if track_id not in track_id_list:
                track_id_list.append(int(track_id))                    

            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis_img, f"ID: {track_id}", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)



        if args.visualize :                
            cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
            cv2.imshow('yolo', yolo_plot)
            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            cv2.imshow('Tracking', vis_img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            

        if idx in stop_frame_ids and args.save_frame:
            cv2.imwrite(os.path.join(save_dir, f'{idx}.jpg'), vis_img)            
        deque_list.append(vis_img)
        
        print("track id list :" , sorted(track_id_list))
        
        if args.save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_path = os.path.join(save_dir, "tracking.mp4")
                video_writer = cv2.VideoWriter(
                    video_path, 
                    fourcc, 
                    15.0,  # FPS 설정 (0.0 -> 30.0)
                    (vis_img.shape[1], vis_img.shape[0]), 
                    True
                )
            video_writer.write(vis_img)
    
    # 비디오 저장 종료
    if video_writer is not None:
        video_writer.release()

if __name__ == "__main__":
    main()