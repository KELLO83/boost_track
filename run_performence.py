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
from natsort import natsorted
from collections import deque
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class VisualizationConfig:
    visualize: bool = False
    save_video: bool = True
    save_frame: bool = False
    stop_frame_ids: List[int] = None

class ModelConfig:
    WEIGHTS = {
        'swinv2': 'Micrsorft_swinv2_large_patch4_window12_192_22k.pth',
        'convNext': 'convnext_xlarge_22k_1k_384_ema.pth',
        'CLIP': 'CLIPReID_MSMT17_clipreid_12x12sie_ViT-B-16_60.pth',
        'CLIP_RGB': 'CLIPReID_MSMT17_clipreid_12x12sie_ViT-B-16_60.pth',
        'La_Transformer': 'LaTransformer.pth',
        'CTL': 'CTL.pth',
        'VIT-B/16+ICS_SSL': 'vit_base_ics_cfs_lup.pth',
        'VIT_SSL_MARKET': 'VIT_SSL_MSMT17_L.pth'
    }

    @staticmethod
    def get_model_name(reid_model: str) -> str:
        model_name = os.path.splitext(reid_model)[0]
        return 'VIT_SSL_BASE' if model_name == 'VIT-B/16+ICS_SSL' else model_name

class Visualizer:
    def __init__(self):
        self.id_colors: Dict[int, List[int]] = {}
        
    def get_id_color(self, track_id: int) -> List[int]:
        if track_id not in self.id_colors:
            self.id_colors[track_id] = [random.randint(150, 255) for _ in range(3)]
        return self.id_colors[track_id]

    def draw_detection_boxes(self, image: np.ndarray, dets: np.ndarray) -> np.ndarray:
        vis_img = image.copy()
        for i, (x1, y1, x2, y2, conf) in enumerate(dets):
            cv2.putText(vis_img, f"#{i}", (int(x2)-30, int(y1)+20),
                      cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        return vis_img

    def draw_tracking_results(self, image: np.ndarray, tlwhs: List, ids: List[int]) -> Tuple[np.ndarray, List[int]]:
        vis_img = image.copy()
        track_id_list = []
        
        for tlwh, track_id in zip(tlwhs, ids):
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            color = self.get_id_color(track_id)
            
            if track_id not in track_id_list:
                track_id_list.append(int(track_id))
                
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis_img, f"ID: {track_id}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                        
        return vis_img, track_id_list

class VideoWriter:
    def __init__(self, save_dir: str, model_name: str, emb_method: str, video_name: str):
        os.makedirs(save_dir, exist_ok=True)
        name = f"{model_name}_{emb_method}_{video_name}"
        self.video_path = os.path.join(save_dir, f"{name}_tracking.mp4")
        self.writer = None

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.video_path,
                fourcc,
                15.0,
                (frame.shape[1], frame.shape[0]),
                True
            )
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"Video saved to: {self.video_path}")

def process_yolo_detection(results) -> Optional[np.ndarray]:
    dets = []
    xywhs = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = boxes.cls[i].cpu().numpy()
            
            if cls == 0:  # person class only
                dets.append([x1, y1, x2, y2, conf])
                # Convert xyxy to xywh (center format)
                w = x2 - x1
                h = y2 - y1
                x_center = x1 + w/2
                y_center = y1 + h/2
                xywhs.append([x_center, y_center, w, h])
                
    return np.array(dets) if dets else None, np.array(xywhs) if xywhs else None

def setup_tracker(args) -> BoostTrack:
    config = BoostTrackConfig(
        reid_model_path=f'external/weights/{args.reid_model}',
        device='cuda',
        max_age=15,
        min_hits=3,
        det_thresh=0.4,
        iou_threshold=0.9,
        emb_sim_score=0.60,
        lambda_iou=0.05,
        lambda_mhd=0.25,
        lambda_shape=0.95,
        use_dlo_boost=True,
        use_duo_boost=True,
        dlo_boost_coef=0.75,
        use_rich_s=True,
        use_sb=True,
        use_vt=True,
        s_sim_corr=True,
        use_reid=True,
        use_cmc=False,
        local_feature=True,
        feature_avg=True,
        model_name=args.model_name
    )
    return BoostTrack(config)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("BoostTrack for image sequence")
    parser.add_argument("--yolo_model", type=str, default="yolo11x.pt")
    parser.add_argument("--img_path", type=str, default="data/cam2_labeld")
    parser.add_argument("--model_name", type=str, 
                       choices=['convNext', 'dinov2', 'swinv2', 'CLIP', 'CLIP_RGB',
                               'La_Transformer', 'CTL', 'VIT-B/16+ICS_SSL', 'VIT_SSL_MARKET'],
                       default='VIT-B/16+ICS_SSL')
    parser.add_argument("--reid_model", type=str, default=None)
    parser.add_argument('--emb_method', type=str, default='default',
                       choices=['default', 'mean', 'enhanced_mean', 'enhanced_mean_V2'])
    parser.add_argument('--visualize', action='store_true', default = True)
    parser.add_argument('--save_video', action='store_true', default = False)
    parser.add_argument('--save_frame', action='store_true', default = False)
    
    args = parser.parse_args()
    if args.reid_model is None:
        args.reid_model = ModelConfig.WEIGHTS[args.model_name]
        print(f"Using weights: {args.reid_model}")
    
    return args

def main():
    args = parse_args()
    
    # Setup
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = False
    GeneralSettings.values['embedding_method'] = args.emb_method
    
    # Initialize components
    model = YOLO(args.yolo_model)
    tracker = setup_tracker(args)
    visualizer = Visualizer()
    vis_config = VisualizationConfig(
        visualize=args.visualize,
        save_video=args.save_video,
        save_frame=args.save_frame,
        stop_frame_ids=[142, 198, 562, 648, 656, 680]
    )
    
    cam_name = args.img_path.split('/')[-1]
    
    # Setup save directory
    save_dir = f'{args.model_name}_{cam_name}_view'
    model_name = ModelConfig.get_model_name(args.reid_model)
    
    label_dir = f'yolo_detection_xml_format/{cam_name}'
    print(f"Saving to: {save_dir}")
    
    # Create label directory
    os.makedirs(label_dir, exist_ok=True)
    
    # Initialize video writer if needed
    video_writer = VideoWriter(save_dir, model_name, args.emb_method, os.path.basename(args.img_path)) if args.save_video else None
    
    # Process images
    img_list = natsorted([f for f in os.listdir(args.img_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    deque_list = deque(maxlen=3)
    
    for idx, img_name in enumerate(tqdm(img_list)):
        frame_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(args.img_path, img_name)
        
        # Read and process image
        np_img = cv2.imread(img_path)
        if np_img is None:
            continue
            
        # YOLO detection
        results = model.predict(np_img, device='cuda', classes=[0], augment=True,
                              iou=0.35, conf=0.55)
        
        yolo_plot = results[0].plot()
        dets, xywhs = process_yolo_detection(results)
        
        if dets is None or len(dets) == 0:
            continue
            
    
        
        # Add detection numbers
        yolo_plot = visualizer.draw_detection_boxes(yolo_plot, dets)
        
        # Prepare image for tracking
        img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().cuda()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Update tracking
        targets = tracker.update(dets, img_tensor, np_img, str(frame_id))
        tlwhs, ids, confs = utils.filter_targets(targets,
                                               GeneralSettings['aspect_ratio_thresh'],
                                               GeneralSettings['min_box_area'])
        
        # Visualize tracking results
        vis_img, track_id_list = visualizer.draw_tracking_results(np_img, tlwhs, ids)
        
        
        
          

        # Save XML annotation -> yolo 디텍션 결과
        if tlwhs is not None and len(xywhs) > 0:
            xml_path = os.path.join(label_dir, f"{frame_id}.xml")
            img_path = os.path.join(os.path.dirname(label_dir), args.img_path, f"{frame_id}.jpg")
            # Convert tlwh to center format (xywh)
            boxes = []
            for tlwh in tlwhs:
                x = tlwh[0] + tlwh[2]/2  # center x
                y = tlwh[1] + tlwh[3]/2  # center y
                w = tlwh[2]  # width
                h = tlwh[3]  # height
                boxes.append([x, y, w, h])
            track_ids = [int(id) for id in ids] if ids is not None else [-1] * len(tlwhs)
  
        # Display results
        if vis_config.visualize:
            cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
            cv2.imshow('yolo', yolo_plot)
            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            cv2.imshow('Tracking', vis_img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        
        # Save frame if needed
        if idx in vis_config.stop_frame_ids and vis_config.save_frame:
            cv2.imwrite(os.path.join(save_dir, f'{idx}.jpg'), vis_img)
        
        deque_list.append(vis_img)
        print("track id list:", sorted(track_id_list))
        
        # Write video frame
        if video_writer is not None:
            video_writer.write(vis_img)
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()

if __name__ == "__main__":
    main()