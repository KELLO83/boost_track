from optparse import NO_DEFAULT
import os
import cv2
import numpy as np
import torch
import argparse
from collections import deque
from typing import Dict, List, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from ultralytics import YOLO
from natsort import natsorted
import xml.etree.ElementTree as ET
from default_settings import GeneralSettings, BoostTrackConfig
from tracker.boost_track import BoostTrack
import utils
#from id_switch_analyzer import IDSwitchAnalyzer
from id_switch_analyzer2 import IDSwitchAnalyzer
from typing import Dict, List, Tuple , Optional
import random
from collections import deque

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

    def draw_detection_boxes(self, image: np.ndarray, dets: np.ndarray, yolo_boxes: List = None, frame_mapping: Dict = None) -> np.ndarray:
        vis_img = image.copy()

        box_mapping = {}
        for i, (x1, y1, x2, y2, conf) in enumerate(dets): 
            if frame_mapping is not None and i in frame_mapping:
                track_id = frame_mapping[i]
                color = self.get_id_color(track_id)
                
                box_mapping.update({i: [track_id, x1, y1, x2, y2 , conf]})
                
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(vis_img, f"Y#{i}->T#{track_id}", (int(x1), int(y1)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            else:
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                cv2.putText(vis_img, f"Y#{i}", (int(x1), int(y1)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        return vis_img , box_mapping

    def draw_tracking_results(self, image: np.ndarray, tlwhs: List, ids: List[int]) -> Tuple[np.ndarray, List[int]]:
        vis_img = image.copy()
        track_id_list = []
        
        for tlwh, track_id in zip(tlwhs, ids):
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            color = self.get_id_color(track_id)
            
            mid_x , mid_y = x1 + w/2 , y1 + h/2
            
            if track_id not in track_id_list:
                track_id_list.append(int(track_id))
                
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(vis_img, f"ID: {track_id}", (int(mid_x), int(mid_y)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                        
        return vis_img, track_id_list

    def draw_xml_boxes(self, image: np.ndarray, xml_boxes: List , idx : int) -> np.ndarray:
        """XML 파일의 박스와 ID를 시각화"""
        vis_img = image.copy()
        for name, xmin, ymin, xmax, ymax in xml_boxes:
            try:
                id_num = int(''.join(filter(str.isdigit, name)))
                color = self.get_id_color(id_num)
                
                cv2.rectangle(vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(vis_img, f"ID: {id_num}", (int(xmin), int(ymin)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            except:
                cv2.rectangle(vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 2)
                cv2.putText(vis_img, name, (int(xmin), int(ymin)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(vis_img, f"Frame: {idx}", (10, 30),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        return vis_img 

    def generate_unique_color(self, label: str) -> Tuple[int, int, int]:
        hash_value = hash(label)
        
        r = (hash_value & 0xFF0000) >> 16
        g = (hash_value & 0x00FF00) >> 8
        b = hash_value & 0x0000FF
        
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        if brightness < 50:
            r = min(r + 100, 255)
            g = min(g + 100, 255)
            b = min(b + 100, 255)
        
        return (int(r), int(g), int(b))

    def visualize_yolo_xml_mapping(self, image: np.ndarray, yolo_boxes: List[List[float]], xml_file: str, yolo_label_mapping: Dict[int, str]) -> np.ndarray:
        """
        YOLO 박스와 XML 박스의 매핑 관계를 시각화합니다.
        
        Args:
            image: 원본 이미지
            yolo_boxes: YOLO 탐지 결과 박스 리스트 [x1, y1, x2, y2]
            xml_file: XML 파일 경로
            yolo_label_mapping: YOLO 인덱스와 XML 레이블 매핑 딕셔너리
        
        Returns:
            np.ndarray: 매핑 관계가 시각화된 이미지
        """
        vis_img = image.copy()
        xml_boxes = get_bboxes_from_xml(xml_file)
        
        # 색상 맵 생성 (매칭된 박스 쌍마다 고유한 색상 사용)
        colors = {}
        for xml_label in set(yolo_label_mapping.values()):
            colors[xml_label] = self.generate_unique_color(xml_label)
        
        # YOLO 박스 그리기
        for yolo_idx, yolo_box in enumerate(yolo_boxes):
            if yolo_idx in yolo_label_mapping:
                x1, y1, x2, y2 = map(int, yolo_box[:4])
                xml_label = yolo_label_mapping[yolo_idx]
                color = colors[xml_label]
                
                # YOLO 박스는 실선으로 그리기
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # YOLO 인덱스와 매칭된 XML 레이블 표시
                label = f"YOLO_{yolo_idx}: {xml_label}"
                cv2.putText(vis_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 매칭된 XML 박스 찾기
                for xml_box in xml_boxes:
                    xml_name = xml_box[0]
                    if xml_name == xml_label:
                        xml_x1, xml_y1, xml_x2, xml_y2 = map(int, xml_box[1:])
                        
                        # XML 박스는 점선으로 그리기
                        for i in range(0, 360, 20):  # 점선 효과를 위한 작은 선분들
                            start_angle = i
                            end_angle = min(i + 10, 360)
                            
                            # 박스의 각 변을 점선으로 그리기
                            # 상단 가로선
                            cv2.line(vis_img, 
                                   (int(xml_x1 + (xml_x2-xml_x1)*start_angle/360), xml_y1),
                                   (int(xml_x1 + (xml_x2-xml_x1)*end_angle/360), xml_y1),
                                   color, 2)
                            # 하단 가로선
                            cv2.line(vis_img,
                                   (int(xml_x1 + (xml_x2-xml_x1)*start_angle/360), xml_y2),
                                   (int(xml_x1 + (xml_x2-xml_x1)*end_angle/360), xml_y2),
                                   color, 2)
                            # 왼쪽 세로선
                            cv2.line(vis_img,
                                   (xml_x1, int(xml_y1 + (xml_y2-xml_y1)*start_angle/360)),
                                   (xml_x1, int(xml_y1 + (xml_y2-xml_y1)*end_angle/360)),
                                   color, 2)
                            # 오른쪽 세로선
                            cv2.line(vis_img,
                                   (xml_x2, int(xml_y1 + (xml_y2-xml_y1)*start_angle/360)),
                                   (xml_x2, int(xml_y1 + (xml_y2-xml_y1)*end_angle/360)),
                                   color, 2)
                        
                        # XML 박스 레이블 표시
                        label = f"XML: {xml_name}"
                        cv2.putText(vis_img, label, (xml_x1, xml_y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_img

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

def process_yolo_detection(results) -> Optional[Tuple[np.ndarray, np.ndarray, List[List]]]:
    dets = []
    xywhs = []
    yolo_boxes = []  
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = boxes.cls[i].cpu().numpy()
            
            if cls == 0:  
                dets.append([x1, y1, x2, y2, conf])
                w = x2 - x1
                h = y2 - y1
                x_center = x1 + w/2
                y_center = y1 + h/2
                xywhs.append([x_center, y_center, w, h])
                yolo_boxes.append([x1, y1, x2, y2, conf, i])  
                
    return (np.array(dets) if dets else None, 
            np.array(xywhs) if xywhs else None,
            yolo_boxes if yolo_boxes else None)

def setup_tracker(args) -> BoostTrack:
    config = BoostTrackConfig(
        reid_model_path=f'external/weights/{args.reid_model}',
        device='cuda',
        max_age=15,
        min_hits=0,
        det_thresh=0.55, # 탐지 신뢰도 임계값
        iou_threshold=0.65,
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

def get_bboxes_from_xml(xml_file):
    """XML 파일에서 bounding box 정보 추출"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    
    for obj in root.findall('.//object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bboxes.append((name, xmin, ymin, xmax, ymax))
    
    return sorted(bboxes) 


def map_yolo_to_xml_labels(yolo_boxes: List[List[float]], xml_file: str) -> Dict[int, str]:
    """
    YOLO 객체 탐지 결과와 XML 파일의 박스를 비교하여 IOU가 가장 높은 것을 매핑합니다.
    
    Args:
        yolo_boxes: YOLO 탐지 결과 박스 리스트 [x1, y1, x2, y2]
        xml_file: XML 파일 경로
    
    Returns:
        Dict[int, str]: YOLO 탐지 순서를 키로, XML name 필드 값을 값으로 하는 딕셔너리
    """
    # XML 파일에서 박스 정보 추출
    xml_boxes = get_bboxes_from_xml(xml_file)
    
    yolo_label_mapping = {}
    
    # 각 YOLO 박스에 대해
    for yolo_idx, yolo_box in enumerate(yolo_boxes):
        max_iou = 0
        best_xml_name = None
        
        # 모든 XML 박스와 비교
        for xml_box in xml_boxes:
            # XML 박스 좌표와 이름 추출
            x1, y1, x2, y2 = xml_box[1:]
            xml_name = xml_box[0]
            
            # IOU 계산
            iou = calculate_iou(yolo_box, [x1, y1, x2, y2])
            
            # 현재까지의 최대 IOU보다 크면 업데이트
            if iou > max_iou:
                max_iou = iou
                best_xml_name = xml_name
        
        # IOU가 임계값(예: 0.5) 이상인 경우에만 매핑
        if max_iou >= 0.5 and best_xml_name is not None:
            yolo_label_mapping[yolo_idx] = best_xml_name
    
    return yolo_label_mapping

def calculate_iou(box1, box2):
    """
    두 박스 간의 IoU(Intersection over Union)를 계산
    box 형식: [x1, y1, x2, y2]
    """

    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0.0
    
    return iou

def match_detections_with_xml(box_mapping: Dict, xml_result: List , track_boxes : List) -> Dict:
    """ 
    YOLO 탐지 순서와 XML ID 간의 매핑을 분석
    
    Args:
    - box_mapping: YOLO 탐지 순서와 : [track_id , x1,y1,x2,y2,conf]
    - xml_result: XML에서 추출한 박스 정보 [xml_id, x1,y1,x2,y2]
    
    Returns:
    - 매핑 결과 딕셔너리
      {
        'yolo_to_tracking': {yolo_idx: tracking_id},
        'yolo_to_xml': {yolo_idx: xml_id},
        'tracking_to_xml': {tracking_id: xml_id}
      }
    """
    # XML ID 추출 (name에서 숫자 추출)
    xml_id_mapping = {}
    for idx, (name, xmin, ymin, xmax, ymax) in enumerate(xml_result):
        try:
            xml_id = int(''.join(filter(str.isdigit, name)))
            xml_id_mapping[idx] = xml_id
        except:
            # ID를 추출할 수 없는 경우 스킵
            pass
    
    # IoU를 사용하여 YOLO 박스와 XML 박스 매핑
    yolo_to_xml = {}
    for yolo_idx, (track_info) in box_mapping.items():
        tracking_id, x1, y1, x2, y2, conf = track_info
        yolo_box = [x1, y1, x2, y2]
        
        best_iou = 0
        best_xml_idx = None
        for xml_idx, (name, xmin, ymin, xmax, ymax) in enumerate(xml_result):
            xml_box = [xmin, ymin, xmax, ymax]
            iou = calculate_iou(yolo_box, xml_box)
            
            if iou > best_iou:
                best_iou = iou
                best_xml_idx = xml_idx

        if best_iou >= 0.7: # 해당코드가 문제가 발생할수도있음 만약에 XML박스와 yolo박스가 임계치가 만족하지않는다면
            yolo_to_xml[yolo_idx] = best_xml_idx
    
    # 최종 매핑 결과 생성
    """
    YOLO_IDX 탐지순서 : 부여받은 Tracking_ID (가변)
    YOLO_IDX : XML_IDX (불변)
    Tracking_ID : XML_ID  
    
    """
    mapping_result = {
        'yolo_to_tracking': {k: v[0] for k, v in box_mapping.items()}, # yolo_IDX  : tracking_ID
        'yolo_to_xml': yolo_to_xml, # yolo_IDX : XML_IDX
        'tracking_to_xml': {} # tracking ID : XML_ID 
    }
    
    # tracking_to_xml 매핑 추가
    for yolo_idx, tracking_id in mapping_result['yolo_to_tracking'].items():
        if yolo_idx in mapping_result['yolo_to_xml']:
            xml_idx = mapping_result['yolo_to_xml'][yolo_idx]
            if xml_idx in xml_id_mapping:
                mapping_result['tracking_to_xml'][tracking_id] = xml_id_mapping[xml_idx]
    
    return mapping_result


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
      
    img_files = natsorted([f for f in os.listdir(args.img_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    xml_files = natsorted([f for f in os.listdir(args.img_path) if f.endswith('.xml')])
    
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = False
    GeneralSettings.values['embedding_method'] = args.emb_method

    model = YOLO(args.yolo_model)
    tracker = setup_tracker(args)
    visualizer = Visualizer()
    id_switch_analyzer = IDSwitchAnalyzer()  
    vis_config = VisualizationConfig(
        visualize=args.visualize,
        save_video=args.save_video,
        save_frame=args.save_frame,
        stop_frame_ids=[i for i in range(len(img_files)) if i%100 == 0]
    )
    
    cam_name = args.img_path.split('/')[-1]

    save_dir = f'{args.model_name}_{cam_name}_view'
    model_name = ModelConfig.get_model_name(args.reid_model)
    
    video_writer = VideoWriter(save_dir, model_name, args.emb_method, os.path.basename(args.img_path)) if args.save_video else None
    
    valid_pairs = []
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        xml_file = f"{base_name}.xml"
        not_found_list = []
        if xml_file in xml_files:
            valid_pairs.append((img_file, xml_file))
        else:
            not_found_list.append(img_file)
            
    yolo_label_mapping  = {}

    previous_track_images = deque(maxlen=5)
    
    for idx, (img_name, xml_name) in enumerate(tqdm(valid_pairs)):
        frame_id = int(os.path.splitext(img_name)[0]) 
        img_path = os.path.join(args.img_path, img_name)
        xml_path = os.path.join(args.img_path, xml_name)
        
        np_img = cv2.imread(img_path)
        if np_img is None:
            continue
            
        results = model.predict(np_img, device='cuda', classes=[0], augment=True,
                              iou=0.65, conf=0.55) # ultimcy.data.
        
        yolo_plot = results[0].plot()
        boxes = results[0].boxes
        for i in range(len(boxes)):
            if boxes.cls[i].cpu().numpy() == 0:  
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cv2.putText(yolo_plot, f"#{i}", (int(x1), int(y1)-10),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(yolo_plot, f"Frame: {idx} ({img_name})", (10, 30),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)        
        
        dets, xywhs, yolo_boxes = process_yolo_detection(results) 
        """
        ====    Yolo탐지 전처리 결과  ===
        Dets = [[x1,y1,x2,y2,conf] , [...]]
        xywhs = [x_center , y_center , w, h]
        yolo_boxes = [x1,y1,x2,y2,conf , yolo_IDX]
        
        """
        
        if dets is None or len(dets) == 0: # 검출객체가없다면 Image Pass
            continue
        
        img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().cuda()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        targets = tracker.update(dets, img_tensor, np_img, str(frame_id)) # 반환 형태: [[x1, y1, x2, y2, score, id], ...]
        tlwhs, ids, confs = utils.filter_targets(targets,
                                               GeneralSettings['aspect_ratio_thresh'],
                                               GeneralSettings['min_box_area'])
        
        frame_mapping = {}
        track_boxes = []
        
        
        
        
        """
        해당코드는   Deprecation  예정
        """
        if yolo_boxes is not None and ids is not None: # 무언가 탐지를한다면?
            for yolo_box in yolo_boxes:
                det_idx = yolo_box[5]  # yolo탐지순서 -> Yolo_IDX
                max_iou = 0
                matched_id = None
                for tlwh, track_id in zip(tlwhs, ids):
                    x1, y1 = tlwh[0], tlwh[1]
                    x2, y2 = x1 + tlwh[2], y1 + tlwh[3]
                    track_box = [x1, y1, x2, y2]
                    track_boxes.append(track_box)
                    iou = calculate_iou(yolo_box[:4], track_box) # yolo box 와 track box IOU를 통햬 먜칭진행
                    if iou > max_iou and iou > 0.6: # max이면서 겹칩정도가 적어도 N 이상은 되야함 
                        max_iou = iou
                        matched_id = track_id
                        
                if matched_id is not None:
                    frame_mapping[det_idx] = matched_id # Yolo_INDEX : Track_ID Yolo탐지순서가 어떤 트랙ID를 부여받았는지?

            yolo_tracking_id, box_mapping = visualizer.draw_detection_boxes(np_img, dets, yolo_boxes, frame_mapping)    #{yolo idx : [track_id , x1,y1,x2,y2,conf]}
        track_img, track_id_list = visualizer.draw_tracking_results(np_img, tlwhs, ids)
    
        print("=============================")
        print('Targets : ',targets)
        print("Dets : " , dets)
        print("TLWHS" , tlwhs)
        print("Track_boxes " , track_boxes)
        print("Box_Mapping :" ,box_mapping)
        print("=============================")
      
      
        # xml 레이블값들의 정보를 가져오고 시각화하는 함수임 레이블링 확인 코드
        
        xml_result = get_bboxes_from_xml(xml_path) #  [xml_id , x1,y1,x2,y2 ] 
        xml_vis = visualizer.draw_xml_boxes(np_img, xml_result , idx) 
        
        # 매핑 분석
    
        """
        YOLO_IDX 탐지순서 : 부여받은 Tracking_ID (가변)
        YOLO_IDX : XML_IDX (불변)
        Tracking_ID : XML_ID  
        """
        mapping_analysis = match_detections_with_xml(box_mapping, xml_result , track_boxes)
        
        # # 매핑 결과 출력
        print("\n=== Mapping Analysis ===")
        # print("YOLO to Tracking ID:", mapping_analysis['yolo_to_tracking'])
        # print("YOLO to XML ID:", mapping_analysis['yolo_to_xml'])
        # print("Tracking ID to XML ID:", mapping_analysis['tracking_to_xml'])
        
        # YOLO 객체 탐지 순서와 XML 박스를 매핑
        yolo_label_mapping = map_yolo_to_xml_labels(yolo_boxes, xml_path)
        
        # YOLO 바운딩 박스 매핑 생성
        yolo_bbox_mapping = {}
        for i, box in enumerate(yolo_boxes):
            x1, y1, x2, y2 = box[:4]  # box는 [x1, y1, x2, y2, conf, det_idx] 형식
            yolo_bbox_mapping[int(box[5])] = (float(x1), float(y1), float(x2), float(y2))
        
        # # ID Switch 분석 업데이트
        # switch_analysis = id_switch_analyzer.update(
        #     frame_id=frame_id,
        #     yolo_track_mapping=mapping_analysis['yolo_to_tracking'], 
        #     yolo_label_mapping=yolo_label_mapping,
        #     yolo_bbox_mapping=yolo_bbox_mapping,
        # )
        
        
        # # 현재 프레임에서 ID switch가 발생했다면 출력
        # IS_ID_SW = False
        # if switch_analysis['current_switches']:
        #     print(f"\nFrame {frame_id}: ID Switches detected:")
        #     for switch in switch_analysis['current_switches']:
        #         print(f"  Ground Truth {switch['xml_id']}: Track ID changed from {switch['old_track_id']} to {switch['new_track_id']}")
        #         if switch['frames_since_last'] > 0:
        #             print(f"    After {switch['frames_since_last']} frames from last appearance")
                    
        #     IS_ID_SW = True
        
        # # ID Switch 분석 시각화
        # id_switch_vis, previous_track_images = id_switch_analyzer.visualize_id_switches(
        #     image = np_img.copy(),
        #     switches = switch_analysis['current_switches'],
        #     current_mappings=switch_analysis['frame_mappings'],
        #     frame_id = frame_id,
        #     track_img = track_img  # 바운딩 박스와 ID가 표시된 이미지 전달
        # )
        
        cv2.putText(track_img, f"Frame: {frame_id}", (10, 30),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        
        # YOLO와 XML 매핑 시각화
        mapping_vis = visualizer.visualize_yolo_xml_mapping(np_img.copy(), yolo_boxes, xml_path, yolo_label_mapping)
        
        # Display results
        if vis_config.visualize :
            cv2.namedWindow('yolo_plot', cv2.WINDOW_NORMAL)
            cv2.imshow('yolo_plot', yolo_plot)
            cv2.namedWindow('yolo_tracking_id', cv2.WINDOW_NORMAL)
            cv2.imshow('yolo_tracking_id', yolo_tracking_id)
            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            cv2.imshow('Tracking', track_img)
            
            cv2.namedWindow('XML_GT', cv2.WINDOW_NORMAL)
            cv2.imshow('XML_GT', xml_vis)
            
            cv2.namedWindow('YOLO-XML Mapping', cv2.WINDOW_NORMAL)
            cv2.imshow('YOLO-XML Mapping', mapping_vis)
            
            # cv2.namedWindow('ID Switch Analysis', cv2.WINDOW_NORMAL)
            # cv2.imshow('ID Switch Analysis', id_switch_vis)
            
            # 이전 tracking 이미지 표시
            prev_window_names = []  # 이전 tracking 이미지 창 이름 저장
            for i, prev_img_info in enumerate(previous_track_images):
                window_name = f"Previous Track (Frame {prev_img_info['frame_id']}, XML ID: {prev_img_info['xml_id']}, Track ID: {prev_img_info['track_id']})"
                prev_window_names.append(window_name)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, prev_img_info['image'])
            
            cv2.waitKey(0)
            
            # 이전 tracking 이미지 창들만 닫기
            for window_name in prev_window_names:
                cv2.destroyWindow(window_name)

            
        # Write video frame
        if video_writer is not None:
            video_writer.write(track_img)
        
 
    # Cleanup
    if video_writer is not None:
        video_writer.release()
        
    # 분석 결과 출력
    id_switch_analyzer.print_summary()

if __name__ == "__main__":
    main()