import os
from typing import Dict, List, Tuple
import cv2
import numpy as np
from collections import defaultdict, deque


class IDSwitchAnalyzer:
    """
    Tracking ID Switch를 분석하는 클래스
    XML ground truth 레이블을 기준으로 tracking ID의 변경을 감지하고 분석합니다.
    """
    def __init__(self):
        self.label_to_first_track_id = {}
        self.label_to_current_track_id = {}
        self.id_switches = defaultdict(list)
        self.frame_count = 0
        self.frame_history = {}
        self.label_appearances = defaultdict(list)
        self.label_to_recent_bbox = {}
        self.track_to_xml_mapping = {}
        self.total_switches = 0
        self.track_img_buffer = deque(maxlen=30)

    def calculate_iou(self, bbox1, bbox2):
        """
        두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다.
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        intersection = max(0, abs(x2 - x1)) * max(0, abs(y2 - y1))
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def update(self, frame_id: int, yolo_track_mapping: Dict[int, int], 
               yolo_label_mapping: Dict[int, str], yolo_bbox_mapping: Dict[int, Tuple[int, int, int, int]]) -> Dict:
        """
                
        현재 프레임의 tracking 결과와 XML 레이블을 분석하여 ID switch를 감지합니다.
        
        Args:
            frame_id: 현재 프레임 번호
            yolo_track_mapping: YOLO 탐지 순서와 tracking ID 매핑 {yolo_idx: track_id}
            yolo_label_mapping: YOLO 탐지 순서와 XML 레이블 매핑 {yolo_idx: xml_id} 탐지순서 : 실제정답
            yolo_bbox_mapping: YOLO 탐지 순서와 바운딩 박스 매핑 {yolo_idx: (x1, y1, x2, y2)} 탐지순서 : 바운딩박스          
        
        Returns:
            현재 프레임의 ID switch 분석 결과
        """
        self.frame_count += 1
        current_switches = []
        frame_mappings = {}

        # 1단계: 현재 tracking 중인 객체들의 xml_id 매핑 확인 일단 tracking중인 객체가있나요? 즉 xml 과 tracking id 매핑이 이루어진 추적중인 객체가있는지 확인중.
        current_track_to_xml = {}
        for xml_id, track_id in self.label_to_current_track_id.items():
            current_track_to_xml[track_id] = xml_id
            
            
            

        xml_id_detections = {}
        best_match_scores = defaultdict(float)
        used_track_ids = set()
        
        # 2단계: YOLO 탐지와 tracking box 매칭 시 IOU 체크
        for yolo_idx in yolo_track_mapping:
            if yolo_idx in yolo_label_mapping and yolo_idx in yolo_bbox_mapping:
                track_id = yolo_track_mapping[yolo_idx] # yolo_idx 탐지순서가 가지고있는 tracking_id (가변)
                xml_id = yolo_label_mapping[yolo_idx] # yolo_idx 탐지순서가 가지고있는 실제 xml_label (불변)
                bbox = yolo_bbox_mapping[yolo_idx] # yolo_idx 가 가지고있는 yolo_box 영역 실제 영역
                
                # 현재 추적중인 객체인가요? 
                if track_id in current_track_to_xml:
                    current_xml = current_track_to_xml[track_id] # tracking_id 가 가지고있는 실제 XML_ID // 이전 프레임에서 해당 track_id가 추적하던 xml_id
                    if current_xml != xml_id:  # 다른 xml_id와 매칭되려고 할 때  // 현재 프레임에서 해당 track_id에 매칭된 xml_id
                        
                        
                    
                        # 이중감지 방지를 위한 코드 xml_id 이중디텍션일떄 서로 분류할려고
                        if current_xml in self.label_to_recent_bbox:
                            if track_id in used_track_ids:
                                continue
                                
                            iou = self.calculate_iou(self.label_to_recent_bbox[current_xml], bbox)
                            current_score = iou

                            if xml_id in best_match_scores:
                                if current_score <= best_match_scores[xml_id]:
                                    continue

                            if iou < 0.7: # 임계값 자율적으로 조정이
                                print(f"Warning: YOLO label {xml_id} IOU too low: {iou}")
                                continue
                                
                            best_match_scores[xml_id] = current_score
                            used_track_ids.add(track_id)
                
                
                            
                            xml_id_detections[xml_id] = {
                                'track_id': track_id,
                                'bbox': bbox,
                                'yolo_idx': yolo_idx
                            }

            det = xml_id_detections[xml_id]
            track_id = det['track_id']
            bbox = det['bbox']
            frame_mappings[track_id] = xml_id

            if xml_id not in self.label_appearances:
                self.label_appearances[xml_id] = []
            self.label_appearances[xml_id].append(frame_id)


            # Handle track_id conflict with stricter IOU check
            prev_xml_id = self.track_to_xml_mapping.get(track_id)
            if prev_xml_id is not None and prev_xml_id != xml_id:
                if prev_xml_id in self.label_to_recent_bbox:
                    iou = self.calculate_iou(self.label_to_recent_bbox[prev_xml_id], bbox)
                    if iou < 0.5:  # IOU가 낮으면 재할당하지 않음
                        print(f"Prevented ID reassignment: Track ID {track_id} from XML {prev_xml_id} to {xml_id} (IOU: {iou})")
                        continue

            if xml_id in self.label_to_current_track_id:
                current_track_id = self.label_to_current_track_id.get(xml_id)
                if track_id != current_track_id:
                    frames_since_last = frame_id - self.label_appearances[xml_id][-2] if len(self.label_appearances[xml_id]) > 1 else 0
                    switch_info = {
                        'frame_id': frame_id,
                        'xml_id': xml_id,
                        'old_track_id': current_track_id,
                        'new_track_id': track_id,
                        'frames_since_last': frames_since_last,
                        'total_switches': len(self.id_switches[xml_id]) + 1,
                        'iou': self.calculate_iou(self.label_to_recent_bbox[xml_id], bbox)
                    }
                    self.id_switches[xml_id].append(switch_info)
                    current_switches.append(switch_info)
                    self.total_switches += 1
                    print(f"ID Switch detected: XML ID {xml_id} Track ID {current_track_id} -> {track_id}")

                    # Remove previous track_id mapping
                    if current_track_id in self.track_to_xml_mapping:
                        del self.track_to_xml_mapping[current_track_id]

                # Update mappings
                self.track_to_xml_mapping[track_id] = xml_id
                self.label_to_current_track_id[xml_id] = track_id
            else:
                self.label_to_first_track_id[xml_id] = track_id
                self.label_to_current_track_id[xml_id] = track_id
                self.id_switches[xml_id] = []
                self.track_to_xml_mapping[track_id] = xml_id

            self.label_to_recent_bbox[xml_id] = bbox

        self.frame_history[frame_id] = frame_mappings
        return {
            'frame_id': frame_id,
            'current_switches': current_switches,
            'frame_mappings': frame_mappings,
            'initial_mappings': {v: k for k, v in self.label_to_first_track_id.items()}
        }

    def visualize_id_switches(self, image: np.ndarray, switches: List[Dict], 
                            current_mappings: Dict[int, str], frame_id: int, track_img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        현재 프레임의 ID switch 정보를 시각화합니다.
        
        Args:
            image: 원본 이미지
            switches: ID switch 정보 리스트
            current_mappings: 현재 프레임의 track_id와 xml_id 매핑
            frame_id: 현재 프레임 번호
            track_img: tracking 결과 이미지
            
        Returns:
            시각화된 이미지와 이전 프레임 이미지 리스트
        """
        vis_img = image.copy()
        previous_track_images = []
        
        # 이미지 상단에 현재 프레임 정보 표시
        cv2.putText(vis_img, f"Frame: {frame_id}.jpg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 현재 매핑 정보 표시 (Tracking ID -> XML ID)
        y_offset = 70
        cv2.putText(vis_img, "Current ID Mappings (Tracking -> Ground Truth):", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # 현재 활성화된 tracking ID들의 매핑 표시
        if current_mappings:
            for track_id, xml_id in sorted(current_mappings.items()):
                # tracking ID가 switch 없이 유지되고 있는지 확인
                has_switches = any(s['xml_id'] == xml_id for s in switches)
                color = (0, 255, 0) if not has_switches else (200, 200, 200)
                cv2.putText(vis_img, f"Track ID {track_id} -> Ground Truth {xml_id}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        
        # ID switch 히스토리 표시
        y_offset += 20
        cv2.putText(vis_img, "ID Switch History:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        
        # ID switch가 발생한 객체들의 xml_id 수집
        switched_xml_ids = set()
        for switch in switches:
            if switch['frame_id'] == frame_id:  # 현재 프레임에서 발생한 switch만 고려
                switched_xml_ids.add(switch['xml_id'])
        
        # 현재 프레임의 바운딩 박스 표시
        for xml_id in switched_xml_ids:
            if xml_id in self.label_to_recent_bbox:
                bbox = self.label_to_recent_bbox[xml_id]
                bbox = list(map(int, bbox))
                # 빨간색으로 바운딩 박스 그리기
                cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                # ID 텍스트 표시 (XML ID와 누적 switch 횟수)
                switch_count = len(self.id_switches[xml_id])
                cv2.putText(vis_img, f"ID: {xml_id} (Switches: {switch_count})", 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 모든 switch 표시 (시간순)
        for switch in switches:
            text = f"Frame {switch['frame_id']}: Ground Truth {switch['xml_id']} "
            text += f"(Track {switch['old_track_id']} -> {switch['new_track_id']} )"
            if switch['frames_since_last'] > 1:
                text += f" [Gap: {switch['frames_since_last']} frames]"
            color = (0, 0, 255) if switch['frame_id'] == frame_id else (200, 200, 200)
            cv2.putText(vis_img, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        # 총 ID switch 횟수 표시
        cv2.putText(vis_img, f"Total ID Switches: {self.total_switches}", 
                    (10, vis_img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 현재 프레임의 detection 정보 생성
        current_detections = {
            'frame_id': frame_id,
            'image': track_img.copy(),
            'detections': {}
        }
        
        # 현재 프레임의 detection 정보 추가
        for track_id, xml_id in current_mappings.items():
            if xml_id in self.label_to_recent_bbox:
                current_detections['detections'][track_id] = {
                    'xml_id': xml_id,
                    'bbox': self.label_to_recent_bbox[xml_id],
                    'visual_pos': self._calculate_visual_position(self.label_to_recent_bbox[xml_id])
                }
        
        # tracking 이미지와 detection 정보를 함께 버퍼에 저장
        self.track_img_buffer.append(current_detections)
        
        # 현재 프레임에서 ID switch가 발생한 객체들 확인
        current_switches = [s for s in switches if s['frame_id'] == frame_id]
        for switch in current_switches:
            xml_id = switch['xml_id']
            old_track_id = switch['old_track_id']
            found = False  # 최근 프레임 발견 여부 플래그

            # 버퍼를 역순으로 검색 (최신 → 오래된 순)
            for past_frame in reversed(list(self.track_img_buffer)):
                if past_frame['frame_id'] >= frame_id:
                    continue  # 현재/미래 프레임 건너뛰기

                # 해당 프레임의 모든 detection 확인
                for tid, det in past_frame['detections'].items():
                    if tid == old_track_id and det['xml_id'] == xml_id:
                        previous_track_images.append({
                            'frame_id': past_frame['frame_id'],
                            'image': past_frame['image'],
                            'bbox': det['bbox'],
                            'position': det['visual_pos'],
                            'xml_id': xml_id,
                            'track_id': old_track_id
                        })
                        found = True
                        break  # 가장 최근 프레임 발견 → 내부 루프 종료

                if found:
                    break  # 외부 루프 종료

        return vis_img, previous_track_images

    def _calculate_visual_position(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        바운딩 박스의 중심점 계산
        """
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        return (center_x, center_y)


def get_bboxes_from_xml(xml_path: str) -> List[Dict]:
    """
    XML 파일에서 바운딩 박스 정보를 추출합니다.
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        label = obj.find('name').text
        
        bboxes.append({
            'name': label,
            'bbox': (x1, y1, x2, y2)
        })
    
    return bboxes


def generate_unique_color(label: str) -> Tuple[int, int, int]:
    """
    레이블에 대한 고유한 색상을 생성합니다.
    """
    import hashlib
    
    hash_value = int(hashlib.md5(label.encode()).hexdigest(), 16)
    r = (hash_value % 256)
    g = ((hash_value // 256) % 256)
    b = ((hash_value // (256 * 256)) % 256)
    
    return (b, g, r)
