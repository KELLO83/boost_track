import os
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from natsort import natsorted

target_path = 'updated_xml_no_duplicates'


id = {}
def get_color_by_id(id_num):
    
    if id_num in id:
        return id[id_num]
    else:
        color = [random.randint(150, 255) for _ in range(3)]
        id[id_num] = color
        return color
    
    

def draw_boxes_from_xml(image_path, xml_path):
    """XML 파일에서 바운딩 박스 정보를 읽어와 이미지에 그리기"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # XML 파싱
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 각 객체에 대해 바운딩 박스 그리기
    for obj in root.findall('.//object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        if bbox is not None:
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            # ID 추출 (name 필드에서 숫자만 추출)
            try:
                id_num = ''.join(filter(str.isdigit, name))
                if id_num:
                    id_text = f"ID: {id_num}"
                else:
                    id_text = name
            except:
                id_text = name

            # 바운딩 박스 그리기    
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), get_color_by_id(id_num), 2)
            
            # ID 텍스트 그리기
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, id_text, (xmin, ymin-10), font, 0.5, get_color_by_id(id_num), 2)

    # 이미지 파일명 표시
    image_name = os.path.basename(image_path)
    cv2.putText(image, image_name, (10, 30), font, 1, (255, 255, 255), 2)
    
    return image

def visualize_dataset():
    """데이터셋의 모든 이미지와 XML 시각화"""
    # 이미지와 XML 파일 가져오기
    jpg_files = natsorted(glob(os.path.join(target_path, '*.jpg')))
    
    for jpg_file in jpg_files:
        # 대응하는 XML 파일 경로
        xml_file = os.path.join(target_path, os.path.splitext(os.path.basename(jpg_file))[0] + '.xml')
        
        if not os.path.exists(xml_file):
            print(f"Warning: XML file not found for {jpg_file}")
            continue

        # 바운딩 박스 그리기
        image_with_boxes = draw_boxes_from_xml(jpg_file, xml_file)
        
        if image_with_boxes is not None:
            # 이미지 표시
            cv2.imshow('Visualization', image_with_boxes)
            
            # 키 입력 대기
            key = cv2.waitKey(0)
            
            # 'q' 키를 누르면 종료
            if key == ord('q'):
                break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_dataset()
