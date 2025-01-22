import os
import xml.etree.ElementTree as ET
import natsort
def parse_bounding_boxes(xml_file):
    """
    XML 파일을 파싱하여, (xmin, ymin, xmax, ymax) 형태의 박스 좌표와
    그에 대응되는 name 값을 리스트로 반환합니다.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects_info = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        objects_info.append({
            'name': name,
            'bbox': (xmin, ymin, xmax, ymax)
        })
    return objects_info

def calculate_iou(box1, box2):
    """
    두 박스의 IoU (Intersection over Union)를 계산하여 반환합니다.
    box = (xmin, ymin, xmax, ymax)
    """
    (x1, y1, x2, y2) = box1
    (x1_p, y1_p, x2_p, y2_p) = box2

    # 교집합 영역
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    # 교집합이 존재하지 않는 경우
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)

    # 각 박스 면적
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    # 합집합 영역
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def update_person_to_number(source_xml, target_xml):
    """
    source_xml(숫자 <name>을 갖는 XML)과 target_xml("person"만 있는 XML)을 비교하여
    target_xml의 <name> 필드를 기준 XML의 <name> 숫자로 교체한 뒤,
    ElementTree 객체를 반환합니다.
    """
    source_objects = parse_bounding_boxes(source_xml)
    target_tree = ET.parse(target_xml)
    target_root = target_tree.getroot()

    # 'object' 태그 순회
    for target_obj in target_root.findall('object'):
        target_name_elem = target_obj.find('name')
        if target_name_elem is None:
            continue

        # <name>이 'person'인 경우에만 교체 진행
        if target_name_elem.text.lower() == 'person':
            bndbox = target_obj.find('bndbox')
            if bndbox is None:
                continue

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            target_box = (xmin, ymin, xmax, ymax)

            highest_iou = 0.0
            best_name = None

            # source_xml과 IoU 비교하여 가장 큰 IoU 갖는 이름 찾기
            for source_obj in source_objects:
                source_box = source_obj['bbox']
                iou = calculate_iou(target_box, source_box)

                if iou > highest_iou:
                    highest_iou = iou
                    best_name = source_obj['name']

            # IoU가 0보다 크면 교체
            if best_name and highest_iou > 0:
                target_name_elem.text = best_name

    # 수정 후 트리 객체 반환
    return target_tree

def batch_update_person(
    source_dir,      # 기준 XML 폴더 경로
    target_dir,      # "person" XML 폴더 경로
    output_dir       # 결과 XML 폴더 경로
):
    """
    두 폴더 내 동일한 파일 이름의 XML을 순회하며, 타겟 XML의 <name>을
    기준(소스) XML을 바탕으로 교체한 뒤 output_dir에 저장합니다.
    """

    # output_dir이 존재하지 않으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # 소스 폴더 내 파일명을 기준으로 순회
    source_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.xml')]

    for source_file in source_files:
        source_path = os.path.join(source_dir, source_file)
        target_path = os.path.join(target_dir, source_file)

        # target_path가 실제로 있을 때만 처리
        if os.path.exists(target_path):
            updated_tree = update_person_to_number(source_path, target_path)

            # 결과물 저장 경로 만들기
            output_path = os.path.join(output_dir, source_file)
            updated_tree.write(output_path, encoding="utf-8", xml_declaration=True)
            print(f"{source_file} 처리 완료 → {output_path}")
        else:
            print(f"대상 XML이 없어서 건너뜀: {source_file}")

# ---------------------
# 사용 예시
if __name__ == "__main__":
    source_xml_path = "cam2_xml_format"
    target_xml_path = "yolo_detection_xml_format/cam2_xml_format"
    output_xml_path = "updated_xml"

    batch_update_person(source_xml_path, target_xml_path, output_xml_path)
