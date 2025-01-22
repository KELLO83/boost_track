import os
import shutil
from glob import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET

target_path = 'updated_xml'
output_path = 'updated_xml_no_duplicates'

def get_frame_number(filename):
    """파일 이름에서 프레임 번호를 추출"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(base_name)
    except ValueError:
        return None

def create_output_directory(output_path):
    """출력 디렉토리 생성"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")

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
    
    return sorted(bboxes)  # 일관된 비교를 위해 정렬

def is_duplicate_frame(curr_img_path, curr_xml_path, prev_img_path, prev_xml_path):
    """두 프레임이 중복인지 확인"""
    # 이미지 비교
    curr_img = cv2.imread(curr_img_path)
    prev_img = cv2.imread(prev_img_path)
    
    if curr_img is None or prev_img is None:
        return False
    
    # 이미지 크기가 다르면 중복이 아님
    if curr_img.shape != prev_img.shape:
        return False
    
    # 1. 이미지 유사도 검사 (여러 방법 조합)
    # MSE 계산
    mse = np.mean((curr_img - prev_img) ** 2)
    
    # 절대 차이 계산
    diff = cv2.absdiff(curr_img, prev_img)
    diff_mean = np.mean(diff)
    
    # 2. XML bounding box 비교
    curr_bboxes = get_bboxes_from_xml(curr_xml_path)
    prev_bboxes = get_bboxes_from_xml(prev_xml_path)
    
    if len(curr_bboxes) != len(prev_bboxes):
        return False
        
    # bbox 좌표 차이 계산
    def calculate_bbox_difference(box1, box2):
        name1, x1_1, y1_1, x2_1, y2_1 = box1
        name2, x1_2, y1_2, x2_2, y2_2 = box2
        if name1 != name2:
            return float('inf')
        coord_diff = abs(x1_1 - x1_2) + abs(y1_1 - y1_2) + abs(x2_1 - x2_2) + abs(y2_1 - y2_2)
        return coord_diff
    
    bbox_diffs = [calculate_bbox_difference(b1, b2) 
                 for b1, b2 in zip(curr_bboxes, prev_bboxes)]
    
    max_bbox_diff = max(bbox_diffs) if bbox_diffs else float('inf')
    
    # 3. 종합적인 중복 판단
    # 임계값 설정 (더 엄격한 기준 적용)
    MSE_THRESHOLD = 100.0  # MSE 임계값
    DIFF_THRESHOLD = 10.0  # 절대 차이 임계값
    BBOX_DIFF_THRESHOLD = 5.0  # bbox 좌표 차이 임계값
    
    # 모든 조건을 만족해야 중복으로 판단
    is_duplicate = (
        mse < MSE_THRESHOLD and
        diff_mean < DIFF_THRESHOLD and
        max_bbox_diff < BBOX_DIFF_THRESHOLD
    )
    
    return is_duplicate

def is_similar_images(img1_path, img2_path, threshold=0.98):
    """두 이미지의 픽셀 유사도를 비교"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return False
    
    # 이미지 크기가 다르면 중복이 아님
    if img1.shape != img2.shape:
        return False
    
    # 이미지를 그레이스케일로 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 구조적 유사도 계산 (SSIM)
    score = cv2.matchTemplate(gray1, gray2, cv2.TM_CCORR_NORMED)[0][0]
    
    return score > threshold

def remove_duplicates(target_path, output_path):
    """중복된 프레임 제거 및 파일 복사"""
    # 출력 디렉토리 생성
    create_output_directory(output_path)
    
    # jpg와 xml 파일 목록 가져오기
    jpg_files = glob(os.path.join(target_path, '*.jpg'))
    xml_files = glob(os.path.join(target_path, '*.xml'))
    
    # 프레임 번호별로 파일 정리
    frame_files = {}
    for jpg_file in jpg_files:
        frame_num = get_frame_number(jpg_file)
        if frame_num is not None:
            xml_file = os.path.join(target_path, f"{frame_num}.xml")
            if xml_file in xml_files:
                frame_files[frame_num] = (jpg_file, xml_file)
    
    # 프레임 번호로 정렬
    sorted_frames = sorted(frame_files.keys())
    
    # 중복 프레임 체크 및 파일 복사
    prev_frame = None
    prev_jpg = None
    prev_xml = None
    copied_count = 0
    skipped_count = 0
    
    for frame in sorted_frames:
        curr_jpg, curr_xml = frame_files[frame]
        
        # 첫 프레임이거나 중복이 아닌 경우에만 복사
        if prev_frame is None or not is_duplicate_frame(curr_jpg, curr_xml, prev_jpg, prev_xml):
            # 파일 복사
            jpg_dest = os.path.join(output_path, os.path.basename(curr_jpg))
            xml_dest = os.path.join(output_path, os.path.basename(curr_xml))
            
            shutil.copy2(curr_jpg, jpg_dest)
            shutil.copy2(curr_xml, xml_dest)
            
            copied_count += 1
            prev_frame = frame
            prev_jpg = curr_jpg
            prev_xml = curr_xml
        else:
            skipped_count += 1
    
    print(f"Processed files:")
    print(f"- Copied: {copied_count} frames")
    print(f"- Skipped duplicates: {skipped_count} frames")
    print(f"Files have been copied to: {output_path}")

def remove_duplicates_by_pixels(input_path, output_path):
    """이미지 픽셀 비교를 통한 중복 제거"""
    # 출력 디렉토리 생성
    create_output_directory(output_path)
    
    # jpg와 xml 파일 목록 가져오기
    jpg_files = sorted(glob(os.path.join(input_path, '*.jpg')))
    xml_files = {os.path.splitext(os.path.basename(f))[0]: f 
                 for f in glob(os.path.join(input_path, '*.xml'))}
    
    # 중복 체크 및 파일 복사
    prev_img = None
    copied_count = 0
    skipped_count = 0
    
    for curr_jpg in jpg_files:
        base_name = os.path.splitext(os.path.basename(curr_jpg))[0]
        curr_xml = xml_files.get(base_name)
        
        if curr_xml is None:
            continue
            
        # 첫 이미지이거나 이전 이미지와 유사하지 않은 경우에만 복사
        if prev_img is None or not is_similar_images(curr_jpg, prev_img):
            # 파일 복사
            jpg_dest = os.path.join(output_path, os.path.basename(curr_jpg))
            xml_dest = os.path.join(output_path, os.path.basename(curr_xml))
            
            shutil.copy2(curr_jpg, jpg_dest)
            shutil.copy2(curr_xml, xml_dest)
            
            copied_count += 1
            prev_img = curr_jpg
        else:
            skipped_count += 1
    
    print(f"Processed files using pixel comparison:")
    print(f"- Copied: {copied_count} frames")
    print(f"- Skipped duplicates: {skipped_count} frames")
    print(f"Files have been copied to: {output_path}")

if __name__ == "__main__":
    target_path = 'updated_xml'
    output_path = 'updated_xml_no_duplicates'
    output2_path = 'updated_xml_no_duplicates2'
    remove_duplicates(target_path, output_path)
    
    remove_duplicates_by_pixels(output_path, output2_path)
