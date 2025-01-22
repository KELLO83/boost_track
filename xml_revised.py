import os
import shutil
import xml.etree.ElementTree as ET
from glob import glob

def backup_xml_files(src_dir: str, backup_dir: str):
    """
    XML 파일들을 백업 디렉토리에 복사
    
    Args:
        src_dir: 원본 XML 파일이 있는 디렉토리
        backup_dir: 백업할 디렉토리
    """
    # 백업 디렉토리 생성
    os.makedirs(backup_dir, exist_ok=True)
    
    # XML 파일 목록 가져오기
    xml_files = glob(os.path.join(src_dir, '*.xml'))
    
    # 파일 이름에서 숫자만 추출하여 정렬
    def get_number(filepath):
        basename = os.path.splitext(os.path.basename(filepath))[0]
        # 백슬래시와 기타 문자 제거하고 숫자만 추출
        number = ''.join(c for c in basename if c.isdigit())
        try:
            return int(number)
        except ValueError:
            return float('inf')  # 숫자가 아닌 경우 마지막으로 정렬
    
    xml_files.sort(key=get_number)
    
    # 각 파일을 백업 디렉토리에 복사
    for i, xml_file in enumerate(xml_files, 1):
        backup_name = f"{i}.xml"
        backup_path = os.path.join(backup_dir, backup_name)
        shutil.copy2(xml_file, backup_path)
        print(f"Backed up {xml_file} to {backup_path}")

def modify_images_above_number(xml_dir: str, number_mappings: dict, min_image_number: int, max_image_number: int):
    """
    특정 번호 범위 내의 모든 이미지 XML 파일에서 특정 name들을 변경
    
    Args:
        xml_dir: XML 파일이 있는 디렉토리 경로
        number_mappings: 변경할 번호 매핑 딕셔너리 {이전 번호: 새로운 번호}
        min_image_number: 이 번호 이상의 이미지부터 처리
        max_image_number: 이 번호 이하의 이미지까지 처리
    """
    # XML 파일 목록 가져오기
    xml_files = glob(os.path.join(xml_dir, '*.xml'))
    
    for xml_file in xml_files:
        # 파일 이름에서 번호 추출
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        try:
            image_number = int(base_name)
            # 지정된 범위 내의 파일만 처리
            if min_image_number <= image_number <= max_image_number:
                # XML 파일 파싱
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 변경 여부를 추적하는 플래그
                modified = False
                
                # 모든 object 태그를 순회하며 번호 변경
                for obj in root.findall('.//object'):
                    name = obj.find('name')
                    if name is not None and name.text in number_mappings:
                        name.text = str(number_mappings[name.text])
                        modified = True
                
                # 변경된 경우에만 파일 저장
                if modified:
                    tree.write(xml_file, encoding='utf-8', xml_declaration=True)
                    print(f"Modified {xml_file}")
        except ValueError:
            # 파일 이름이 숫자가 아닌 경우 건너뛰기
            continue


if __name__ == "__main__":
    
    backup_path = 'backup_xml'
    xml_path = 'updated_xml'  # XML 파일이 있는 디렉토리


    backup_xml_files(xml_path , backup_path)
    modify_images_above_number(xml_path, number_mappings, 565, 686)