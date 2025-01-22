import os
import cv2
import numpy as np
from xml_duplicate_delete import is_duplicate_frame, get_bboxes_from_xml
from skimage.metrics import structural_similarity as ssim

def test_image_similarity(img1_path, img2_path):
    """이미지 유사도를 다양한 메트릭으로 테스트"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error reading images: {img1_path} or {img2_path}")
        return
    
    # MSE 계산
    mse = np.mean((img1 - img2) ** 2)
    
    # 절대 차이 계산
    diff = cv2.absdiff(img1, img2)
    diff_mean = np.mean(diff)
    
    # SSIM 계산
    ssim_score = ssim(img1, img2, multichannel=True)
    
    print(f"\nImage Similarity Metrics between {os.path.basename(img1_path)} and {os.path.basename(img2_path)}:")
    print(f"MSE: {mse:.2f}")
    print(f"Mean Absolute Difference: {diff_mean:.2f}")
    print(f"SSIM Score: {ssim_score:.4f}")
    
    # 차이를 시각화하여 저장
    diff_color = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=10), cv2.COLORMAP_JET)
    output_path = "difference_visualization"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(os.path.join(output_path, f"diff_{os.path.basename(img1_path)}"), diff_color)

def test_bbox_similarity(xml1_path, xml2_path):
    """XML 파일의 bounding box 유사도 테스트"""
    bboxes1 = get_bboxes_from_xml(xml1_path)
    bboxes2 = get_bboxes_from_xml(xml2_path)
    
    print(f"\nBounding Box Comparison between {os.path.basename(xml1_path)} and {os.path.basename(xml2_path)}:")
    print(f"Number of boxes in first XML: {len(bboxes1)}")
    print(f"Number of boxes in second XML: {len(bboxes2)}")
    
    if len(bboxes1) != len(bboxes2):
        print("Different number of bounding boxes - not duplicates")
        return False
    
    max_diff = 0
    for box1, box2 in zip(bboxes1, bboxes2):
        name1, x1_1, y1_1, x2_1, y2_1 = box1
        name2, x1_2, y1_2, x2_2, y2_2 = box2
        
        if name1 != name2:
            print(f"Different object names: {name1} vs {name2}")
            return False
        
        coord_diff = abs(x1_1 - x1_2) + abs(y1_1 - y1_2) + abs(x2_1 - x2_2) + abs(y2_1 - y2_2)
        max_diff = max(max_diff, coord_diff)
        
        print(f"Object {name1}:")
        print(f"  Coordinate difference: {coord_diff:.2f}")
    
    print(f"Maximum coordinate difference: {max_diff:.2f}")
    return max_diff

def test_duplicate_detection(target_path):
    """특정 디렉토리의 연속된 프레임들에 대해 중복 검사 수행"""
    jpg_files = sorted([f for f in os.listdir(target_path) if f.endswith('.jpg')])
    
    if len(jpg_files) < 2:
        print("Not enough files for testing")
        return
    
    print("Testing duplicate detection on consecutive frames...")
    for i in range(len(jpg_files) - 1):
        curr_jpg = os.path.join(target_path, jpg_files[i])
        next_jpg = os.path.join(target_path, jpg_files[i + 1])
        
        curr_xml = curr_jpg.replace('.jpg', '.xml')
        next_xml = next_jpg.replace('.jpg', '.xml')
        
        if not (os.path.exists(curr_xml) and os.path.exists(next_xml)):
            continue
            
        print(f"\nTesting frames {jpg_files[i]} and {jpg_files[i + 1]}")
        
        # 이미지 유사도 테스트
        test_image_similarity(curr_jpg, next_jpg)
        
        # Bounding box 유사도 테스트
        test_bbox_similarity(curr_xml, next_xml)
        
        # 전체 중복 판단
        is_dup = is_duplicate_frame(curr_jpg, curr_xml, next_jpg, next_xml)
        print(f"\nFinal duplicate detection result: {'Duplicate' if is_dup else 'Not duplicate'}")
        print("-" * 80)

if __name__ == "__main__":
    target_path = "updated_xml"  # 테스트할 디렉토리 경로
    
    if not os.path.exists(target_path):
        print(f"Directory {target_path} does not exist!")
    else:
        test_duplicate_detection(target_path)
        print("\nTest completed. Check 'difference_visualization' directory for visual results.")
