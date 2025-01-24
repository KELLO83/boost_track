import os
import cv2
import glob
import random


# 경로 설정
image_dir = "cam0/images/"
label_dir = "cam0/labels/"
output_video = "output.avi"

# 이미지와 라벨 파일 읽기

def sort_key(file_path):
    # 숫자를 기준으로 파일 이름 정렬
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return int(base_name)  # 파일 이름이 숫자로 되어 있다고 가정


image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")), key=sort_key)
label_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(label_dir, "*.txt"))}
# 출력 동영상 설정
frame_width, frame_height = 1920, 1080  # 해상도 설정 (이미지 해상도에 맞게 수정)
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# 색상 설정 (랜덤 색상)
object_colors = {}
def get_color(object_id):
    if object_id not in object_colors:
        object_colors[object_id] = [random.randint(0, 255) for _ in range(3)]
    return object_colors[object_id]


# 이미지 처리
def draw_boxes(image, label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    height, width, _ = image.shape
    for line in lines:
        parts = line.strip().split()
        object_id = int(parts[0])  # 객체 ID
        x_center, y_center, w, h = map(float, parts[1:])

        # YOLO 좌표 -> 픽셀 좌표 변환
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        # 박스와 객체 ID 표시
        color = get_color(object_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"ID: {object_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 모든 이미지 처리 및 영상 저장
for image_file in image_files:
    base_name = os.path.splitext(os.path.basename(image_file))[0]

    # 이미지 읽기
    img = cv2.imread(image_file)
    
    if base_name in label_files:
        # 라벨 파일 경로
        label_file = label_files[base_name]

        # 바운딩 박스 그리기
        draw_boxes(img, label_file)
    else:
        print(f"라벨이 없는 이미지 건너뛰기: {image_file}")

    # 프레임 크기 조정 (필요시 생략 가능)
    img_resized = cv2.resize(img, (frame_width, frame_height))

    # 동영상에 추가
    video_writer.write(img_resized)

# 자원 해제
video_writer.release()
print(f"동영상 저장 완료: {output_video}")
