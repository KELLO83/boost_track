import torch
from tracker.CLIP.model.clip.model import RGBEncodedCLIP

# CLIP_Reid Base
embed_dim = 768                  
image_resolution = 232          # 14.5x16≈232
h_resolution = 14              # 패치 개수로 직접 지정
w_resolution = 15              # 패치 개수로 직접 지정 (14x15=210)
vision_layers = 12             
vision_width = 768             # Base 모델 width
vision_patch_size = 16         
vision_stride_size = 16        
context_length = 77            
vocab_size = 49408            
transformer_width = 512        
transformer_heads = 8          
transformer_layers = 12        

# RGBEncodedCLIP 모델 생성
model = RGBEncodedCLIP(
    embed_dim=embed_dim,
    image_resolution=image_resolution,
    vision_layers=vision_layers,
    vision_width=vision_width,
    vision_patch_size=vision_patch_size,
    vision_stride_size=vision_stride_size,
    context_length=context_length,
    vocab_size=vocab_size,
    transformer_width=transformer_width,
    transformer_heads=transformer_heads,
    transformer_layers=transformer_layers,
    h_resolution=h_resolution,
    w_resolution=w_resolution
)

# 사전 학습된 가중치 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

weight_path = 'external/weights/CLIPReID_MSMT17_clipreid_12x12sie_ViT-B-16_60.pth'
model.load_state_dict(torch.load(weight_path), strict=False)
print("Model loaded successfully!")

# 샘플 입력 생성
input_dummy = torch.randn(1, 3, 224, 240).to(device)  # 이미지 입력

# RGB 통계 샘플 생성 (mean과 std 각각 3차원)
rgb_mean = torch.tensor([[0.5, 0.4, 0.3]]).to(device)  # 예시 RGB 평균값
rgb_std = torch.tensor([[0.2, 0.2, 0.2]]).to(device)   # 예시 RGB 표준편차
rgb_stats = torch.cat([rgb_mean, rgb_std], dim=1)      # [1, 6] 형태로 결합

# 모델 추론
with torch.no_grad():
    # 이미지와 RGB 통계로 유사도 계산
    similarity = model(input_dummy, rgb_stats)
    print("Similarity Shape:", similarity.shape)


print(model)


# # Base 모델 입력 크기
# input_dummy = torch.randn(1, 3, 224, 240).to(device)  # h=224(14패치), w=240(15패치)

# # 모델 추론
# with torch.no_grad():
#     # 이미지 특징만 추출
#     image_features = model.encode_image(input_dummy)  # text 없이 이미지 인코딩만 수행
#     print("Image features value:", image_features[-1])  # [1, 768] for Large model
#     print("Image features shape:", image_features[-1].shape)
    
    
#     for i in image_features:
#         print(i.shape)