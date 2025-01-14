import torch
from tracker.TransReID_SSL.transreid_pytorch.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_EXTEND

def test_vit_market():
    # 가중치 파일 구조 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('external/weights/VIT_SSL_MARKET_L.pth', map_location=device)
    
    print("Checkpoint structure:")
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # 기본 모델 생성 - 가중치 파일에 맞게 설정
    base = VIT_EXTEND(
        img_size = (384, 128),         # stem_conv=True일 때 448x448 입력으로 14x14 패치 생성
        stride_size=16,                # 기본 스트라이드
        drop_path_rate=0.1,
        camera=0,                      # SIE 비활성화
        view=0,                        # SIE 비활성화
        sie_xishu=0.0,                # SIE 완전 비활성화
        local_feature = False,            # 중간 feature map 사용
        gem_pool = True,              # Global average pooling 사용
        stem_conv = True               # Stem convolution 사용
    )
    
    print("model structure:")
    print(base)
    
    # 전체 모델 구조 생성
    class FullModel(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        
        def forward(self, x):
            return self.base(x)
        
        def forward_features(self, x, cam_label=None, view_label=None):
            return self.base.forward_features(x, cam_label, view_label)
    
    model = FullModel(base)
    
    print("\nModel configuration:")
    print(f"- Architecture: ViT-Base (768 dim, 12 heads)")
    print(f"- Input size: 384x128")
    print(f"- Patch size: 8x8 (14x14 patches)")
    print(f"- Local feature: Enabled (return intermediate features)")
    print(f"- SIE: Disabled")
    print(f"- STEM_CONV: Enabled")
    
    # Load weights
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    # Test with dummy input to check feature map shapes
    model.to(device)
    print("\nFeature map shapes:")
    B = 1  # batch size
    dummy_input = torch.randn(B, 3, 384, 128).to(device)  # 실제 입력 크기 사용
    with torch.no_grad():
        features = model.forward_features(dummy_input, None, None)
        if isinstance(features, tuple):
            for i, feat in enumerate(features):
                print(f"Feature {i} shape: {feat.shape}")
        else:
            print(f"Feature shape: {features.shape}")
            
            

if __name__ == "__main__":
    test_vit_market()
