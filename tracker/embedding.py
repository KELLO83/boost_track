from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import numpy as np
import torchvision.transforms as T
from .TransReID.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_BASE
from .TransReID_SSL.transreid_pytorch.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_EXTEND # TransReID SSL VIT_BASE 16
from .TransReID_SSL.transreid_pytorch.model.make_model import swin_base_patch4_window7_224 as SWIN_TRANSFORMER
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch.nn.functional as F
import torchinfo
import math

@dataclass
class TransReIDConfig:
    """TransReID 모델 설정"""
    class ModelConfig:
        NAME: str = 'transformer'
        TRANSFORMER_TYPE: str = 'vit_base_patch16_224_TransReID'
        STRIDE_SIZE: List[int] = [12, 12]  # TransReID 논문 권장값
        DROP_PATH: float = 0.1
        DROP_OUT: float = 0.0
        ATT_DROP_RATE: float = 0.0
        
        # Side Information Embedding (SIE)
        SIE_COE: float = 3.0
        SIE_CAMERA: bool = True  # camera-aware
        SIE_VIEW: bool = False
        
        # Jigsaw Patch Module (JPM)
        JPM: bool = True
        
        # Pretrain settings
        PRETRAIN_CHOICE: str = 'imagenet'
        PRETRAIN_PATH: str = ''
        
    class InputConfig:
        SIZE_TRAIN: List[int] = [256, 128]  # Market1501 이미지 크기
        SIZE_TEST: List[int] = [256, 128]
        
    def __init__(self):
        self.MODEL = self.ModelConfig()
        self.INPUT = self.InputConfig()

"""
TransReID를 사용한 임베딩 계산
"""

class EmbeddingComputer:
    def __init__(self, config):
        self.model = None
        self.device = config.device
        self.dataset = config.dataset
        self.test_dataset = True
        self.config = config
        
        if config.SSL_VIT:
            self.crop_size = (384, 384)  # SWIN_TRASFORMER
        else:
            self.crop_size = (256, 128)  # TransReID*(ViT) 
            
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.grid_off = True
        self.max_batch = 8
        
        # TransReID 전처리 파이프라인
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def visualize_attention(self, feat, img):
        try:
            print('img shape: ', img.shape)
            print('feat shape: ', feat.shape)
            
            try:
                B, N, D = feat.shape # [BATCH , TOKEN , EMBEDDING_DIM]
            except Exception as e:
                raise Exception("Error in visualize_attention: Invalid input shape")

            # 동적 패치 수 계산
            num_patches = N - 1  # CLS 토큰 제외
            
            # 이미지 크기 및 비율 고려한 패치 그리드 계산
            img_h, img_w = img.shape[:2]
            aspect_ratio = img_w / img_h
            
            # 패치 그리드 계산 개선 (더 유연한 알고리즘)
            num_patches_h = max(int(math.sqrt(num_patches / aspect_ratio)), 1)
            num_patches_w = max(int(num_patches / num_patches_h), 1)
            
            # 실제 패치 수와 계산된 패치 수 조정
            total_calculated_patches = num_patches_h * num_patches_w
            if total_calculated_patches > num_patches:
                num_patches_w = num_patches // num_patches_h
            
            print(f'Dynamic patch calculation: {num_patches_h} x {num_patches_w} = {num_patches_h * num_patches_w} patches')
            print(f'Original token count: {N}, Patch tokens: {num_patches}')
            
            target_tokens = [i for i in range(N) if i % 50 == 0]
                        
            with torch.no_grad():
                # 마지막 어텐션 레이어 선택
                attn_layer = self.model.blocks[-1].attn
                x = feat.to(self.device)
                
                # QKV 계산
                qkv = attn_layer.qkv(x)
                qkv = qkv.reshape(B, N, 3, attn_layer.num_heads, D // attn_layer.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # 어텐션 가중치 계산
                attn = (q @ k.transpose(-2, -1)) * attn_layer.scale
                attn = attn.softmax(dim=-1)
                attn_weights = attn.mean(1)
            
            H, W = img.shape[:2]
            visualizations = []
            
            for i in target_tokens:
                if i >= N:
                    continue
                
                token_name = "CLS" if i == 0 else f"Patch_{i}"
                
                try:
                    # 패치 어텐션 맵 추출 (CLS 토큰 제외)
                    attn_map = attn_weights[0, i, 1:].cpu().numpy()
                    
                    # 1D 어텐션 맵을 2D로 재구성 (안전한 인덱싱)
                    attn_map_2d = np.zeros((num_patches_h, num_patches_w))
                    for idx, val in enumerate(attn_map[:num_patches_h * num_patches_w]):
                        row = idx // num_patches_w
                        col = idx % num_patches_w
                        attn_map_2d[row, col] = val
                    
                    # 이미지 크기로 리사이즈
                    attn_map_resized = cv2.resize(attn_map_2d, (W, H), interpolation=cv2.INTER_LINEAR)
                    
                    # 정규화
                    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)
                    
                    # 히트맵 생성
                    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img.copy(), 0.6, heatmap, 0.4, 0)
                    
                    # 패치 위치 표시 (CLS 토큰 제외)
                    if i > 0:
                        patch_idx = i - 1  # CLS 토큰을 제외한 실제 패치 인덱스
                        
                        # 패치의 행과 열 계산
                        row = patch_idx // num_patches_w
                        col = patch_idx % num_patches_w
                        
                        # 원본 이미지 크기에 맞게 패치 위치 조정
                        patch_h = H / num_patches_h
                        patch_w = W / num_patches_w
                        
                        patch_y = int(row * patch_h)
                        patch_x = int(col * patch_w)
                        patch_h = int(patch_h)
                        patch_w = int(patch_w)
                        
                        # 패치 영역 표시
                        cv2.rectangle(overlay, 
                                    (patch_x, patch_y),
                                    (patch_x + patch_w, patch_y + patch_h),
                                    (0, 255, 0), 2)
                    
                    cv2.namedWindow(f'Attention Map - {token_name}', cv2.WINDOW_NORMAL)
                    cv2.imshow(f'Attention Map - {token_name}', overlay)
                    visualizations.append((token_name, overlay))
                
                except Exception as e:
                    print(f"Error processing token {i}: {e}")
                    continue
            
            key = cv2.waitKey(0)
            return visualizations
            
        except Exception as e:
            print(f"Error in visualize_attention: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def compute_embedding(self, img, bbox, tag):
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        # 캐시 키 생성 시 bbox 정보도 포함
        cache_key = f"{tag}_{hash(str(bbox))}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.model is None:
            self.initialize_model()

        # 이미지 크롭 및 전처리
        crops = []
        original_crops = []  # 원본 크롭 이미지 저장
        h, w = img.shape[:2]
        results = np.round(bbox).astype(np.int32)
        results[:, 0] = results[:, 0].clip(0, w)
        results[:, 1] = results[:, 1].clip(0, h)
        results[:, 2] = results[:, 2].clip(0, w)
        results[:, 3] = results[:, 3].clip(0, h)

        for p in results:
            crop = img[p[1]:p[3], p[0]:p[2]]
            if crop.size == 0:  # 빈 크롭 체크
                continue
            if crop.shape[0] < 2 or crop.shape[1] < 2:  # 최소 크기 확인
                continue
            
            original_crops.append(crop.copy())  # 원본 크롭 저장
            
            # BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # TransReID 전처리
            crop = self.transform(crop) # resize + Normalize
            crops.append(crop)

        if not crops:  # 유효한 크롭이 없는 경우
            return np.array([])

        crops = torch.stack(crops)
        crops = crops.to(self.device)
                
        # 배치 처리
        embs = []
        with torch.no_grad():
            for i in range(0, len(crops), self.max_batch):
                batch = crops[i:i + self.max_batch]
                try:
                    # camera_label과 view_label 생성
                    batch_size = batch.shape[0]
                    camera_label = torch.zeros(batch_size).long().to(self.device)
                    view_label = torch.zeros(batch_size).long().to(self.device)
                    
                    # forward pass로 특징 추출
                    feat = self.model(batch, cam_label=camera_label, view_label=view_label)
                    print("forward result: ", feat.shape)
                    
                    # Swin Transformer는 이미 [batch_size, 1024] 형태로 출력됨
                    # 추가 처리 없이 바로 사용
                    embs.append(feat.cpu())
                    
                except RuntimeError as e:
                    print(f"Error processing batch: {e}")
                    print(f"Batch shape: {batch.shape}")
                    continue

        if not embs:  # 모든 배치가 실패한 경우
            print("Warning: All batches failed to process")
            return np.array([])

        # 배치 결합 및 정규화
        embs = torch.cat(embs, dim=0)  # [N, 1024] 형태로 결합
        embs = F.normalize(embs, p=2, dim=1)  # L2 정규화
        embs = embs.numpy()
        
        # 임베딩 norm 체크
        norms = np.linalg.norm(embs, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5):
            print(f"Warning: Embeddings not properly normalized. Norms: {norms}")
        
        self.cache[cache_key] = embs
        return embs

    def compute_similarity(self, query_emb, gallery_embs, k=1):
        """
        두 임베딩 간의 유사도를 계산합니다.
        
        Args:
            query_emb: 쿼리 임베딩 (N x D)
            gallery_embs: 갤러리 임베딩 (M x D)
            k: top-k 유사도를 반환
        
        Returns:
            distances: 유사도 점수 (0~1 사이의 값)
            indices: 가장 유사한 갤러리 인덱스
        """

        query_emb = torch.from_numpy(query_emb)
        gallery_embs = torch.from_numpy(gallery_embs)

        query_emb = F.normalize(query_emb, p=2, dim=1)
        gallery_embs = F.normalize(gallery_embs, p=2, dim=1)
        

        similarities = torch.mm(query_emb, gallery_embs.t())
        distances = (1 - similarities) / 2.0  # 0 ~ 1 사이로 정규화
        
        distances = distances.numpy()
        
        indices = np.argsort(distances, axis=1)[:, :k]
        sorted_distances = np.take_along_axis(distances, indices, axis=1)
        

        print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
        
        return sorted_distances, indices

    def initialize_model(self):
        import torch.nn as nn
        if self.config.SSL_VIT: 
            print("SSL Swin Transformer model loading...")

            # 이미지 크기를 정수로 변환하여 전달
            img_size = max(self.crop_size)
            
            self.model = SWIN_TRANSFORMER(
                img_size=img_size,     # 이미지 크기 (정수)
                drop_rate=0.0,         # 드롭아웃 비율
                attn_drop_rate=0.0,    # 어텐션 드롭아웃 비율
                drop_path_rate=0.1,    # 드롭 패스 비율
                camera_num=0,          # 카메라 수
                view_num=0,            # 뷰 수
                num_classes=1,    
                patch_norm=True,
                qkv_bias=True,
            )
            
            #print(self.model)
            # 분류 헤드 제거 (특징 추출용)
            if hasattr(self.model, 'head'):
                self.model.head = nn.Identity()
            
            # if hasattr(self.model , 'avgpool'):
            #     self.model.avgpool = nn.Identity()
                
            print(self.model)
            
            print(f"Swin Transformer initialized with image size: {self.crop_size}")
        
        
        
        else:
            print("TransReID ViT model loading...")
            self.model = VIT_BASE(
                img_size = self.img_size,     # input image size
                stride_size=16,          # patch (feature) extraction stride
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                camera=0,                # SIE 완전 비활성화
                view=0,                  # SIE 완전 비활성화
                sie_xishu=0.0,          # SIE 비율
                local_feature=True,     # local feature extraction
                num_classes=1           # number of classification classes
            )
        
        checkpoint = torch.load(self.config.reid_model_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('backbone.'):
                k = k[9:]
            new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False) # strict 매칭된 가중치만 로딩
        self.model.eval()
        self.model.cuda()

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)
