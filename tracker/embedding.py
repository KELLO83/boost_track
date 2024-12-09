from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import numpy as np
import torchvision.transforms as T
#from .TransReID.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_BASE
#from .TransReID_SSL.transreid_pytorch.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_EXTEND # TransReID SSL VIT_BASE 16
#from .TransReID_SSL.transreid_pytorch.model.make_model import swin_base_patch4_window7_224 as SWIN_TRANSFORMER
from .TransReID_SSL.transreid_pytorch.model.backbones.Microsoft_swinv2_trasformer import SwinTransformerV2 as MS_Swin_Transformer_V2
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
        self.config = config
        self.model_type = 'swinv2' if config.SSL_VIT else 'vit'
        self.crop_size = (192, 192)
        self.max_batch = 8
        self.device = torch.device('cuda') 
        
        # 전처리 파이프라인 설정 -
        self.transform = T.Compose([
            T.ToPILImage(),  # numpy array를 PIL Image로 변환
            T.Resize(self.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.grid_off = True
        
        self.initialize_model()

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
        """이미지에서 검출된 객체의 임베딩을 계산합니다."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
            
        if len(bbox) == 0:
            return np.array([])
            
        # 캐시 확인
        if tag != self.cache_name:
            self.cache = {}
            self.cache_name = tag
            
        # 배치 처리를 위한 준비
        batch_size = min(self.max_batch, len(bbox))
        n_batches = math.ceil(len(bbox) / batch_size)
        embeddings = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(bbox))
            batch_bbox = bbox[start_idx:end_idx]
            
            # 배치 내의 각 객체에 대한 임베딩 계산
            batch_tensors = []
            for box in batch_bbox:
                x1, y1, x2, y2 = map(int, box)
                box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"
                
                if box_key in self.cache:
                    embedding = self.cache[box_key]
                else:
                    # 이미지 크롭 및 전처리
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue
                        
                    tensor = self.transform(cropped)
                    batch_tensors.append(tensor)
                    
            if not batch_tensors:
                continue
                
            # 배치 텐서 생성 및 GPU로 이동
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            # 임베딩 계산
            with torch.no_grad():
                batch_embeddings = self.model(batch_input)
                
            # 정규화 및 캐시 저장
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            for j, box in enumerate(batch_bbox):
                if j < len(batch_embeddings):
                    x1, y1, x2, y2 = map(int, box)
                    box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"
                    self.cache[box_key] = batch_embeddings[j]
                    embeddings.append(batch_embeddings[j])
        
        return np.array(embeddings) if embeddings else np.array([])

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
        if self.model_type == 'swinv2':
            print("Swin Transformer V2 model loading...")
            
            # Swin Transformer V2 설정값
            init_args = {
                'img_size': 192,
                'patch_size': 4,
                'embed_dim': 192,
                'depths': [2, 2, 18, 2],
                'num_heads': [6, 12, 24, 48],
                'window_size': 12,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.2,
                'patch_norm': True,
                'use_checkpoint': False,
                'pretrained_window_sizes': [12, 12, 12, 12]
            }

            print("Model initialization arguments:", init_args)
            
            # 모델 초기화
            self.model = MS_Swin_Transformer_V2(**init_args)
            
            self.model.to(self.device)
            
            # 분류 헤드 제거
            if hasattr(self.model, 'head'):
                self.model.head = torch.nn.Identity()
            
            print(f"Model initialized on {self.device}")
            
            # 추론 모드로 설정
            self.model.eval()
            
        else:
            print("TransReID ViT model loading...")
            self.model = VIT_BASE(
                img_size = self.crop_size,     # input image size
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

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)
