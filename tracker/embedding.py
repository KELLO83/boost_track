from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import numpy as np
import torchvision.transforms as T
from .TransReID.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch.nn.functional as F

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
        self.crop_size = (256, 128)  # height, width 순서
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
            
            # BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # TransReID 전처리
            crop = self.transform(crop)
            crops.append(crop)

        if not crops:  # 유효한 크롭이 없는 경우
            return np.array([])

        crops = torch.stack(crops)
        crops = crops.to(self.device)
        
        # 디버깅 정보 출력
        print(f"Input tensor shape: {crops.shape}")
        
        # 배치 처리
        embs = []
        with torch.no_grad():
            for i in range(0, len(crops), self.max_batch):
                batch = crops[i:i + self.max_batch]
                try:
                    # camera_label과 view_label 생성
                    batch_size = batch.shape[0]
                    camera_label = torch.zeros(batch_size).long().to(self.device)  # 모든 이미지를 camera 0으로 설정
                    view_label = torch.zeros(batch_size).long().to(self.device)    # 모든 이미지를 view 0으로 설정
                    
                    feat = self.model(batch, cam_label=camera_label, view_label=view_label)
                    if isinstance(feat, tuple):
                        feat = feat[0]  # global feature만 사용
                    embs.append(feat.cpu())
                except RuntimeError as e:
                    print(f"Error processing batch: {e}")
                    print(f"Batch shape: {batch.shape}")
                    continue

        if not embs:  # 모든 배치가 실패한 경우
            print("Warning: All batches failed to process")
            return np.array([])

        embs = torch.cat(embs, dim=0)
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
        # numpy array를 torch tensor로 변환
        query_emb = torch.from_numpy(query_emb)
        gallery_embs = torch.from_numpy(gallery_embs)
        
        # L2 정규화 적용
        query_emb = F.normalize(query_emb, p=2, dim=1)
        gallery_embs = F.normalize(gallery_embs, p=2, dim=1)
        
        # 코사인 유사도 계산 (1 - cosine_similarity)
        # cosine_similarity는 -1 ~ 1 사이의 값을 가지므로
        # 1 - cosine_similarity는 0 ~ 2 사이의 값을 가짐
        # 이를 0 ~ 1 사이로 정규화
        similarities = torch.mm(query_emb, gallery_embs.t())
        distances = (1 - similarities) / 2.0  # 0 ~ 1 사이로 정규화
        
        # numpy로 변환
        distances = distances.numpy()
        
        # top-k 유사도 및 인덱스 반환
        indices = np.argsort(distances, axis=1)[:, :k]
        sorted_distances = np.take_along_axis(distances, indices, axis=1)
        
        # 디버깅을 위한 출력
        print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
        
        return sorted_distances, indices

    def initialize_model(self):
        print("TransReID ViT model loading...")
        cfg = TransReIDConfig()
        
        # Market1501 데이터셋 기준 설정
        num_class = 751  # Market1501 클래스 수
        camera_num = 6   # Market1501 카메라 수
        view_num = 0     # 시점 정보 미사용
        
        model = vit_base_patch16_224_TransReID(
            img_size=(256, 128),  # height x width
            stride_size=16,       # stride_size
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            camera=camera_num,    # camera label 수
            view=view_num,        # view label 수
            sie_xishu=cfg.MODEL.SIE_COE,  # SIE coefficient
            local_feature=False,  # global feature만 사용
            num_classes=num_class # 클래스 수
        )

        # 사전 학습된 가중치 로드
        checkpoint = torch.load(self.config.reid_model_path)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # state_dict 키 이름 정리
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # module. 제거
            if k.startswith('backbone.'):
                k = k[9:]  # backbone. 제거
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        model.cuda()
        self.model = model

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)
