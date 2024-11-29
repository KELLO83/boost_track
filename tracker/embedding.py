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
    def __init__(self, dataset, test_dataset=True, grid_off=True, max_batch=1024, reid_model_path=None):
        self.model = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (128, 256)  # width x height
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.grid_off = grid_off
        self.max_batch = max_batch
        self.reid_model_path = reid_model_path
        
        # TransReID normalize 값 사용
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def compute_embedding(self, img, bbox, tag):
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: The number of cached embeddings don't match the "
                    "number of detections.\nWas the detector model changed? Delete cache if so."
                )
            return embs

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
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
            crop = torch.from_numpy(crop).float()
            crop = crop.permute(2, 0, 1) / 255.0
            crop = self.normalize(crop)
            crops.append(crop)
            cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
            cv2.imshow("crop", crop.permute(1, 2, 0).numpy())
            print(crop.shape)
            cv2.waitKey(0)
            
        crops = torch.stack(crops)

        # TransReID를 통한 임베딩 추출
        embs = []
        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx:idx + self.max_batch].cuda()
            with torch.no_grad():
                feat = self.model(batch_crops)
                # L2 정규화
                feat = torch.nn.functional.normalize(feat, dim=1)
            embs.append(feat.cpu())
            
        embs = torch.cat(embs, dim=0)
        embs = embs.numpy()

        self.cache[tag] = embs
        return embs

    def compute_similarity(self, query_emb, gallery_embs, k=1):
        """
        두 임베딩 간의 유사도를 계산합니다.
        
        Args:
            query_emb: 쿼리 임베딩 (N x D)
            gallery_embs: 갤러리 임베딩 (M x D)
            k: top-k 유사도를 반환
        
        Returns:
            distances: 유사도 점수
            indices: 가장 유사한 갤러리 인덱스
        """
        # 코사인 유사도 계산
        query_emb = torch.from_numpy(query_emb)
        gallery_embs = torch.from_numpy(gallery_embs)
        
        # 정규화
        query_emb = torch.nn.functional.normalize(query_emb, dim=1)
        gallery_embs = torch.nn.functional.normalize(gallery_embs, dim=1)
        
        # 유사도 계산 (1 - cosine_similarity)
        distances = 1 - torch.mm(query_emb, gallery_embs.t())
        distances = distances.numpy()
    
        # top-k 유사도 및 인덱스 반환
        indices = np.argsort(distances, axis=1)[:, :k]
        sorted_distances = np.take_along_axis(distances, indices, axis=1)
        
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
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            camera=camera_num if cfg.MODEL.SIE_CAMERA else 0,
            view=view_num if cfg.MODEL.SIE_VIEW else 0,
            sie_xishu=cfg.MODEL.SIE_COE,
            local_feature=cfg.MODEL.JPM
        )

        # 사전 학습된 가중치 로드
        checkpoint = torch.load(self.reid_model_path)
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
