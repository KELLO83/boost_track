from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import numpy as np
import torchvision.transforms as T
from .TransReID_SSL.transreid_pytorch.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_EXTEND
from .TransReID_SSL.transreid_pytorch.model.backbones.Microsoft_swinv2_trasformer import SwinTransformerV2 as MS_Swin_Transformer_V2
from .ConvNeXt.models.convnext import ConvNeXt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch.nn.functional as F
import torchinfo
import math
import torch


class EmbeddingComputer:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.model_type = config.Model_Name
        
        if config.Model_Name == 'dinov2':
            self.crop_size = (448, 448)  # 14의 배수인 448x448 사용 (14*32)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.crop_size), antialias=True),  
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.crop_size = (384, 384)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        print("Model Input size : ", self.crop_size)
            
        self.max_batch = 8
        self.device = torch.device('cuda') 
        self.initialize_model()
        
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
        print("Model type : ", self.model_type)

        if self.model_type == 'convNext':
            print("ConvNeXt model loading...")
            
            # ConvNeXt 모델 초기화 (Base 모델 기준)
            model = ConvNeXt(
                depths=[3, 3, 27, 3],           # 각 스테이지의 블록 수
                dims=[192, 384, 768, 1536],     # 각 스테이지의 채널 수
            )
            if hasattr(model, 'head'):
                model.head = torch.nn.Identity()
            
            print(model)
            input_dummy = torch.randn(1, 3, 384, 384)
            print(model(input_dummy).shape)
            
            # 분류 헤드 제거 (특징 추출만 사용)

            model.to(self.device)
            model.eval()
            
            # 사전학습된 가중치 로드
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
            
            model.load_state_dict(new_state_dict, strict=False)
            self.model = model
            
            return
        
        if self.model_type == 'dinov2':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
            model.to(self.device)
            model.eval()
            self.model = model
            
            return
        
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

            self.model = MS_Swin_Transformer_V2(**init_args)
            
            self.model.to(self.device)

            if hasattr(self.model, 'head'): # MLP 제거 특징맵 사용
                self.model.head = torch.nn.Identity()
            
            print(f"Model initialized on {self.device}")
            
            self.model.eval()
            
        else:
            print("TransReID ViT model loading...")
            self.model = VIT_EXTEND(
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

