from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import numpy as np
import torchvision.transforms as T
#from .TransReID_SSL.transreid_pytorch.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_EXTEND
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
        elif config.Model_Name == 'CLIP':
            self.crop_size = (224,240)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        elif config.Model_Name == 'swinv2':
            self.crop_size = (192 , 192)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size),
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
        
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        
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
        batch_image = []
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
                    batch_image.append(cropped) # CLIP_RGB 에서 이미지의 rgb분포도를 사용하기위하여
                    if cropped.size == 0:
                        continue
                        
                    tensor = self.transform(cropped)
                    batch_tensors.append(tensor)
                    
            if not batch_tensors:
                continue
                
            # 배치 텐서 생성 및 GPU로 이동
            batch_input = torch.stack(batch_tensors)
            batch_input = batch_input.to(self.device, non_blocking=True)  # non_blocking=True로 설정하여 성능 향상
            
            # 임베딩 계산
            with torch.no_grad():
                
                if self.model_type == 'CLIP_RGB':
                    preprocessed = self.preprocess_with_rgb_stats(batch_img)
                    batch_input, rgb_stats_batch = preprocessed 
                    batch_embeddings = self.model(batch_input , rgb_stats_batch) [1,211,768]
                    
                if self.model_type == 'CLIP': # CLIP 전용 forward 추론 -> image만을 가지고 추론
                    image_features = self.model.encode_image(batch_input) # CLIP 는 3가지 특징맵을 반환함
                    batch_embeddings = image_features[-1] # 마지막 사용 # [1 , 211, 768]
                    batch_embeddings = batch_embeddings[-1][0 : ]  # [batch_size , token , embeddding]
                
                else: # SWINV2 , CONVNEXT
                    batch_embeddings = self.model(batch_input)  # swinV2 [1 1536] convnext [1 2048]
                
        
            print("batch_embeddings : ", batch_embeddings.shape)
            
            # GPU에서 정규화 수행 후 CPU로 이동
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
        from tracker.CLIP.model.clip.model import CLIP 
               
        print("Model type : ", self.model_type)
        weight_path = self.config.reid_model_path
        if self.model_type == 'CLIP':
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
        
        
        
            model = CLIP(
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

            model.load_state_dict(torch.load(weight_path), strict=False)
            model.to(self.device)
            self.model = model
        
            return
        
        
        if self.model_type == 'convNext':

            from .ConvNeXt.models.convnext import ConvNeXt
            config_model ={
                'xlarge':{
                    'depths':[3, 3, 27, 3],
                    'dims':[256, 512, 1024, 2048]
                },
                'large':{
                    'depths':[3, 3, 27, 3],
                    'dims':[192, 384, 768, 1536]
                },
                'base':{
                    'depths':[3, 3, 27, 3],
                    'dims':[128, 256, 512, 1024]                            
                },
                'small':{
                    'depths':[3, 3, 27, 3],
                    'dims':[96, 192, 384, 768]
                }
                
            }
            SIZE = str(self.config.reid_model_path).split('/')[-1].split('_')[1]
            
            
            print(f"ConvNeXt model loading {SIZE}...")

            # ConvNeXt 모델 초기화 (Base 모델 기준)
            model = ConvNeXt(
                depths = config_model[SIZE]['depths'],
                dims = config_model[SIZE]['dims']
            )
            
            model.to("cuda")
            if hasattr(model, 'head'):
                model.head = torch.nn.Identity()
            
            
            #print(model)
            #input_dummy = torch.randn(1, 3, 384, 384).to("cuda")
            #print(model(input_dummy).shape)
            
            # 분류 헤드 제거 (특징 추출만 사용)
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
            from .MS_Swin_Transformer.models.swin_transformer_v2 import SwinTransformerV2 as MS_Swin_Transformer_V2
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
            
            # 모델을 GPU로 이동
            self.model = self.model.to(self.device)
            
            # head를 Identity로 변경하고 GPU로 이동
            if hasattr(self.model, 'head'):
                self.model.head = torch.nn.Identity().to(self.device)
            
            # 모델의 모든 파라미터가 GPU로 이동했는지 확인
            for param in self.model.parameters():
                if param.device != self.device:
                    print(f"Warning: Parameter found on {param.device}, moving to {self.device}")
                    param.data = param.data.to(self.device)
            
            print(f"Model initialized on {self.device}")
            print(f"Model parameters device check: {next(self.model.parameters()).device}")
            
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



    def preprocess_with_rgb_stats(self, batch_img):
        """
        RGB 통계를 사용하여 이미지 배치를 전처리합니다.
        Args:
            batch_img: 전처리할 이미지 배치
        Returns:
            tuple: (전처리된 이미지 텐서, RGB 통계 텐서)
        """
        processed_batch = []
        for img in batch_img:
            processed = self.transform(img)
            processed_batch.append(processed)
        
        batch_tensor = torch.stack(processed_batch).to(self.device)
        
        # RGB 통계 계산 (mean과 std 각각 3차원)
        rgb_mean = torch.tensor([[0.5, 0.4, 0.3]]).to(self.device)  # RGB 평균값
        rgb_std = torch.tensor([[0.2, 0.2, 0.2]]).to(self.device)   # RGB 표준편차
        rgb_stats = torch.cat([rgb_mean, rgb_std], dim=1)      # [1, 6] 형태로 결합
        
        # 배치 크기만큼 RGB 통계 복제
        rgb_stats_batch = rgb_stats.repeat(len(batch_img), 1)
        
        return batch_tensor, rgb_stats_batch