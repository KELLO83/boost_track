from ast import mod
from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import math
from torchvision import transforms
import cv2
#from .TransReID_SSL.transreid_pytorch.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_EXTEND
#from .TransReID_SSL.transreid_pytorch.model.backbones.Microsoft_swinv2_trasformer import SwinTransformerV2 as MS_Swin_Transformer_V2
from .ConvNeXt.models.convnext import ConvNeXt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torchinfo


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
        elif config.Model_Name == 'Clip' or config.Model_Name == 'CLIP_RGB':
            self.crop_size = (224,240)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        elif config.Model_Name == 'La_Transformer':
            self.crop_size = (224, 224)
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
        
    def compute_embedding(self, img, boxes, tag=''):
        """
        이미지와 박스들에 대한 임베딩을 계산합니다.
        """
        embeddings = []
        batch_bbox = []
        batch_img = []
        
        # 배치 처리를 위한 데이터 준비
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"
            
            # 이미 캐시된 경우 재사용
            if box_key in self.cache:
                embeddings.append(self.cache[box_key])
                continue
                
            # 캐시되지 않은 경우 배치에 추가
            batch_bbox.append(box)
            cropped = img[y1:y2, x1:x2]
            batch_img.append(cropped)
        
        # 배치가 비어있으면 조기 반환
        if not batch_img:
            return np.array(embeddings) if embeddings else np.array([])
        
        # 이미지 전처리
        if self.model_type == 'CLIP_RGB':
            preprocessed = self.preprocess_with_rgb_stats(batch_img)
            batch_input, rgb_stats_batch = preprocessed
            
        else:
            batch_input = self.preprocess(batch_img)
        
        # 임베딩 계산
        with torch.no_grad():
            if self.model_type == 'La_Transformer':
                batch_embeddings = self.model(batch_input)  # [1, 14, 768]
                batch_embeddings = batch_embeddings.squeeze(0)  # [14, 768]
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                # 14개의 part features를 평균하여 하나의 벡터로 만듦
                batch_embeddings = torch.mean(batch_embeddings, dim=0, keepdim=True)  # [1, 768]
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                batch_embeddings = batch_embeddings.cpu().numpy()  # [1, 768]
                
                # LA-Transformer는 전체 특징맵을 하나의 임베딩으로 저장
                for box in batch_bbox:
                    x1, y1, x2, y2 = map(int, box)
                    box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"
                    if box_key not in self.cache:
                        self.cache[box_key] = batch_embeddings.squeeze(0)  # [768]
                        embeddings.append(batch_embeddings.squeeze(0))
                
                return np.array(embeddings) if embeddings else np.array([])  # [num_dets, 768]
            
            else:
                # 다른 모델들의 처리 (CLIP, CLIP_RGB 등)
                if self.model_type == 'CLIP_RGB':
                    batch_embeddings = self.model(batch_input, rgb_stats=rgb_stats_batch)
                elif self.model_type == 'CLIP':
                    image_features = self.model.encode_image(batch_input)
                    batch_embeddings = image_features[-1]
                else:
                    batch_embeddings = self.model(batch_input)
                
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                batch_embeddings = batch_embeddings.cpu().numpy()
                
                # 기존 방식의 캐시 저장
                current_idx = 0
                for box in batch_bbox:
                    if current_idx >= len(batch_embeddings):
                        break
                    x1, y1, x2, y2 = map(int, box)
                    box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"
                    if box_key not in self.cache:
                        self.cache[box_key] = batch_embeddings[current_idx]
                        embeddings.append(batch_embeddings[current_idx])
                        current_idx += 1
        
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
        
        if self.model_type == 'La_Transformer':
            from tracker.LA_Transformer.LATransformer.model import LATransformer
            from timm import create_model
            
            # 1. 기본 ViT 모델 생성
            base_model = create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0  # 분류 헤드 제거
            )
            
            # 2. LA-Transformer 초기화
            model = LATransformer(
                model=base_model,
                lmbd=4
            )
            
            # 3. 사전 학습된 가중치 로드
            state_dict = torch.load(weight_path, map_location='cpu')
            
            # classifier 관련 키 제거
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                if not k.startswith('classifier') and 
                                    not k.startswith('model.head')}
            
            # 필터링된 가중치만 로드
            model.load_state_dict(filtered_state_dict, strict=False)
            
            # classifier 속성 제거
            for i in range(14):  # 14개의 classifier 제거
                if hasattr(model, f'classifier{i}'):
                    delattr(model, f'classifier{i}')
            
            
            self.model = model.eval()  # 평가 모드로 설정
            self.model.to("cuda")
            print(model)
            print("La Transformer Model Initialized (Feature Extractor Only)")
            
            return
        
        if self.model_type == 'CLIP_RGB':
            from  tracker.CLIP.model.clip.model import RGBEncodedCLIP
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
            
            
                        # 샘플 입력 생성
            input_dummy = torch.randn(1, 3, 224, 240).to(self.device)  # 이미지 입력

            # RGB 통계 샘플 생성 (mean과 std 각각 3차원)
            rgb_mean = torch.tensor([[0.5, 0.4, 0.3]]).to(self.device)  # 예시 RGB 평균값
            rgb_std = torch.tensor([[0.2, 0.2, 0.2]]).to(self.device)   # 예시 RGB 표준편차
            rgb_stats = torch.cat([rgb_mean, rgb_std], dim=1)      # [1, 6] 형태로 결합
            
            
            model = model.to(self.device)
            model.load_state_dict(torch.load(weight_path) , strict = False)
            
            
            with torch.no_grad():
                # 이미지와 RGB 통계로 유사도 계산
                similarity = model(input_dummy, rgb_stats)
                print("Image-RGB Similarity:", similarity)
                print("Similarity Shape:", similarity.shape)

            self.model = model
            return
        
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

    def preprocess(self, batch_img):
        """
        이미지 배치를 전처리합니다.
        """
        if self.model_type == 'La_Transformer':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                   (0.26862954, 0.26130258, 0.27577711))
            ])

        processed = []
        for img in batch_img:
            if img.size == 0:
                continue
            # Convert to RGB if necessary
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            processed_tensor = transform(img)
            processed.append(processed_tensor)
            
        if not processed:
            return None
            
        batch_tensor = torch.stack(processed)
        return batch_tensor.to(self.device)

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
