from ast import Import
from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import torchvision
import numpy as np
import torchvision.transforms as T
from .TransReID_SSL.transreid_pytorch.model.backbones.vit_pytorch import vit_base_patch16_224_TransReID as VIT_EXTEND
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torchinfo
import torch
import torch.nn as nn

from tracker.CLIP.model.clip.model import RGBEncodedCLIP


class EmbeddingComputer:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.model_type = config.model_name
        
        if config.model_name == 'dinov2':
            self.crop_size = (448, 448)  # 14의 배수인 448x448 사용 (14*32)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.crop_size), antialias=True),  
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif config.model_name == 'CLIP' or config.model_name == 'CLIP_RGB':
            self.crop_size = (224, 224)  # 16의 배수로 맞춤 (14x14 패치 + 1 cls token = 197)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        elif config.model_name == 'swinv2':
            self.crop_size = (192 , 192)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        elif config.model_name == 'La_Transformer' or config.model_name == 'CTL':
            self.crop_size = (224, 224)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size),  # LA Transformer requires 224x224 input
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
            
        elif config.model_name =='VIT-B/16+ICS_SSL':
            self.crop_size = (256 , 128)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        elif config.model_name == 'VIT_SSL_MARKET':
            self.crop_size = (384, 128)
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
                    preprocessed = self.preprocess_with_rgb_stats(batch_image)
                    batch_input, rgb_stats_batch = preprocessed 
                    # forward에서 이미 정규화된 결합 특징을 반환
                    batch_embeddings = self.model(batch_input, rgb_stats_batch)
                    
                elif self.model_type == 'CLIP': # CLIP 전용 forward 추론 -> image만을 가지고 추론 VIT사용
                    image_features = self.model.encode_image(batch_input) # CLIP 는 3가지 특징맵을 반환함
                    batch_embeddings = image_features[-1] # 마지막 사용 # [1 , 197, 512]
                    batch_embeddings = batch_embeddings[-1][0 : ]  # [batch_size , token , embeddding]
                    print("CLIP BATCH EMBEDDINGS SHAPE : ", batch_embeddings.shape)

                    
                elif self.model_type == 'La_Transformer':
                    batch_embeddings = self.model(batch_input)  # [1, 14, 768]
                    # 모든 부분 특징의 평균을 계산하여 하나의 특징 벡터로 만듦
                    batch_embeddings = torch.mean(batch_embeddings, dim=1)  # [1, 768] # 2차원으로 반영
                
                elif self.model_type == 'CTL':
                    batch_embeddings = self.model(batch_input)  # [1, 2048, 14, 14]
                    # Global average pooling으로 공간 차원 제거
                    batch_embeddings = torch.mean(batch_embeddings, dim=[2, 3])  # [1, 2048]
                
                elif self.model_type == 'swinv2' or self.model_type == 'convNext': # SWINV2 , CONVNEXT
                    batch_embeddings = self.model(batch_input)  # swinV2 [1 1536] convnext [1 2048] La Transformer [1, 14, 768]
                    
                
                elif self.model_type == 'VIT-B/16+ICS_SSL' or self.model_type == 'VIT_SSL_MARKET':
                    batch_embeddings = self.model(batch_input)  # [1 ,768] local 특징만 사용
        
        
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
        
        print("Weight path : ", weight_path)
        if self.model_type == 'CTL':
            import os
            if not os.path.exists(weight_path):
                print(f"Warning: Weight file not found at {weight_path}")
                print("Continuing with pretrained weights only...")
            
            from tracker.centroids_reid.modelling.backbones.resnet_ibn_a import resnet50_ibn_a
            model = resnet50_ibn_a(last_stride=1, pretrained=True)
            
            # Remove FC layer and modify avgpool
            model.avgpool = nn.Identity()  # 공간 정보 유지
            model.fc = nn.Identity()
            
            if os.path.exists(weight_path):
                try:
                    state_dict = torch.load(weight_path, map_location=self.device)
                    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    print(f"Error loading weights: {e}")
                    print("Continuing with pretrained weights only...")
            
            model.to(self.device)
            self.model = model
            
            # Test forward pass
            input_dummy = torch.randn(1, 3, 224, 224).to(self.device)
            out = model(input_dummy)
            print("Raw output shape:", out.shape)
            
            # Global average pooling to get final shape
            out = torch.mean(out, dim=[2, 3])  # [1, 2048, 14, 14] -> [1, 2048]
            print("Final output shape:", out.shape)
            
            return
        
        if self.model_type == 'La_Transformer':
            import timm
            from tracker.LA_Transformer.LATransformer.model import LATransformer
            
            # Load base ViT model from timm
            base_model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=751  # Set to match the original model's class number
            )
            
            # Initialize LA Transformer with the base model
            model = LATransformer(base_model, lmbd=0.2)
            
            # Load pretrained weights
            state_dict = torch.load(weight_path, map_location=self.device)
            
            # Filter out unexpected keys
            model_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            missing_keys = [k for k in model_dict.keys() if k not in filtered_state_dict]
            unexpected_keys = [k for k in state_dict.keys() if k not in model_dict]
            
            if unexpected_keys:
                print("Unexpected keys in state_dict:", unexpected_keys)
            if missing_keys:
                print("Missing keys in state_dict:", missing_keys)
            
            # Load filtered weights
            model.load_state_dict(filtered_state_dict, strict=False)
            
            model.to(self.device)
            self.model = model
            print(model)
            print("LA Transformer model loaded successfully")
            return
        
        if self.model_type == 'CLIP_RGB':
            # CLIP_RGB Base (사전 훈련된 가중치와 호환되는 설정 사용)
            from tracker.CLIP.model.clip.model import RGBEncodedCLIP
            embed_dim = 768                  
            image_resolution = 224          # 입력 이미지 크기
            h_resolution = 14              # 224/16 = 14 (패치 개수)
            w_resolution = 14              # 224/16 = 14 (패치 개수)
            vision_layers = 12             
            vision_width = 768             
            vision_patch_size = 16         
            vision_stride_size = 16        
            context_length = 77            # 사전 훈련된 모델의 context_length 유지
            vocab_size = 49408            # 사전 훈련된 모델의 vocab_size 유지
            transformer_width = 512        # 사전 훈련된 모델의 transformer_width 유지
            transformer_heads = 8          
            transformer_layers = 12        
            
            model = RGBEncodedCLIP(
                embed_dim=embed_dim,
                image_resolution=image_resolution,
                h_resolution=h_resolution,
                w_resolution=w_resolution,
                vision_layers=vision_layers,
                vision_width=vision_width,
                vision_patch_size=vision_patch_size,
                vision_stride_size=vision_stride_size,
                context_length=context_length,
                vocab_size=vocab_size,
                transformer_width=transformer_width,
                transformer_heads=transformer_heads,
                transformer_layers=transformer_layers,
            )

            model.load_state_dict(torch.load(weight_path), strict=False)
            model.to(self.device)
            self.model = model
        
            return
        
        if self.model_type == 'CLIP':
            # CLIP Base (ViT-B/16)
            embed_dim = 512                  # 사전 훈련된 모델과 맞춤
            image_resolution = 224          
            vision_layers = 12             
            vision_width = 768              
            vision_patch_size = 16         
            vision_stride_size = 16        
            context_length = 77            
            vocab_size = 49408            
            transformer_width = 512         
            transformer_heads = 8           
            transformer_layers = 12    
            h_resolution = 14              # 224/16 = 14 (패치 개수)
            w_resolution = 14              # 224/16 = 14 (패치 개수)
        
        
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
        
        if self.model_type =='VIT-B/16+ICS_SSL':
            print("TransReID-SSL VIT-Base+ICS model loading...")
            
            self.model = VIT_EXTEND(
                img_size = (256, 128),         # 16x8 패치 구조를 위한 크기
                stride_size=16,                # 패치 크기는 16x16으로 고정
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                camera=0,                      # SIE 비활성화
                view=0,                        # SIE 비활성화
                sie_xishu=0.0,                # SIE 완전 비활성화
                local_feature=False,            # 중간 feature map 사용
                gem_pool = True,               # Global average pooling 사용 안함
                stem_conv = True,               # Stem convolution 사용
                num_classes=0                 # FC layer 제거
            )
            
            print("Model configuration:")
            print(f"- Architecture: ViT-Base (768 dim, 12 heads)")
            print(f"- Input size: 256x128")
            print(f"- Patch size: 16x16 (16x8 patches)")
            print(f"- Local feature: Enabled (return intermediate features)")
            print(f"- SIE: Disabled")
            print(f"- STEM_CONV: Enabled")
            
            checkpoint = torch.load(weight_path, map_location=self.device)
            
            print("\nModel architecture before removing FC:")
            print(self.model)
            
            
            # Remove FC layer
            self.model.fc = nn.Identity()
            print("\nModel architecture after removing FC:")
            print(self.model)
            
            self.model.to(self.device)
            # Test with dummy input to check feature map shapes
            print("\nFeature map shapes:")
            B = 1  # batch size
            dummy_input = torch.randn(B, 3, 256, 128).to(self.device)
            with torch.no_grad():
                features = self.model.forward_features(dummy_input, None, None)
                if isinstance(features, tuple):
                    for i, feat in enumerate(features):
                        print(f"Feature {i} shape: {feat.shape}")
                else:
                    print(f"Feature shape: {features.shape}")
            
            # Filter out unexpected keys
            model_dict = self.model.state_dict()
            filtered_state_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            missing_keys = [k for k in model_dict.keys() if k not in filtered_state_dict]
            unexpected_keys = [k for k in checkpoint.keys() if k not in model_dict]
            
            if unexpected_keys:
                print("Unexpected keys in state_dict:", unexpected_keys)
            if missing_keys:
                print("Missing keys in state_dict:", missing_keys)
            
            # Load filtered weights
            self.model.load_state_dict(filtered_state_dict, strict=False) 
            self.model.eval()

        if self.model_type =='VIT_SSL_MARKET':
            print("VIT_SSL_MARKET model loading SuperVised Reid...")

            checkpoint = torch.load(f'{weight_path}', map_location=self.device)
            
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
            model.to(self.device)
            print("\nFeature map shapes:")
            B = 1  # batch size
            dummy_input = torch.randn(B, 3, 384, 128).to(self.device)  # 실제 입력 크기 사용
            with torch.no_grad():
                features = model.forward_features(dummy_input, None, None)
                if isinstance(features, tuple):
                    for i, feat in enumerate(features):
                        print(f"Feature {i} shape: {feat.shape}")
                else:
                    print(f"Feature shape: {features.shape}")
            
            self.model = model
            
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