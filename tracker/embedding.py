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


@dataclass
class ModelConfig:
    """Base configuration class for all models"""
    crop_size: Tuple[int, int]
    model_type: str
    
    def get_transform(self) -> T.Compose:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(self.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class ModelFactory:
    """Factory class for creating model instances"""
    @staticmethod
    def create_model(config, device):
        model_creators = {
            'dinov2': Dinov2ModelCreator(),
            'CLIP': CLIPModelCreator(),
            'CLIP_RGB': CLIPRGBModelCreator(),
            'swinv2': SwinV2ModelCreator(),
            'La_Transformer': LATransformerModelCreator(),
            'CTL': CTLModelCreator(),
            'VIT-B/16+ICS_SSL': VITSSLModelCreator(),
            'VIT_SSL_MARKET': VITSSLMarketModelCreator(),
            'convNext': ConvNextModelCreator(),
        }
        
        creator = model_creators.get(config.model_name)
        if not creator:
            raise ValueError(f"Unsupported model type: {config.model_name}")
        
        return creator.create_model(config, device)

class BaseModelCreator:
    """Base class for model creators"""
    def create_model(self, config, device):
        raise NotImplementedError

class Dinov2ModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
        model.to(device)
        model.eval()
        return model

class CLIPModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        embed_dim = 512
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
        h_resolution = 14
        w_resolution = 14
        
        from tracker.CLIP.model.clip.model import CLIP
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
        model.load_state_dict(torch.load(config.reid_model_path), strict=False)
        model.to(device)
        return model

class CLIPRGBModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        embed_dim = 768
        image_resolution = 224
        h_resolution = 14
        w_resolution = 14
        vision_layers = 12
        vision_width = 768
        vision_patch_size = 16
        vision_stride_size = 16
        context_length = 77
        vocab_size = 49408
        transformer_width = 512
        transformer_heads = 8
        transformer_layers = 12
        
        from tracker.CLIP.model.clip.model import RGBEncodedCLIP
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
        model.load_state_dict(torch.load(config.reid_model_path), strict=False)
        model.to(device)
        return model

class SwinV2ModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        from .MS_Swin_Transformer.models.swin_transformer_v2 import SwinTransformerV2 as MS_Swin_Transformer_V2
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
        model = MS_Swin_Transformer_V2(**init_args)
        model.to(device)
        if hasattr(model, 'head'):
            model.head = torch.nn.Identity().to(device)
        model.eval()
        return model

class LATransformerModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        import timm
        from tracker.LA_Transformer.LATransformer.model import LATransformer
        base_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=751
        )
        model = LATransformer(base_model, lmbd=0.2)
        model.load_state_dict(torch.load(config.reid_model_path), strict=False)
        model.to(device)
        return model

class CTLModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        import os
        from tracker.centroids_reid.modelling.backbones.resnet_ibn_a import resnet50_ibn_a
        model = resnet50_ibn_a(last_stride=1, pretrained=True)
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()
        if os.path.exists(config.reid_model_path):
            try:
                state_dict = torch.load(config.reid_model_path, map_location=device)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Continuing with pretrained weights only...")
        model.to(device)
        return model

class VITSSLModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        model = VIT_EXTEND(
            img_size=(256, 128),
            stride_size=16,
            drop_path_rate=0.1,
            camera=0,
            view=0,
            sie_xishu=0.0,
            local_feature=False,
            gem_pool=True,
            stem_conv=True,
            num_classes=0
        )
        model.load_state_dict(torch.load(config.reid_model_path), strict=False)
        model.to(device)
        return model

class VITSSLMarketModelCreator(BaseModelCreator):
    def create_model(self, config, device):
        checkpoint = torch.load(config.reid_model_path, map_location=device)
        base = VIT_EXTEND(
            img_size=(384, 128),
            stride_size=16,
            drop_path_rate=0.1,
            camera=0,
            view=0,
            sie_xishu=0.0,
            local_feature=False,
            gem_pool=True,
            stem_conv=True,
            num_classes=0
        )
        class FullModel(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            
            def forward(self, x):
                return self.base(x)
            
            def forward_features(self, x, cam_label=None, view_label=None):
                return self.base.forward_features(x, cam_label, view_label)
        
        model = FullModel(base)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        return model

class ConvNextModelCreator(BaseModelCreator):
    def create_model(self, config, device):
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
        SIZE = str(config.reid_model_path).split('/')[-1].split('_')[1]
        model = ConvNeXt(
            depths = config_model[SIZE]['depths'],
            dims = config_model[SIZE]['dims']
        )
        model.to(device)
        if hasattr(model, 'head'):
            model.head = torch.nn.Identity().to(device)
        model.eval()
        return model

class ModelConfigFactory:
    """Factory for creating model configurations"""
    @staticmethod
    def create_config(model_name: str) -> ModelConfig:
        configs = {
            'dinov2': ModelConfig(crop_size=(448, 448), model_type='dinov2'),
            'CLIP': ModelConfig(crop_size=(224, 224), model_type='CLIP'),
            'CLIP_RGB': ModelConfig(crop_size=(224, 224), model_type='CLIP_RGB'),
            'swinv2': ModelConfig(crop_size=(192, 192), model_type='swinv2'),
            'La_Transformer': ModelConfig(crop_size=(224, 224), model_type='La_Transformer'),
            'CTL': ModelConfig(crop_size=(224, 224), model_type='CTL'),
            'VIT-B/16+ICS_SSL': ModelConfig(crop_size=(256, 128), model_type='VIT-B/16+ICS_SSL'),
            'VIT_SSL_MARKET': ModelConfig(crop_size=(384, 128), model_type='VIT_SSL_MARKET'),
            'convNext': ModelConfig(crop_size=(384, 384), model_type='convNext'),
        }
        
        if model_name not in configs:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        return configs[model_name]



class BatchEmbeddingProcessor:
    """배치 임베딩 처리를 위한 기본 클래스"""
    def process_batch(self, batch_input, model, batch_image=None):
        raise NotImplementedError

class CLIPRGBProcessor(BatchEmbeddingProcessor):
    def __init__(self):
        self.device = torch.device('cuda')
        
    def preprocess_with_rgb_stats(self, batch_img, transform):
        """
        RGB 통계를 사용하여 이미지 배치를 전처리합니다.
        Args:
            batch_img: 전처리할 이미지 배치
            transform: 이미지 변환을 위한 transform 함수
        Returns:
            tuple: (전처리된 이미지 텐서, RGB 통계 텐서)
        """
        processed_batch = []
        for img in batch_img:
            processed = transform(img)
            processed_batch.append(processed)
        
        batch_tensor = torch.stack(processed_batch).to(self.device)
        
        # RGB 통계 계산 (mean과 std 각각 3차원)
        rgb_mean = torch.tensor([[0.5, 0.4, 0.3]]).to(self.device)  # RGB 평균값
        rgb_std = torch.tensor([[0.2, 0.2, 0.2]]).to(self.device)   # RGB 표준편차
        rgb_stats = torch.cat([rgb_mean, rgb_std], dim=1)      # [1, 6] 형태로 결합
        
        # 배치 크기만큼 RGB 통계 복제
        rgb_stats_batch = rgb_stats.repeat(len(batch_img), 1)
        
        return batch_tensor, rgb_stats_batch
        
    def process_batch(self, batch_input, model, batch_image=None):
        batch_input, rgb_stats_batch = self.preprocess_with_rgb_stats(batch_image, model.transform)
        return model(batch_input, rgb_stats_batch)

class CLIPProcessor(BatchEmbeddingProcessor):
    def process_batch(self, batch_input, model, batch_image=None):
        image_features = model.encode_image(batch_input)
        batch_embeddings = image_features[-1]
        return batch_embeddings[-1][0:]

class LaTransformerProcessor(BatchEmbeddingProcessor):
    def process_batch(self, batch_input, model, batch_image=None):
        batch_embeddings = model(batch_input)
        return torch.mean(batch_embeddings, dim=1)

class CTLProcessor(BatchEmbeddingProcessor):
    def process_batch(self, batch_input, model, batch_image=None):
        batch_embeddings = model(batch_input)
        return torch.mean(batch_embeddings, dim=[2, 3])

class DefaultProcessor(BatchEmbeddingProcessor):
    def process_batch(self, batch_input, model, batch_image=None):
        return model(batch_input)

class BatchProcessorFactory:
    """배치 프로세서 생성을 위한 팩토리 클래스"""
    @staticmethod
    def create_processor(model_type: str) -> BatchEmbeddingProcessor:
        processors = {
            'CLIP_RGB': CLIPRGBProcessor(),
            'CLIP': CLIPProcessor(),
            'La_Transformer': LaTransformerProcessor(),
            'CTL': CTLProcessor(),
            'swinv2': DefaultProcessor(),
            'convNext': DefaultProcessor(),
            'VIT-B/16+ICS_SSL': DefaultProcessor(),
            'VIT_SSL_MARKET': DefaultProcessor(),
        }
        return processors.get(model_type, DefaultProcessor())



class EmbeddingComputer:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.model_type = config.model_name
        
        self.model_config = ModelConfigFactory.create_config(config.model_name)
        self.transform = self.model_config.get_transform()
        
        print("Model Input size : ", self.model_config.crop_size)
            
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
                    batch_image.append(cropped)
                    if cropped.size == 0:
                        continue
                        
                    tensor = self.transform(cropped)
                    batch_tensors.append(tensor)
                    
            if not batch_tensors:
                continue
                
            # 배치 텐서 생성 및 GPU로 이동
            batch_input = torch.stack(batch_tensors)
            batch_input = batch_input.to(self.device, non_blocking=True)
            
            # 배치 프로세서를 사용하여 임베딩 계산
            processor = BatchProcessorFactory.create_processor(self.model_type)
            with torch.no_grad():
                batch_embeddings = processor.process_batch(batch_input, self.model, batch_image)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                batch_embeddings = batch_embeddings.cpu().numpy()
            
            for j, box in enumerate(batch_bbox):
                if j < len(batch_embeddings):
                    x1, y1, x2, y2 = map(int, box)
                    box_key = f"{tag}_{x1}_{y1}_{x2}_{y2}"
                    self.cache[box_key] = batch_embeddings[j]
                    embeddings.append(batch_embeddings[j])
        
        return np.array(embeddings) if embeddings else np.array([])

    def initialize_model(self):
        self.model = ModelFactory.create_model(self.config, self.device)