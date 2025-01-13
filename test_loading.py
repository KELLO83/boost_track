import sys
import torch
from yacs.config import CfgNode as CN
import os.path as osp
import math

# Add TransReID-SSL path
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from tracker.TransReID_SSL.transreid_pytorch.model.make_model import make_model

def get_default_config():
    cfg = CN()
    
    # model
    cfg.MODEL = CN()
    cfg.MODEL.NAME = 'transformer'
    cfg.MODEL.PRETRAIN_CHOICE = 'self'  # 사전 학습된 가중치 사용
    cfg.MODEL.PRETRAIN_PATH = 'external/weights/transformer_120.pth'  # 실제 가중치 파일 경로
    cfg.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'  # ViT-B/16가 최고 성능
    cfg.MODEL.STRIDE_SIZE = [16, 16]  # 16x16 stride
    cfg.MODEL.PATCH_SIZE = [16, 16]  # 16x16 patch size
    cfg.MODEL.PATCH_NORM = True  # Use patch normalization
    
    # SSL specific settings
    cfg.MODEL.SSL = CN()
    cfg.MODEL.SSL.NECK = 'prj'  # SSL projection head
    cfg.MODEL.SSL.FEAT_DIM = 2048  # SSL feature dimension
    
    # JPM and SIE settings
    cfg.MODEL.JPM = True  # Enable JPM for part-based feature learning
    cfg.MODEL.SIE_CAMERA = True  # Enable camera embedding
    cfg.MODEL.SIE_VIEW = True  # Enable view embedding
    cfg.MODEL.SIE_COE = 3.0  # SIE coefficient
    cfg.MODEL.RE_ARRANGE = True  # Enable feature rearrangement
    cfg.MODEL.SHUFFLE_GROUP = 4  # Increased shuffle groups
    cfg.MODEL.SHIFT_NUM = 8  # Increased shift number
    cfg.MODEL.DEVIDE_LENGTH = 4  # Optimal divide length
    
    # Model parameters
    cfg.MODEL.DROP_PATH = 0.1  # Increased for better regularization
    cfg.MODEL.DROP_OUT = 0.0  # Light dropout
    cfg.MODEL.ATT_DROP_RATE = 0.0  # Attention dropout
    cfg.MODEL.NECK = 'bnneck'
    cfg.MODEL.NECK_FEAT = 'after'
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'  # Circle loss performs better
    cfg.MODEL.LAST_STRIDE = 1
    cfg.MODEL.GEM_POOLING = False  # 기존 설정 유지
    cfg.MODEL.STEM_CONV = False  # Use conv stem
    cfg.MODEL.PRETRAIN_HW_RATIO = 1
    cfg.MODEL.COS_LAYER = False
    cfg.MODEL.REDUCE_FEAT_DIM = False
    cfg.MODEL.FEAT_DIM = 2048
    cfg.MODEL.DROPOUT_RATE = 0.0
    
    # input
    cfg.INPUT = CN()
    cfg.INPUT.SIZE_TRAIN = [256, 128]  # 원래 입력 크기로 복원
    cfg.INPUT.PROB = 0.5
    cfg.INPUT.RE_PROB = 0.5
    cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    
    # test
    cfg.TEST = CN()
    cfg.TEST.NECK_FEAT = 'after'
    cfg.TEST.FEAT_NORM = 'yes'  # Feature normalization for the loaded weights
    
    # solver
    cfg.SOLVER = CN()
    cfg.SOLVER.COSINE_MARGIN = 0.35  # Circle loss margin
    cfg.SOLVER.COSINE_SCALE = 64  # Circle loss scale
    
    return cfg

def main():
    cfg = get_default_config()
    model = make_model(cfg, 100, 1, 1)  # num_classes=100, camera_num=1, view_num=1
    
    # Load pretrained weights
    if cfg.MODEL.PRETRAIN_PATH and cfg.MODEL.PRETRAIN_CHOICE == 'self':
        state_dict = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove incompatible keys
        incompatible_keys = ['classifier', 'bottleneck']
        state_dict = {k: v for k, v in state_dict.items() 
                     if not any(key in k for key in incompatible_keys)}
        
        # Resize position embeddings if needed
        if 'base.pos_embed' in state_dict:
            pos_embed_checkpoint = state_dict['base.pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            
            # 디버깅을 위한 정보 출력
            print("Original pos_embed shape:", pos_embed_checkpoint.shape)
            print("Embedding size:", embedding_size)
            
            # 기존 position embedding에서 cls token과 patch tokens 분리
            cls_pos_embed = pos_embed_checkpoint[:, 0:1, :]
            pos_embed_checkpoint = pos_embed_checkpoint[:, 1:, :]
            
            print("After splitting - cls token shape:", cls_pos_embed.shape)
            print("After splitting - patch tokens shape:", pos_embed_checkpoint.shape)
            
            # Patch position embedding 크기 조정
            pos_h = cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.PATCH_SIZE[0]
            pos_w = cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.PATCH_SIZE[1]
            print("Target size (h, w):", pos_h, pos_w)
            
            # 원본 크기는 14x14 (196 - 1 = 195 patches)
            pos_embed_checkpoint = pos_embed_checkpoint.reshape(1, 16, 12, embedding_size)
            print("Reshaped patch tokens shape:", pos_embed_checkpoint.shape)
            
            pos_embed_checkpoint = torch.nn.functional.interpolate(
                pos_embed_checkpoint.permute(0, 3, 1, 2),
                size=(pos_h, pos_w),
                mode='bilinear',
                align_corners=True)
            print("After interpolation shape:", pos_embed_checkpoint.shape)
            
            pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 3, 1).reshape(1, -1, embedding_size)
            print("Final patch tokens shape:", pos_embed_checkpoint.shape)
            
            # cls token과 조정된 patch position embedding 결합
            new_pos_embed = torch.cat((cls_pos_embed, pos_embed_checkpoint), dim=1)
            print("Final pos_embed shape:", new_pos_embed.shape)
            
            state_dict['base.pos_embed'] = new_pos_embed
        
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded pretrained weights from {cfg.MODEL.PRETRAIN_PATH}")
    
    print("Model created with TransReID-SSL and JPM:", cfg.MODEL.JPM)
    print("SSL pretrain:", cfg.MODEL.PRETRAIN_CHOICE)
    print(model)
    
if __name__ == '__main__':
    main()