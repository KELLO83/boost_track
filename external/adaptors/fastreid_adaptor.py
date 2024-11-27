import torch
import os
from pathlib import Path

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer


def setup_cfg(config_file, opts):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    
    if isinstance(opts[1], Path):
        opts[1] = str(opts[1])
        
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


class FastReID(torch.nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        if isinstance(weights_path, Path):
            weights_path = str(weights_path)
            
        weights_name = os.path.basename(weights_path)
        
        if "R101-ibn" in weights_name:
            config_file = "external/fast_reid/configs/MOT17/sbs_R101-ibn.yml"
        elif "S50" in weights_name:
            config_file = "external/fast_reid/configs/MOT17/sbs_S50.yml"
        else:
            config_file = "external/fast_reid/configs/MOT17/sbs_R50-ibn.yml"
            
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model.cuda()
        Checkpointer(self.model).load(weights_path)
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def forward(self, batch):
        with torch.no_grad():
            return self.model(batch)
