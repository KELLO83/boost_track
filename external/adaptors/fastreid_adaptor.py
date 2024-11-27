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
        CONFIGS_NAME = ['DukeMTMC', 'Market1501', 'MSMT17' , 'MOT17']
        DATASET_MAPPING = {
            'duke': 'DukeMTMC',
            'market': 'Market1501',
            'msmt': 'MSMT17',
            'mot17': 'MOT17',
        }
        
        if isinstance(weights_path, Path):
            weights_path = str(weights_path)
            
        weights_name = os.path.basename(weights_path)
        print('weights_name: ', weights_name)
        DatasetName = weights_name.split('_')[0].lower()
        if DatasetName in DATASET_MAPPING:
            model_name = '_'.join(weights_name.split('_')[1:]).replace('.pth', '.yml')
            config_file = f'external/fast_reid/configs/{DATASET_MAPPING[DatasetName]}/{model_name}'

        else:
            raise ValueError(f'Unknown dataset: {DatasetName}')
            
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model.cuda()

        Checkpointer(self.model).load(weights_path)
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def forward(self, batch):
        with torch.no_grad():
            return self.model(batch)
