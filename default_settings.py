from typing import Union, Dict, Tuple
from pathlib import Path


def get_detector_path_and_im_size(args) -> Tuple[str, Tuple[int, int]]:
    if args.dataset == "mot17":
        if args.test_dataset:
            detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
        else:
            detector_path = "external/weights/bytetrack_ablation.pth.tar"
        size = (800, 1440)
    elif args.dataset == "mot20":
        if args.test_dataset:
            detector_path = "external/weights/bytetrack_x_mot20.tar"
            size = (896, 1600)
        else:
            # Just use the mot17 test model as the ablation model for 20
            detector_path = "external/weights/bytetrack_x_mot17.pth.tar"
            size = (800, 1440)
    else:
        raise RuntimeError("Need to update paths for detector for extra datasets.")
    return detector_path, size


class GeneralSettings:
    values: Dict[str, Union[float, bool, int, str]] = {
        'max_age': 10,
        'min_hits': 2,
        'det_thresh': 0.5,
        'iou_threshold': 0.2,
        'use_ecc': True,
        'use_embedding': True,
        'dataset': 'mot17',
        'test_dataset': False,
        'min_box_area': 10,
        'aspect_ratio_thresh': 1.6
    }

    dataset_specific_settings: Dict[str, Dict[str, Union[float, bool, int]]] = {
        "mot17": {"det_thresh": 0.6},
        "mot20": {"det_thresh": 0.4},
    }

    video_to_frame_rate = {"MOT17-13-FRCNN": 25, "MOT17-11-FRCNN": 30,
                           "MOT17-10-FRCNN": 30, "MOT17-09-FRCNN": 30,
                           "MOT17-05-FRCNN": 14, "MOT17-04-FRCNN": 30,
                           "MOT17-02-FRCNN": 30, "MOT20-05": 25,
                           "MOT20-03": 25, "MOT20-02": 25, "MOT20-01": 25,
                           "MOT17-14-SDP": 25, "MOT17-12-SDP": 30,
                           "MOT17-08-SDP": 30, "MOT17-07-SDP": 30,
                           "MOT17-06-SDP": 14, "MOT17-03-SDP": 30,
                           "MOT17-01-SDP": 30, "MOT17-14-FRCNN": 25,
                           "MOT17-12-FRCNN": 30, "MOT17-08-FRCNN": 30,
                           "MOT17-07-FRCNN": 30, "MOT17-06-FRCNN": 14,
                           "MOT17-03-FRCNN": 30, "MOT17-01-FRCNN": 30,
                           "MOT17-14-DPM": 25, "MOT17-12-DPM": 30,
                           "MOT17-08-DPM": 30, "MOT17-07-DPM": 30,
                           "MOT17-06-DPM": 14, "MOT17-03-DPM": 30, "MOT17-01-DPM": 30,
                           "MOT20-08": 25, "MOT20-07": 25, "MOT20-06": 25, "MOT20-04": 25
                           }

    @staticmethod
    def max_age(seq_name: str) -> int:
        try:
            return max(int(GeneralSettings.video_to_frame_rate[seq_name] * 2), 30)
        except:
            return 30

    @staticmethod
    def __class_getitem__(key: str):
        try:
            return GeneralSettings.dataset_specific_settings[GeneralSettings.values['dataset']][key]
        except:
            return GeneralSettings.values[key]


class BoostTrackConfig:
    def __init__(self,
                 reid_model_path: Union[str, Path] = 'external/weights/mot17_sbs_S50.pth',
                 device: str = 'cuda',
                 max_age: int = 30,
                 min_hits: int = 3,
                 det_thresh: float = 0.5,
                 iou_threshold: float = 0.3,
                 use_reid: bool = True,
                 use_cmc: bool = True,
                 dataset: str = 'mot17',
                 min_box_area: int = 10,
                 aspect_ratio_thresh: float = 1.6,
                 lambda_iou: float = 0.5,
                 lambda_mhd: float = 0.25,
                 lambda_shape: float = 0.25,
                 use_dlo_boost: bool = True,
                 use_duo_boost: bool = True,
                 dlo_boost_coef: float = 0.6,
                 s_sim_corr: bool = False):
        
        self.reid_model_path = Path(reid_model_path)
        self.device = device
        self.max_age = max_age
        self.min_hits = min_hits
        self.det_thresh = det_thresh
        self.iou_threshold = iou_threshold
        self.use_reid = use_reid
        self.use_cmc = use_cmc
        self.dataset = dataset
        self.min_box_area = min_box_area
        self.aspect_ratio_thresh = aspect_ratio_thresh
        
        # BoostTrack 특정 설정
        self.lambda_iou = lambda_iou
        self.lambda_mhd = lambda_mhd
        self.lambda_shape = lambda_shape
        self.use_dlo_boost = use_dlo_boost
        self.use_duo_boost = use_duo_boost
        self.dlo_boost_coef = dlo_boost_coef
        self.s_sim_corr = s_sim_corr

    @classmethod
    def get_default_config(cls):
        return cls()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Invalid parameter: {k}")


class BoostTrackSettings:
    values: Dict[str, Union[float, bool, int, str]] = {
        'lambda_iou': 0.7,  # 0 to turn off
        'lambda_mhd': 0.25,  # 0 to turn off
        'lambda_shape': 0.25,  # 0 to turn off
        'use_dlo_boost': True,  # False to turn off
        'use_duo_boost': True,  # False to turn off
        'dlo_boost_coef': 0.6,  # Irrelevant if use_dlo_boost == False
        's_sim_corr': False  # Which shape similarity function should be used (True == corrected version)
    }
    dataset_specific_settings: Dict[str, Dict[str, Union[float, bool, int]]] = {
        "mot17": {"dlo_boost_coef": 0.65},
        "mot20": {"dlo_boost_coef": 0.5},
    }

    @staticmethod
    def __class_getitem__(key: str):
        try:
            return BoostTrackSettings.dataset_specific_settings[GeneralSettings.values['dataset']][key]
        except:
            return BoostTrackSettings.values[key]


class BoostTrackPlusPlusSettings:
    values: Dict[str, bool] = {
        'use_rich_s': True,
        'use_sb': True,
        'use_vt': True
    }

    @staticmethod
    def __class_getitem__(key: str):
        return BoostTrackPlusPlusSettings.values[key]

