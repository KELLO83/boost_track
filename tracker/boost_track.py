"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import os
from copy import deepcopy
from typing import Optional, List
import pandas as pd

import cv2
import numpy as np
from ultralytics import cfg


from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings, BoostTrackConfig
from tracker.embedding import EmbeddingComputer
from tracker.assoc import associate, iou_batch, MhDist_similarity, shape_similarity, soft_biou_batch, linear_assignment
from tracker.ecc import ECC
from tracker.kalmanfilter import KalmanFilter
from .embedding_factory import EmbeddingHistoryFactory
from .embedding_history import EmbeddingHistory, MeanEmbeddingHistory, TemplateEmbeddingHistory

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,h,r] where x,y is the centre of the box and h is the height and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    r = w / float(h + 1e-6)

    return np.array([x, y, h,  r]).reshape((4, 1))




def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,h,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h

    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, emb: Optional[np.ndarray] = None):
        """
        Initialises a tracker using initial bounding box.
        """

        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update-1)

    def update(self, bbox: np.ndarray, score: float = 0):
        """
        Updates the state vector with observed bbox.
        """

        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h,  w / h]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb


class BoostTrack(object):
    def __init__(self, cfg: BoostTrackConfig = None):
        if cfg is None:
            cfg = BoostTrackConfig.get_default_config()
            
        # ID 카운터 초기화
        KalmanBoxTracker.count = 0
            
        self.frame_count = 0
        self.trackers = []
        
        #임베딩 유사도값 
        self.emb_sim_score = cfg.emb_sim_score
        
        # 설정 적용
        self.max_age = cfg.max_age
        self.min_hits = cfg.min_hits
        self.det_thresh = cfg.det_thresh
        self.iou_threshold = cfg.iou_threshold
        
        # 매칭 알고리즘에 사용할 가중치
        self.lambda_iou = cfg.lambda_iou
        self.lambda_mhd = cfg.lambda_mhd
        self.lambda_shape = cfg.lambda_shape
        self.use_dlo_boost = cfg.use_dlo_boost
        self.use_duo_boost = cfg.use_duo_boost
        self.dlo_boost_coef = cfg.dlo_boost_coef
        
        # DLO boost 관련 설정 추가 - BoostTrackPlusPlusSettings
        self.use_rich_s = cfg.use_rich_s if hasattr(cfg, 'use_rich_s') else True  # Rich similarity 사용 여부
        self.use_sb = cfg.use_sb if hasattr(cfg, 'use_sb') else True  # Soft boost 사용 여부 
        self.use_vt = cfg.use_vt if hasattr(cfg, 'use_vt') else True  # Varying threshold 사용 여부
        
        # Re-ID 초기화
        if cfg.use_reid:
            self.embedder = EmbeddingComputer(cfg)
        else:
            self.embedder = None
            
        # CMC 초기화
        if cfg.use_cmc:
            self.ecc = ECC(scale=350, use_cache=True)
        else:
            self.ecc = None

        # 임베딩 히스토리 초기화
        embedding_method = GeneralSettings.get_embedding_method()
        self.embedding_history = EmbeddingHistoryFactory.create(embedding_method)
                
        
    def update(self, dets, img_tensor, img_numpy, tag):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        
        dets 형태 [x1,y1,x2,y2,score]   process_yolo_detection dets.append([x1, y1, x2, y2, conf])
                
        """
    
        if dets is None:
            return np.empty((0, 5))
        if not isinstance(dets, np.ndarray):
            dets = dets.cpu().detach().numpy()

        self.frame_count += 1

        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        dets = deepcopy(dets)
        dets[:, :4] /= scale

        if self.ecc is not None: # Enhanced Correlation Coefficient (ECC) 연속된 프레임간의 카메라 움직임을 추정하는 알고리즘
            transform = self.ecc(img_numpy, self.frame_count, tag) # 현재 프레임과 이전프레임 사이에 변환행렬 계산
            for trk in self.trackers:
                trk.camera_update(transform)

        trks = np.zeros((len(self.trackers), 5))
        
        
                
        # 비활성 트래커 관리
        current_inactive = []
        for trk in self.trackers:
            if trk.time_since_update > 1 and trk.time_since_update <= self.max_age * 2:
                if trk not in self.inactive_trackers:  # 중복 방지
                    current_inactive.append(trk)
        self.inactive_trackers = current_inactive
                
        confs = np.zeros((len(self.trackers), 1))

        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            confs[t] = self.trackers[t].get_confidence()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]

        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost(dets, self.use_rich_s, self.use_sb, self.use_vt)

        if self.use_duo_boost:
            dets = self.duo_confidence_boost(dets)

        remain_inds = dets[:, 4] >= self.det_thresh # dets shape: [N, 5] 객체된객체수 , 객체수마다 가지고있는 x1,y1,x2,y2,score        
        dets = dets[remain_inds] # 새로운 ID를 부여받지도 않고, 트래킹 대상으로도 고려되지 않습니다. 임계치미만의 객체들은
        scores = dets[:, 4] # 신뢰도추출 > 임계치 

        # 임베딩 기반 객체 유사도 계산
        emb_cost = None  
        if self.embedder and dets.size > 0:
            # 현재 프레임의 모든 검출 객체에 대한 임베딩 계산
            # dets_embs shape: [N, D] where N=검출 객체 수, D=768(임베딩 차원)
            dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)
            
            if dets_embs.size == 0:  # 임베딩 계산 실패 시
                raise RuntimeError("Embedding computation failed.")
            
            # 트래커의 임베딩 히스토리 업데이트
            for t in range(len(self.trackers)):
                tracker = self.trackers[t]
                self.embedding_history.update_embedding(tracker.id, tracker.get_emb())
            
            if len(self.trackers) > 0 and dets.size > 0:
                emb_cost = self.embedding_history.compute_batch_similarity(dets_embs, self.trackers)
            
    
        emb_cost = None if self.embedder is None else emb_cost  # 임베딩 계산기가 없으면 None 반환

        iou_matrix = iou_batch(dets, trks[:, :4])
        mh_matrix = self.get_mh_dist_matrix(dets)
        cost_matrix = (self.lambda_iou * iou_matrix + self.lambda_mhd * mh_matrix + self.lambda_shape * shape_similarity(dets, trks)) / (self.lambda_iou + self.lambda_mhd + self.lambda_shape)
        
        # 매칭 수행
        matched, unmatched_dets, unmatched_trks, sym_matrix = associate(
            dets,
            trks,
            self.iou_threshold,
            mahalanobis_distance=self.get_mh_dist_matrix(dets),
            track_confidence=confs,
            detection_confidence=scores,
            emb_cost=emb_cost,
            emb_sim_score=self.emb_sim_score,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape
        )
        
        # print(f"\n{'='*50}")
        # print(f"프레임 {self.frame_count}")
        # print(f"현재 트래커 ID: {[t.id for t in self.trackers]}")
        # print(f"비활성 트래커 ID: {[t.id for t in self.inactive_trackers]}\n")
        
        # if len(matched) > 0:
        #     print("[매칭된 정보]")
        #     print(f"검출-트래커 쌍: {matched}")
        #     print(f"매칭된 트래커 ID: {[self.trackers[t].id for t in matched[:, 1]]}")
        #     print("\n[검출 객체 -> 트래커 ID 매핑]")
        #     for m in matched:
        #         det_idx, trk_idx = m
        #         trk_id = self.trackers[trk_idx].id
        #         print(f"검출 {det_idx} -> ID {trk_id}")
        #     print()
        
        # if len(unmatched_dets) > 0:
        #     print(f"[매칭되지 않은 검출]: {unmatched_dets}\n")
        
        # if len(unmatched_trks) > 0:
        #     print("[매칭되지 않은 트래커]")
        #     print(f"인덱스: {unmatched_trks}")
        #     print(f"트래커 ID: {[self.trackers[t].id for t in unmatched_trks]}\n")
        
        # print("=== 상세 매칭 정보 ===\n")
        # print("[임베딩 유사도 행렬]")
        # if emb_cost is not None:
        #     pd.set_option('display.float_format', lambda x: '%.5f' % x)
        #     df = pd.DataFrame(emb_cost, columns=[f'ID_{t.id}' for t in self.trackers])
        #     df.index = [f'Det_{i}' for i in range(len(emb_cost))]
        #     print(df)
        # else:
        #     print("임베딩 정보 없음")
            
        # print("\n[IOU 행렬]")
        # pd.set_option('display.float_format', lambda x: '%.5f' % x)
        # df = pd.DataFrame(iou_matrix, columns=[f'ID_{t.id}' for t in self.trackers])
        # df.index = [f'Det_{i}' for i in range(len(iou_matrix))]
        # print(df)
        
        # print("\n[마할라노비스 거리 행렬]")
        # pd.set_option('display.float_format', lambda x: '%.5f' % x)
        # df = pd.DataFrame(mh_matrix, columns=[f'ID_{t.id}' for t in self.trackers])
        # df.index = [f'Det_{i}' for i in range(len(mh_matrix))]
        # print(df)
        
        # print("\n[최종 매칭 결과 상세]")
        # for m in matched:
        #     det_idx, trk_idx = m
        #     trk_id = self.trackers[trk_idx].id
        #     print(f"검출 {det_idx} -> ID {trk_id} (임베딩: {emb_cost[det_idx, trk_idx]:.5f}, "
        #           f"IOU: {iou_matrix[det_idx, trk_idx]:.5f}, "
        #           f"마할라노비스: {mh_matrix[det_idx, trk_idx]:.5f})")
        
        # print(f"\n{'='*50}")
        
        
        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = 0.95
        dets_alpha = af + (1 - af) * (1 - trust)

        for m in matched: # 기존 tracking 정보 업데이트
            self.trackers[m[1]].update(dets[m[0], :], scores[m[0]])
            if dets_embs is not None:
                self.trackers[m[1]].update_emb(dets_embs[m[0]])
                # 임베딩 히스토리 업데이트
                self.embedding_history.update_embedding(self.trackers[m[1]].id, dets_embs[m[0]])

        for i in unmatched_dets: # re-id가 안된 새로운 id는 tracking추가
            if dets[i, 4] >= self.det_thresh:
                self.trackers.append(KalmanBoxTracker(dets[i, :], emb=dets_embs[i]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id], [trk.get_confidence()])).reshape(1, -1))  
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))



    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].get_confidence()]

        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = self.trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1, ))[:n_dims]
        for i in range(len(self.trackers)):
            x[i] = self.trackers[i].kf.x[:n_dims]
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(np.diag(self.trackers[i].kf.covariance[:n_dims, :n_dims]))

        return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 * sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, n_dims)

        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)

            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(len(boost_detections))
                bdiou_max = bdiou.max(axis=1)

                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(np.intersect1d(boost_detections_args[args], boost_detections_args[tmp]), boost_detections_args[boxi])

                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(remaining_boxes.tolist() + [boost_detections_args[boxi]])

                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True

            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])

        return detections

    def dlo_confidence_boost(self, detections: np.ndarray, use_rich_sim: bool, use_soft_boost: bool, use_varying_th: bool) -> np.ndarray:
        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections
        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, self.trackers[t].time_since_update - 1]

        if use_rich_sim:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections), 1)
            shape_sim = shape_similarity(detections, trackers)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)

        if not use_soft_boost and not use_varying_th:
            max_s = S.max(1)
            coef = self.dlo_boost_coef
            detections[:, 4] = np.maximum(detections[:, 4], max_s * coef)

        else:
            if use_soft_boost:
                max_s = S.max(1)
                alpha = 0.65
                detections[:, 4] = np.maximum(detections[:, 4], alpha*detections[:, 4] + (1-alpha)*max_s**(1.5))
            if use_varying_th:
                threshold_s = 0.95
                threshold_e = 0.8
                n_steps = 20
                alpha = (threshold_s - threshold_e) / n_steps
                tmp = (S > np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)).max(1)
                scores = deepcopy(detections[:, 4])
                scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)

                detections[:, 4] = scores

        return detections
