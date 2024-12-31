import numpy as np
from collections import defaultdict

class EmbeddingHistory:
    def __init__(self):
        """
        현재 프레임의 임베딩을 관리하는 클래스
        """
        self.current_embeddings = {}  # 현재 프레임의 임베딩만 저장
        
    def update_embedding(self, track_id, embedding):
        """현재 프레임의 임베딩으로 업데이트"""
        self.current_embeddings[track_id] = embedding
    
    def compute_batch_similarity(self, dets_embs, trackers):
        """현재 프레임의 임베딩만을 사용하여 유사도 계산"""
        if len(trackers) == 0 or dets_embs.size == 0:
            return np.array([])

        # 현재 트래커들의 임베딩 수집
        trk_embs = []
        for t in range(len(trackers)):
            trk_embs.append(trackers[t].get_emb())
        trk_embs = np.array(trk_embs)  # shape: [M, D]

        # L2 정규화: 각 벡터를 단위 벡터로 변환
        dets_embs_norm = dets_embs / np.linalg.norm(dets_embs, axis=1, keepdims=True)  # [N, D]
        trk_embs_norm = trk_embs / np.linalg.norm(trk_embs, axis=1, keepdims=True)    # [M, D]
        
        # 코사인 유사도 계산
        emb_cost = np.dot(dets_embs_norm, trk_embs_norm.T)  # 값의 범위: [-1, 1]
        
        return emb_cost
    
    def remove_id(self, track_id):
        """특정 ID의 현재 임베딩 제거"""
        if track_id in self.current_embeddings:
            del self.current_embeddings[track_id]
            
    def clear(self):
        """모든 현재 임베딩 초기화"""
        self.current_embeddings.clear()


class MeanEmbeddingHistory:
    def __init__(self, max_history_per_id=10):
        """
        각 ID별 임베딩 히스토리를 관리하고 평균 임베딩을 계산하는 클래스
        
        Args:
            max_history_per_id (int): ID당 저장할 최대 임베딩 개수
        """
        self.max_history = max_history_per_id
        self.embedding_history = defaultdict(list)  # ID별 임베딩 히스토리
        self.mean_embeddings = {}  # ID별 평균 임베딩 캐시
        
    def update_embedding(self, track_id, embedding):
        """
        특정 ID의 임베딩 히스토리 업데이트 및 평균 임베딩 재계산
        
        Args:
            track_id (int): 트래커 ID
            embedding (np.ndarray): 새로운 임베딩 벡터
        """
        # 새로운 임베딩 추가
        self.embedding_history[track_id].append(embedding)
        
        # 최대 히스토리 크기 유지
        if len(self.embedding_history[track_id]) > self.max_history:
            self.embedding_history[track_id].pop(0)  # 가장 오래된 임베딩 제거
            
        # 평균 임베딩 업데이트
        self.mean_embeddings[track_id] = np.mean(self.embedding_history[track_id], axis=0)
    
    def compute_batch_similarity(self, dets_embs, trackers):
        """현재 프레임의 임베딩과 트래커들의 평균 임베딩 간 유사도 계산"""
        if len(trackers) == 0 or dets_embs.size == 0:
            return np.array([])

        # 트래커들의 평균 임베딩 수집
        trk_embs = []
        for t in range(len(trackers)):
            mean_emb = self.get_mean_embedding(t)
            if mean_emb is None:  # 히스토리가 없는 경우 현재 임베딩 사용
                mean_emb = trackers[t].get_emb()
            trk_embs.append(mean_emb)
        trk_embs = np.array(trk_embs)  # shape: [M, D]

        # L2 정규화: 각 벡터를 단위 벡터로 변환
        dets_embs_norm = dets_embs / np.linalg.norm(dets_embs, axis=1, keepdims=True)  # [N, D]
        trk_embs_norm = trk_embs / np.linalg.norm(trk_embs, axis=1, keepdims=True)    # [M, D]
        
        # 코사인 유사도 계산
        emb_cost = np.dot(dets_embs_norm, trk_embs_norm.T)  # 값의 범위: [-1, 1]
        
        return emb_cost
    
    def get_mean_embedding(self, track_id):
        """
        특정 ID의 평균 임베딩 반환
        
        Args:
            track_id (int): 트래커 ID
            
        Returns:
            np.ndarray: 평균 임베딩 벡터, 히스토리가 없으면 None
        """
        return self.mean_embeddings.get(track_id, None)
    
    def remove_id(self, track_id):
        """특정 ID의 모든 히스토리 제거"""
        if track_id in self.embedding_history:
            del self.embedding_history[track_id]
        if track_id in self.mean_embeddings:
            del self.mean_embeddings[track_id]
            
    def clear(self):
        """모든 히스토리 초기화"""
        self.embedding_history.clear()
        self.mean_embeddings.clear()
