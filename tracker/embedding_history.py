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

        # 디버깅: 트래커 정보 출력
        print("\n=== 임베딩 유사도 계산 ===")
        print("트래커 수:", len(trackers))
        print("검출 수:", len(dets_embs))

        # 트래커들의 평균 임베딩 수집
        trk_embs = []
        for t in range(len(trackers)):
            mean_emb = self.get_mean_embedding(t)
            if mean_emb is None:  # 히스토리가 없는 경우 현재 임베딩 사용
                mean_emb = trackers[t].get_emb()
                print(f"트래커 {trackers[t].id}: 히스토리 없음, 현재 임베딩 사용")
            else:
                print(f"트래커 {trackers[t].id}: 평균 임베딩 사용 (히스토리 크기: {len(self.embedding_history.get(t, []))})")
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



class TemplateEmbeddingHistory:
    def __init__(self, max_templates=3, similarity_threshold=0.7):
        self.template_embeddings = {}  # id -> [embeddings]
        self.max_templates = max_templates
        self.similarity_threshold = similarity_threshold
        
    def update(self, track_id, embedding):
        if track_id not in self.template_embeddings:
            self.template_embeddings[track_id] = [embedding]
            return
            
        templates = self.template_embeddings[track_id]
        
        # 현재 템플릿들과의 유사도 계산
        similarities = [np.dot(embedding, temp) for temp in templates]
        max_similarity = max(similarities) if similarities else 0
        
        # 충분히 다른 외관일 경우에만 새 템플릿으로 추가
        if max_similarity < self.similarity_threshold:
            if len(templates) < self.max_templates:
                templates.append(embedding)
            else:
                # 가장 덜 사용된 템플릿 교체
                templates[np.argmin(similarities)] = embedding
    
    def compute_batch_similarity(self, query_embs, trackers):
        cost_matrix = np.zeros((len(query_embs), len(trackers)))
        
        for i, query_emb in enumerate(query_embs):
            for j, tracker in enumerate(trackers):
                track_id = tracker.id
                if track_id in self.template_embeddings:
                    templates = self.template_embeddings[track_id]
                    # 모든 템플릿과의 유사도 중 최대값 사용
                    similarities = [np.dot(query_emb, temp) for temp in templates]
                    cost_matrix[i, j] = max(similarities)
                
        return cost_matrix
    
    
    
class EnhancedTemplateEmbeddingHistory:
    def __init__(self, max_templates=3, similarity_threshold=0.7, temporal_weight=0.8):
        self.template_embeddings = {}  # id -> [(embedding, score, timestamp)]
        self.max_templates = max_templates
        self.similarity_threshold = similarity_threshold
        self.temporal_weight = temporal_weight
        self.template_scores = {}  # id -> [score for each template]
        
    def update(self, track_id, embedding, timestamp):
        if track_id not in self.template_embeddings:
            self.template_embeddings[track_id] = [(embedding, 1.0, timestamp)]
            self.template_scores[track_id] = [1.0]
            return
            
        templates = [t[0] for t in self.template_embeddings[track_id]]
        scores = self.template_scores[track_id]
        
        # 현재 템플릿들과의 유사도 계산 (코사인 유사도 사용)
        similarities = [self._cosine_similarity(embedding, temp) for temp in templates]
        max_similarity = max(similarities) if similarities else 0
        
        # 템플릿 점수 업데이트 (시간 가중치 적용)
        for i in range(len(scores)):
            time_diff = timestamp - self.template_embeddings[track_id][i][2]
            scores[i] *= self.temporal_weight ** time_diff
        
        if max_similarity < self.similarity_threshold:
            if len(templates) < self.max_templates:
                templates.append(embedding)
                scores.append(1.0)
                self.template_embeddings[track_id].append((embedding, 1.0, timestamp))
            else:
                # 점수와 유사도를 모두 고려하여 교체할 템플릿 선택
                replacement_idx = self._select_replacement_template(scores, similarities)
                self.template_embeddings[track_id][replacement_idx] = (embedding, 1.0, timestamp)
                scores[replacement_idx] = 1.0
                
    def _cosine_similarity(self, emb1, emb2):
        """코사인 유사도 계산 (더 정확한 유사도 측정)"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _select_replacement_template(self, scores, similarities):
        """교체할 템플릿 선택 (점수와 유사도 모두 고려)"""
        combined_scores = [score * (1 - sim) for score, sim in zip(scores, similarities)]
        return np.argmin(combined_scores)
    
    def compute_batch_similarity(self, query_embs, trackers):
        cost_matrix = np.zeros((len(query_embs), len(trackers)))
        
        for i, query_emb in enumerate(query_embs):
            for j, tracker in enumerate(trackers):
                track_id = tracker.id
                if track_id in self.template_embeddings:
                    templates = [t[0] for t in self.template_embeddings[track_id]]
                    scores = self.template_scores[track_id]
                    
                    # 코사인 유사도와 템플릿 점수를 결합
                    similarities = [self._cosine_similarity(query_emb, temp) * score 
                                 for temp, score in zip(templates, scores)]
                    cost_matrix[i, j] = max(similarities)
                
        return cost_matrix