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
        for t in trackers:
            trk_id = t.id
            if trk_id in self.current_embeddings:
                trk_embs.append(self.current_embeddings[trk_id])
                #print(f"트래커 {trk_id}: 저장된 최신 임베딩 사용")
            else:
                trk_embs.append(t.get_emb())
                # print(f"트래커 {trk_id}: 트래커의 현재 임베딩 사용")
        
        trk_embs = np.array(trk_embs)  # shape: [M, D]

        # L2 정규화
        dets_embs_norm = dets_embs / (np.linalg.norm(dets_embs, axis=1, keepdims=True) + 1e-6)
        trk_embs_norm = trk_embs / (np.linalg.norm(trk_embs, axis=1, keepdims=True) + 1e-6)
        
        # 코사인 유사도 계산
        emb_cost = np.dot(dets_embs_norm, trk_embs_norm.T)
        
        return emb_cost
    
    def remove_id(self, track_id):
        """특정 ID의 현재 임베딩 제거"""
        if track_id in self.current_embeddings:
            del self.current_embeddings[track_id]
            
    def clear(self):
        """모든 현재 임베딩 초기화"""
        self.current_embeddings.clear()


class MeanEmbeddingHistory:
    def __init__(self, max_history_per_id = 5):
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
        for t in trackers:
            trk_id = t.id  # 실제 트래커 ID 사용
            mean_emb = self.get_mean_embedding(trk_id)
            if mean_emb is None:  # 히스토리가 없는 경우 현재 임베딩 사용
                mean_emb = t.get_emb()
                print(f"트래커 {trk_id}: 히스토리 없음, 현재 임베딩 사용")
            else:
                print(f"트래커 {trk_id}: 평균 임베딩 사용 (히스토리 크기: {len(self.embedding_history.get(trk_id, []))})")
            trk_embs.append(mean_emb)
        
        trk_embs = np.array(trk_embs)  # shape: [M, D]

        # L2 정규화
        dets_embs_norm = dets_embs / (np.linalg.norm(dets_embs, axis=1, keepdims=True) + 1e-6)
        trk_embs_norm = trk_embs / (np.linalg.norm(trk_embs, axis=1, keepdims=True) + 1e-6)
        
        # 코사인 유사도 계산
        emb_cost = np.dot(dets_embs_norm, trk_embs_norm.T)
        
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
    
    
    
    import numpy as np
from collections import defaultdict

class MeanEmbeddingHistory_enhanced:
    def __init__(self, max_history_per_id=5, 
                 bbox_smooth_alpha=0.2,
                 outlier_thresh=0.65):
        """
        각 ID별 임베딩과 bbox 히스토리를 관리, 평균 임베딩 및 스무싱 bbox를 계산하는 클래스
        
        Args:
            max_history_per_id (int): ID당 저장할 최대 임베딩 개수
            bbox_smooth_alpha (float): EMA 스무싱 시 사용할 alpha (0~1 사이)ㄹ 0에 근접시 과거정보 1에근접시 현재정보 반영
            outlier_thresh (float): 평균 임베딩과 새 임베딩 간 코사인 유사도가 이 값 미만이면 outlier로 간주
        """
        self.max_history = max_history_per_id
        self.outlier_thresh = outlier_thresh
        
        # 임베딩 관리
        self.embedding_history = defaultdict(list)  # ID별 임베딩 히스토리
        self.mean_embeddings = {}  # ID별 평균 임베딩 캐시
        
        # 박스 스무싱용
        self.bbox_smooth_alpha = bbox_smooth_alpha
        self.smoothed_bbox = {}  # ID별 스무싱된 bbox (x, y, w, h)
        self.bbox_history = defaultdict(list)  # 필요하다면 bbox 히스토리도 관리
        
    def update_bbox(self, track_id, new_bbox):
        """
        ID별 박스 정보를 EMA 방식으로 스무싱하여 업데이트
        Args:
            track_id (int): 트래커 ID
            new_bbox (tuple or list): (x, y, w, h) 형태
        """
        if track_id not in self.smoothed_bbox:
            # 최초 진입 시에는 그대로 사용
            self.smoothed_bbox[track_id] = np.array(new_bbox, dtype=np.float32)
        else:
            # 기존 박스와 EMA
            old_bbox = self.smoothed_bbox[track_id]
            alpha = self.bbox_smooth_alpha
            updated_bbox = alpha * np.array(new_bbox, dtype=np.float32) \
                           + (1 - alpha) * old_bbox
            self.smoothed_bbox[track_id] = updated_bbox
        
        # 히스토리 저장 (필요 시)
        self.bbox_history[track_id].append(self.smoothed_bbox[track_id].copy())
        if len(self.bbox_history[track_id]) > self.max_history:
            self.bbox_history[track_id].pop(0)
        
    def get_smoothed_bbox(self, track_id):
        """스무싱된 bbox 반환"""
        return self.smoothed_bbox.get(track_id, None)
    
    def update_embedding(self, track_id, embedding):
        """
        특정 ID의 임베딩 히스토리를 업데이트.
        - outlier 검출 (평균 임베딩과 cos유사도가 낮으면 추가 안 함)
        - 최대 히스토리 사이즈 유지
        - mean embedding 재계산
        """
        # 1) outlier 체크
        if track_id in self.mean_embeddings:
            mean_emb = self.mean_embeddings[track_id]
            # L2 정규화
            mean_emb_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
            new_emb_norm = embedding / (np.linalg.norm(embedding) + 1e-6)
            
            cos_sim = np.dot(mean_emb_norm, new_emb_norm)
            if cos_sim < self.outlier_thresh:
                print(f"[Outlier] TrackID={track_id}, cos_sim={cos_sim:.3f} < {self.outlier_thresh} → skip update")
                return  # outlier로 간주하고 히스토리에 추가 안 함
        
        # 2) 히스토리 추가
        self.embedding_history[track_id].append(embedding)
        if len(self.embedding_history[track_id]) > self.max_history:
            self.embedding_history[track_id].pop(0)  # 가장 오래된 임베딩 제거
        
        # 3) mean_embedding 갱신
        self.mean_embeddings[track_id] = np.mean(self.embedding_history[track_id], axis=0)
    
    def compute_batch_similarity(self, dets_embs, trackers):
        """현재 프레임의 임베딩과 트래커들의 평균 임베딩 간 유사도 계산"""
        if len(trackers) == 0 or dets_embs.size == 0:
            return np.array([])

        print("\n=== 임베딩 유사도 계산 ===")
        print("트래커 수:", len(trackers))
        print("검출 수:", len(dets_embs))

        # 트래커들의 평균 임베딩 수집
        trk_embs = []
        for t in trackers:
            trk_id = t.id  # 실제 트래커 ID
            mean_emb = self.get_mean_embedding(trk_id)
            if mean_emb is None:
                # 히스토리가 없으면 현재 임베딩 사용
                mean_emb = t.get_emb()
                print(f"트래커 {trk_id}: 히스토리 없음, 현재 임베딩 사용")
            else:
                print(f"트래커 {trk_id}: 평균 임베딩 사용 (히스토리 크기: {len(self.embedding_history.get(trk_id, []))})")
            trk_embs.append(mean_emb)
        
        trk_embs = np.array(trk_embs)  # shape: [M, D]

        # L2 정규화
        dets_embs_norm = dets_embs / (np.linalg.norm(dets_embs, axis=1, keepdims=True) + 1e-6)
        trk_embs_norm = trk_embs / (np.linalg.norm(trk_embs, axis=1, keepdims=True) + 1e-6)
        
        # 코사인 유사도
        emb_cost = np.dot(dets_embs_norm, trk_embs_norm.T)
        
        return emb_cost
    
    def get_mean_embedding(self, track_id):
        return self.mean_embeddings.get(track_id, None)
    
    def remove_id(self, track_id):
        """특정 ID의 모든 히스토리 제거"""
        if track_id in self.embedding_history:
            del self.embedding_history[track_id]
        if track_id in self.mean_embeddings:
            del self.mean_embeddings[track_id]
        if track_id in self.smoothed_bbox:
            del self.smoothed_bbox[track_id]
        if track_id in self.bbox_history:
            del self.bbox_history[track_id]
            
    def clear(self):
        """모든 히스토리 초기화"""
        self.embedding_history.clear()
        self.mean_embeddings.clear()
        self.smoothed_bbox.clear()
        self.bbox_history.clear()



import numpy as np
from collections import defaultdict

class MeanEmbeddingHistory_enhanced_V2:
    def __init__(self, 
                 max_history_per_id=5, 
                 bbox_smooth_alpha=0.2,
                 min_cos_sim=0.65,   # 이 값 미만이면 Outlier로 완전 무시
                 max_cos_sim=0.7,  # 이 값 이상이면 가중치=1로 완전 반영
                 mid_weight=0.5    # 중간 구간일 때 적용할 부분 가중치
                 ):
        """
        각 ID별 임베딩과 bbox 히스토리를 관리, 
        - 평균 임베딩(가중치 기반) 및 
        - 박스 EMA 스무싱,
        - Outlier (다단계) 처리
        를 수행하는 클래스.
        
        Args:
            max_history_per_id (int): ID당 저장할 최대 임베딩 개수
            bbox_smooth_alpha (float): EMA 스무싱 시 사용할 alpha (0~1 사이)
                                       0에 근접 → 과거 박스에 무게
                                       1에 근접 → 현재 박스에 무게
            min_cos_sim (float): 코사인 유사도가 이 값 미만이면 Outlier로 간주(완전히 무시)
            max_cos_sim (float): 코사인 유사도가 이 값 이상이면 정상(가중치=1)
            mid_weight (float): 중간 구간(min_cos_sim~max_cos_sim)에서 부여할 부분 가중치
        """
        self.max_history = max_history_per_id
        
        # 다단계 Outlier 처리를 위한 Threshold
        self.min_cos_sim = min_cos_sim
        self.max_cos_sim = max_cos_sim
        self.mid_weight = mid_weight
        
        # 임베딩 히스토리: (embedding, weight) 쌍을 저장
        self.embedding_data = defaultdict(list)  
        
        # "현재까지 계산된" 가중 평균 임베딩
        self.mean_embeddings = {}  
        
        # 박스 스무싱용
        self.bbox_smooth_alpha = bbox_smooth_alpha
        self.smoothed_bbox = {}  
        self.bbox_history = defaultdict(list)  
        
    def update_bbox(self, track_id, new_bbox):
        """
        ID별 박스 정보를 EMA 방식으로 스무싱하여 업데이트
        Args:
            track_id (int): 트래커 ID
            new_bbox (tuple or list): (x, y, w, h) 형태
        """
        if track_id not in self.smoothed_bbox:
            # 최초 진입 시에는 그대로 사용
            self.smoothed_bbox[track_id] = np.array(new_bbox, dtype=np.float32)
        else:
            # 기존 박스와 EMA
            old_bbox = self.smoothed_bbox[track_id]
            alpha = self.bbox_smooth_alpha
            updated_bbox = alpha * np.array(new_bbox, dtype=np.float32) \
                           + (1 - alpha) * old_bbox
            self.smoothed_bbox[track_id] = updated_bbox
        
        # (필요 시) bbox 히스토리에 저장
        self.bbox_history[track_id].append(self.smoothed_bbox[track_id].copy())
        if len(self.bbox_history[track_id]) > self.max_history:
            self.bbox_history[track_id].pop(0)
        
    def get_smoothed_bbox(self, track_id):
        """스무싱된 bbox 반환"""
        return self.smoothed_bbox.get(track_id, None)
    
    def update_embedding(self, track_id, new_embedding):
        """
        특정 ID의 임베딩 히스토리를 업데이트.
        - 다단계 Outlier 처리 (코사인 유사도 기반)
        - (embedding, weight) 형태로 저장, 가중 평균 재계산
        """
        # 0) L2 정규화
        new_emb_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-6)
        
        # 1) 현재 mean_embedding이 존재하면, 코사인 유사도 계산
        if track_id in self.mean_embeddings:
            mean_emb = self.mean_embeddings[track_id]
            mean_emb_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)
            
            cos_sim = np.dot(mean_emb_norm, new_emb_norm)
            
            # 2) 다단계 처리
            if cos_sim < self.min_cos_sim:
                # 너무 낮음 -> 완전히 무시 (Outlier)
                print(f"[Outlier-SKIP] TrackID={track_id}, cos_sim={cos_sim:.3f} < {self.min_cos_sim}")
                return
            elif cos_sim < self.max_cos_sim:
                # 중간 구간 -> 부분 가중치
                w = self.mid_weight
                print(f"[Outlier-PARTIAL] TrackID={track_id}, cos_sim={cos_sim:.3f}, weight={w}")
            else:
                # 충분히 높음 -> 가중치 1
                w = 1.0
                print(f"[Outlier-ACCEPT] TrackID={track_id}, cos_sim={cos_sim:.3f}, weight={w}")
        else:
            # mean_emb가 아직 없음 -> 첫 임베딩
            w = 1.0
            print(f"[Init Embedding] TrackID={track_id}, weight={w}")
        
        # 3) (임베딩, 가중치) 히스토리에 추가
        self.embedding_data[track_id].append((new_embedding, w))
        
        # 최대 히스토리 초과 시, 가장 오래된 것 제거
        if len(self.embedding_data[track_id]) > self.max_history:
            self.embedding_data[track_id].pop(0)
        
        # 4) 가중 평균 업데이트
        self.mean_embeddings[track_id] = self._compute_weighted_mean(self.embedding_data[track_id])
    
    def _compute_weighted_mean(self, emb_weight_list):
        """
        (embedding, weight) 리스트를 받아 가중 평균을 구한다.
        emb_weight_list: [(np.array(D,), float), ...]
        """
        if len(emb_weight_list) == 0:
            return None
        
        w_sum = 0.0
        emb_sum = None
        for emb, w in emb_weight_list:
            w_sum += w
            if emb_sum is None:
                emb_sum = emb * w
            else:
                emb_sum += emb * w
        
        if w_sum < 1e-6:
            # 가중치 합이 0에 가까우면 None 반환 or 0벡터
            return None
        
        mean_emb = emb_sum / w_sum
        return mean_emb

    def get_mean_embedding(self, track_id):
        """
        특정 ID의 가중 평균 임베딩 반환
        """
        return self.mean_embeddings.get(track_id, None)
    
    def compute_batch_similarity(self, dets_embs, trackers):
        """
        현재 프레임의 검출 임베딩(dets_embs)과 
        트래커들의 '평균 임베딩'(self.mean_embeddings)을 
        코사인 유사도로 계산
        
        dets_embs: shape [N, D]
        trackers: 트래커 객체 리스트. 각 t에 t.id, t.get_emb() 등이 있다고 가정
        """
        if len(trackers) == 0 or dets_embs.size == 0:
            return np.array([])

        print("\n=== 임베딩 유사도 계산 ===")
        print("트래커 수:", len(trackers))
        print("검출 수:", len(dets_embs))

        trk_embs = []
        for t in trackers:
            trk_id = t.id
            mean_emb = self.get_mean_embedding(trk_id)
            
            if mean_emb is None:
                # 히스토리가 전혀 없으면 현재 임베딩 사용 (혹은 0벡터)
                mean_emb = t.get_emb()  
                print(f"트래커 {trk_id}: 히스토리 없음, 현재 임베딩 사용")
            else:
                hist_size = len(self.embedding_data[trk_id])
                print(f"트래커 {trk_id}: 가중평균 임베딩 사용 (히스토리 크기: {hist_size})")
            
            trk_embs.append(mean_emb)
        
        trk_embs = np.array(trk_embs)  # shape [M, D]

        # L2 정규화
        dets_embs_norm = dets_embs / (np.linalg.norm(dets_embs, axis=1, keepdims=True) + 1e-6)
        trk_embs_norm = trk_embs / (np.linalg.norm(trk_embs, axis=1, keepdims=True) + 1e-6)
        
        # 코사인 유사도
        emb_cost = np.dot(dets_embs_norm, trk_embs_norm.T)
        
        return emb_cost
    
    def remove_id(self, track_id):
        """
        특정 ID의 모든 히스토리 제거
        """
        if track_id in self.embedding_data:
            del self.embedding_data[track_id]
        if track_id in self.mean_embeddings:
            del self.mean_embeddings[track_id]
        if track_id in self.smoothed_bbox:
            del self.smoothed_bbox[track_id]
        if track_id in self.bbox_history:
            del self.bbox_history[track_id]
            
    def clear(self):
        """
        모든 히스토리 초기화
        """
        self.embedding_data.clear()
        self.mean_embeddings.clear()
        self.smoothed_bbox.clear()
        self.bbox_history.clear()