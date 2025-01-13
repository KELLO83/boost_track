from typing import Dict, Type
from .embedding_history import EmbeddingHistory, MeanEmbeddingHistory, TemplateEmbeddingHistory, EnhancedTemplateEmbeddingHistory, MeanEmbeddingHistory_enhanced

class EmbeddingHistoryFactory:
    _registry: Dict[str, Type[EmbeddingHistory]] = {
        'default': EmbeddingHistory,
        'mean': MeanEmbeddingHistory,
        'template': TemplateEmbeddingHistory,
        'enhanced': EnhancedTemplateEmbeddingHistory,
        'enhanced_mean': MeanEmbeddingHistory_enhanced,
        'enhanced_mean_V2': MeanEmbeddingHistory_enhanced
    }

    @classmethod
    def register(cls, name: str, embedding_class: Type[EmbeddingHistory]) -> None:
        """새로운 임베딩 방법을 등록"""
        cls._registry[name] = embedding_class

    @classmethod
    def create(cls, method: str, **kwargs) -> EmbeddingHistory:
        """
        지정된 방법의 임베딩 히스토리 인스턴스 생성
        
        Args:
            method (str): 임베딩 방법 선택
                - 'default': 가장 최근 임베딩만 사용
                - 'mean': 평균 임베딩 사용
                - 'template': 템플릿 임베딩 사용
                - 'enhanced': 향상된 템플릿 임베딩 (outlier 제거, bbox 스무싱 포함)
                - 'enhanced_mean': 향상된 평균 임베딩 (outlier 제거, bbox 스무싱 포함)
                - 'enhanced_mean_V2': 향상된 평균 임베딩 V2 (outlier 제거, bbox 스무싱 포함)
        """
        if method not in cls._registry:
            raise ValueError(f"Unknown embedding method: {method}")
        print("=> Using embedding method:", method)
        
        return cls._registry[method](**kwargs)

    @classmethod
    def get_available_methods(cls) -> list:
        """사용 가능한 임베딩 방법 목록 반환"""
        return list(cls._registry.keys())
