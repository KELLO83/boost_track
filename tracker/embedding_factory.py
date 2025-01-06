from typing import Dict, Type
from .embedding_history import EmbeddingHistory, MeanEmbeddingHistory, TemplateEmbeddingHistory, EnhancedTemplateEmbeddingHistory

class EmbeddingHistoryFactory:
    _registry: Dict[str, Type[EmbeddingHistory]] = {
        'default': EmbeddingHistory,
        'mean': MeanEmbeddingHistory,
        'template': TemplateEmbeddingHistory,
        'enhanced': EnhancedTemplateEmbeddingHistory
    }

    @classmethod
    def register(cls, name: str, embedding_class: Type[EmbeddingHistory]) -> None:
        """새로운 임베딩 방법을 등록"""
        cls._registry[name] = embedding_class

    @classmethod
    def create(cls, method: str, **kwargs) -> EmbeddingHistory:
        """지정된 방법의 임베딩 히스토리 인스턴스 생성"""
        if method not in cls._registry:
            raise ValueError(f"Unknown embedding method: {method}")
        return cls._registry[method](**kwargs)

    @classmethod
    def get_available_methods(cls) -> list:
        """사용 가능한 임베딩 방법 목록 반환"""
        return list(cls._registry.keys())
