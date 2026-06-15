"""嵌入模型封装。索引脚本和后端共用此模块，保证向量维度一致。"""
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_DIM = 512  # BGE-small-zh-v1.5 输出维度

_model = None


def _get_model():
    """延迟加载模型（单例）。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """将文本列表转为向量数组，shape=(len(texts), VECTOR_DIM)，已归一化。"""
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """将单个查询文本转为向量，shape=(VECTOR_DIM,)，已归一化。"""
    model = _get_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embedding, dtype=np.float32)
