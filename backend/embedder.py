"""嵌入模型封装。使用 fastembed (ONNX 运行时)，轻量无 PyTorch。"""
import os
import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_DIM = 512

# 缓存目录：优先用 HF_HOME，否则用 fastembed 默认（系统临时目录）
_CACHE_DIR = os.environ.get("HF_HOME")

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        kwargs = {"model_name": MODEL_NAME}
        if _CACHE_DIR:
            kwargs["cache_dir"] = _CACHE_DIR
        _embedder = TextEmbedding(**kwargs)
    return _embedder


def embed_texts(texts: list[str]) -> np.ndarray:
    """将文本列表转为向量数组，shape=(len(texts), VECTOR_DIM)，已归一化。"""
    if not texts:
        return np.empty((0, VECTOR_DIM), dtype=np.float32)

    model = _get_embedder()
    embeddings = list(model.embed(texts))
    result = np.array(embeddings, dtype=np.float32)
    # 归一化（FAISS 内积搜索 = 余弦相似度）
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return result / norms


def embed_query(query: str) -> np.ndarray:
    """将单个查询文本转为向量，shape=(VECTOR_DIM,)。"""
    return embed_texts([query])[0]
