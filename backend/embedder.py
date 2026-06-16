"""嵌入模型封装。

优先使用 HuggingFace 免费推理 API（无需本地模型下载，适配 Render 免费层）。
本地运行时自动使用 fastembed ONNX 模型（更快，无网络延迟）。
"""
import os
import numpy as np
import requests

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_DIM = 512

# HuggingFace 免费推理 API
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# 是否在本地环境（有 fastembed 可用时优先本地）
_use_local = None


def _check_local():
    global _use_local
    if _use_local is None:
        try:
            from fastembed import TextEmbedding  # noqa: F401
            _use_local = True
        except ImportError:
            _use_local = False
    return _use_local


# ── 本地模式 (fastembed ONNX) ──────────────────────────
_local_model = None


def _embed_local(texts: list[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding
        cache_dir = os.environ.get("HF_HOME")
        kwargs = {"model_name": MODEL_NAME}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        _local_model = TextEmbedding(**kwargs)
    embeddings = list(_local_model.embed(texts))
    result = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return result / norms


# ── 远程模式 (HuggingFace API) ─────────────────────────
def _embed_remote(texts: list[str]) -> np.ndarray:
    """通过 HF 免费推理 API 获取嵌入向量。"""
    resp = requests.post(
        HF_API_URL,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=30,
    )
    if resp.status_code == 503:
        # 模型正在加载，稍等后重试
        resp = requests.post(
            HF_API_URL,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )
    resp.raise_for_status()
    result = np.array(resp.json(), dtype=np.float32)
    # HF API 返回未归一化向量，手动归一化
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return result / norms


# ── 公共接口 ───────────────────────────────────────────
def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, VECTOR_DIM), dtype=np.float32)

    if _check_local():
        return _embed_local(texts)
    else:
        return _embed_remote(texts)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
