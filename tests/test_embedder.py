"""测试嵌入模型封装。需先安装 backend/requirements.txt 的依赖。"""
import sys
sys.path.insert(0, "backend")

import numpy as np
from embedder import embed_texts, embed_query, VECTOR_DIM


def test_embed_texts_shape():
    """embed_texts 返回正确形状的数组。"""
    texts = ["你好世界", "Hello world"]
    result = embed_texts(texts)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, VECTOR_DIM)
    assert result.dtype == np.float32


def test_embed_texts_normalized():
    """嵌入向量已归一化（L2 范数 ≈ 1）。"""
    result = embed_texts(["测试文本"])
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 0.01


def test_embed_query_shape():
    """embed_query 返回正确形状。"""
    result = embed_query("这是一个问题")
    assert isinstance(result, np.ndarray)
    assert result.shape == (VECTOR_DIM,)
    assert result.dtype == np.float32


def test_embed_query_normalized():
    """查询向量已归一化。"""
    result = embed_query("查询")
    norm = np.linalg.norm(result)
    assert abs(norm - 1.0) < 0.01


def test_embeddings_are_reproducible():
    """同一文本两次嵌入结果一致。"""
    a = embed_query("同一个问题")
    b = embed_query("同一个问题")
    np.testing.assert_array_almost_equal(a, b)


def test_semantic_distance():
    """语义相似的文本向量距离更近。"""
    a = embed_query("生成视频失败")
    b = embed_query("seedance 报错")
    c = embed_query("今天天气很好")
    dist_ab = np.linalg.norm(a - b)
    dist_ac = np.linalg.norm(a - c)
    assert dist_ab < dist_ac
