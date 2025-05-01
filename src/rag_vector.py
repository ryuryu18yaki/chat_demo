from __future__ import annotations

import uuid
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

__all__ = ["save_docs_to_chroma", "query_collection"]

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

_OPENAI_MODEL = "text-embedding-3-small"
_openai_client = OpenAI()


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def _embed_text_batch(texts: List[str]) -> List[List[float]]:
    """OpenAI Embedding API を呼び出し、ベクトルを返す (再試行付き)。"""
    resp = _openai_client.embeddings.create(
        model=_OPENAI_MODEL,
        input=texts,
    )
    # API は順序保証
    return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# Chroma helper
# ---------------------------------------------------------------------------

def _get_chroma_client(persist_directory: str | None = None) -> chromadb.ClientAPI:
    if persist_directory is None:
        return chromadb.Client(Settings())  # in‑memory
    return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))


def save_docs_to_chroma(
    *,
    docs: List[Dict[str, Any]],
    collection_name: str,
    persist_directory: str | None = None,
    batch_size: int = 100,
) -> chromadb.api.models.Collection.Collection:
    """チャンク済みドキュメントを Chroma に upsert して Collection を返す。

    Args:
        docs: preprocess_files で得た辞書リスト。
        collection_name: 保存するコレクション名。
        persist_directory: `None` ならインメモリ、パス指定で永続化。
        batch_size: Embedding API 1 呼び出しあたりの件数。
    """
    chroma_client = _get_chroma_client(persist_directory)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # 既存 ID を取得して重複登録を防ぐ
    existing_count = collection.count()

    documents, metadatas, ids = [], [], []
    for doc in docs:
        doc_id = str(uuid.uuid4())
        documents.append(doc["content"])
        metadatas.append(doc["metadata"])
        ids.append(doc_id)

        # バッチサイズに達したらまとめて送信
        if len(documents) >= batch_size:
            _upsert_batch(collection, documents, metadatas, ids)
            documents, metadatas, ids = [], [], []

    # 端数
    if documents:
        _upsert_batch(collection, documents, metadatas, ids)

    # 永続化先がある場合は保存
    if persist_directory is not None:
        chroma_client.persist()

    new_total = collection.count()
    print(f"Chroma collection '{collection_name}': {existing_count} -> {new_total} records")
    return collection


def _upsert_batch(collection, documents, metadatas, ids):
    embeddings = _embed_text_batch(documents)
    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)


# ---------------------------------------------------------------------------
# 検索ユーティリティ
# ---------------------------------------------------------------------------

def query_collection(
    *,
    collection: chromadb.api.models.Collection.Collection,
    query_text: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """コレクションを検索し、上位 `top_k` 件を返す。"""
    query_embed = _embed_text_batch([query_text])[0]
    res = collection.query(query_embeddings=[query_embed], n_results=top_k)

    results: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({
            "content": doc,
            "metadata": meta,
            "distance": dist,
        })
    return results
