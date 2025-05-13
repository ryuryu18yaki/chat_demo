# src/rag_vector.py
from __future__ import annotations
from io import BytesIO
from typing import List, Dict, Any
import uuid

# SQLite バージョン強制上書き（pysqlite3でchromadbが使えるようにする）
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import torch
from uuid import uuid4

__all__ = ["save_docs_to_chroma", "query_collection"]

# ---------------------------------------------------------------------------
# モデル＆クライアント初期化
# ---------------------------------------------------------------------------
_OPENAI_MODEL = "text-embedding-ada-002"
_openai_client = OpenAI()

# _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# _clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------------------------------------------------------------------
# テキスト埋め込み（バッチ）
# ---------------------------------------------------------------------------
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def _embed_text_batch(texts: List[str]) -> List[List[float]]:
    resp = _openai_client.embeddings.create(model=_OPENAI_MODEL, input=texts)
    return [item.embedding for item in resp.data]

# # ---------------------------------------------------------------------------
# # 画像埋め込み（CLIP）
# # ---------------------------------------------------------------------------
# def _embed_image(img_bytes: bytes) -> List[float]:
#     img = Image.open(BytesIO(img_bytes)).convert("RGB")
#     inputs = _clip_proc(images=img, return_tensors="pt")
#     with torch.no_grad():
#         feats = _clip_model.get_image_features(**inputs)
#     feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
#     return feats[0].cpu().tolist()

# ---------------------------------------------------------------------------
# ドキュメントを ChromaDB に保存
# ---------------------------------------------------------------------------
def save_docs_to_chroma(
    *,
    docs: List[Dict[str, Any]],
    collection_name: str,
    persist_directory: str | None = None,
    batch_size: int = 50,
) -> chromadb.api.Collection:
    """
    1) ChromaDB のコレクションを作成・取得
    2) preprocess_files 出力 docs をバッチ登録 (ids付き)
    3) 必要なら永続化し、Collection オブジェクトを返す
    """
    # — クライアント作成 —
    if persist_directory:
        client = chromadb.Client(Settings(persist_directory=persist_directory))
    else:
        client = chromadb.Client()  # インメモリ

    # — コレクション取得 or 作成 —
    collection = client.get_or_create_collection(name=collection_name)

    # — docs をバッチ登録 —
    docs_to_index = [d for d in docs if d["metadata"].get("kind") in ("text", "table")]
    for start in range(0, len(docs_to_index), batch_size):
        batch = docs_to_index[start:start+batch_size]
        embeddings, documents, metadatas, ids = [], [], [], []
        for doc in batch:
            emb = _embed_text_batch([doc["content"]])[0]
            embeddings.append(emb)
            documents.append(doc["content"])
            metadata = doc["metadata"]
            metadatas.append(metadata)
            key = f"{metadata['source']}-{metadata['kind']}-{metadata.get('chunk_id', metadata.get('table_id',''))}"
            ids.append(str(uuid.uuid4()))
        collection.upsert(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        
    if persist_directory:
        client.persist()
    return collection

# ---------------------------------------------------------------------------
# コレクション検索ヘルパー
# ---------------------------------------------------------------------------
def query_collection(
    collection: chromadb.api.Collection,
    query: str,
    n_results: int = 5,
) -> List[Dict[str, Any]]:

    """
    テキストクエリを埋め込み検索し、上位 n_results 件を dict のリストで返す。
    """
    # 1) クエリを埋め込み
    q_emb = _embed_text_batch([query])[0]
    # 2) 検索実行
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
    )
    # 3) 結果を抽出
    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]
    # 4) dict リストに整形
    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append({
            "content": doc,
            "metadata": meta,
            "distance": dist,
        })
    return hits