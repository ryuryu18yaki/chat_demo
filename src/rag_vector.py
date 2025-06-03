# src/rag_vector.py
from __future__ import annotations
from io import BytesIO
from typing import List, Dict, Any
import uuid
import hashlib

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

# 🔥 新機能: 安全なID生成
def generate_safe_id(metadata: Dict[str, Any], content: str) -> str:
    """メタデータとコンテンツから安全で一意なIDを生成"""
    source = metadata.get('source', 'unknown')
    kind = metadata.get('kind', 'unknown')
    page = metadata.get('page', 0)
    chunk_id = metadata.get('chunk_id', metadata.get('table_id', 0))
    
    # コンテンツハッシュを生成
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    
    # 安全なID文字列を生成（ChromaDBで許可される文字のみ）
    safe_id = f"{source}_{kind}_p{page}_c{chunk_id}_{content_hash}"
    
    # ChromaDBで問題となる文字を置換
    safe_id = safe_id.replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_")
    
    return safe_id

# ---------------------------------------------------------------------------
# ドキュメントを ChromaDB に保存（修正版）
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
    2) preprocess_files 出力 docs をバッチ登録 (重複除去付き)
    3) 必要なら永続化し、Collection オブジェクトを返す
    
    🔥 修正点:
    - ユニークで安全なID生成
    - 重複チェック機能
    - エラーハンドリング強化
    """
    # — クライアント作成 —
    if persist_directory:
        client = chromadb.Client(Settings(persist_directory=persist_directory))
    else:
        client = chromadb.Client()  # インメモリ

    # — 既存コレクションを削除（重複を避けるため）—
    try:
        client.delete_collection(name=collection_name)
    except ValueError:
        pass  # コレクションが存在しない場合は無視

    # — 新しいコレクション作成 —
    collection = client.create_collection(name=collection_name)

    # — docs をバッチ登録 —
    docs_to_index = [d for d in docs if d["metadata"].get("kind") in ("text", "table")]
    
    # 重複除去用セット
    seen_ids = set()
    processed_docs = []
    
    print(f"🔍 処理対象ドキュメント数: {len(docs_to_index)}")
    
    # 重複除去
    for doc in docs_to_index:
        doc_id = generate_safe_id(doc["metadata"], doc["content"])
        
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            processed_docs.append((doc, doc_id))
        else:
            print(f"⚠️ 重複スキップ: {doc_id}")
    
    print(f"✅ 重複除去後: {len(processed_docs)} ドキュメント")
    
    # バッチ処理
    for start in range(0, len(processed_docs), batch_size):
        batch = processed_docs[start:start+batch_size]
        
        embeddings, documents, metadatas, ids = [], [], [], []
        
        for doc, doc_id in batch:
            try:
                # 埋め込み生成
                emb = _embed_text_batch([doc["content"]])[0]
                
                embeddings.append(emb)
                documents.append(doc["content"])
                metadatas.append(doc["metadata"])
                ids.append(doc_id)  # 🔥 計算されたIDを使用
                
            except Exception as e:
                print(f"❌ 埋め込み生成エラー (ID: {doc_id}): {e}")
                continue
        
        # バッチをコレクションに追加
        if embeddings:  # 空でない場合のみ
            try:
                collection.add(  # upsertではなくaddを使用
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✅ バッチ {start//batch_size + 1} 完了: {len(embeddings)} ドキュメント")
                
            except Exception as e:
                print(f"❌ バッチ追加エラー: {e}")
                # 個別に追加を試行
                for i, (emb, doc, meta, doc_id) in enumerate(zip(embeddings, documents, metadatas, ids)):
                    try:
                        collection.add(
                            embeddings=[emb],
                            documents=[doc],
                            metadatas=[meta],
                            ids=[doc_id]
                        )
                    except Exception as e2:
                        print(f"❌ 個別追加エラー (ID: {doc_id}): {e2}")
        
    # 永続化
    if persist_directory:
        try:
            client.persist()
            print("💾 永続化完了")
        except Exception as e:
            print(f"⚠️ 永続化エラー: {e}")
    
    final_count = collection.count()
    print(f"🎯 最終コレクション数: {final_count}")
    
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
    ids = res.get("ids", [None] * len(documents))[0]  # IDも取得
    
    # 4) dict リストに整形
    hits: List[Dict[str, Any]] = []
    for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
        hits.append({
            "content": doc,
            "metadata": meta,
            "distance": dist,
            "id": doc_id,  # IDも含める
        })
    return hits