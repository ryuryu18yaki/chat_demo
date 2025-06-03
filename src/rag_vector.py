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
        2) preprocess_files 出力 docs をバッチ登録 (重複除去付き)
        3) 必要なら永続化し、Collection オブジェクトを返す
        """
        # — クライアント作成 —
        if persist_directory:
            client = chromadb.Client(Settings(persist_directory=persist_directory))
        else:
            client = chromadb.Client()  # インメモリ

        # — 既存コレクションを削除して新規作成 —
        try:
            client.delete_collection(name=collection_name)
            print(f"🗑️ 既存コレクション削除: {collection_name}")
        except ValueError:
            print(f"📝 新規コレクション作成: {collection_name}")

        collection = client.create_collection(name=collection_name)

        # — docs をバッチ登録 —
        docs_to_index = [d for d in docs if d["metadata"].get("kind") in ("text", "table")]
        print(f"🔍 処理対象ドキュメント: {len(docs_to_index)}")
        
        # 🔥 全体でユニークIDを管理
        global_seen_ids = set()
        processed_docs = []
        
        for doc in docs_to_index:
            metadata = doc["metadata"]
            source = metadata.get('source', 'unknown')
            kind = metadata.get('kind', 'unknown')
            chunk_id = metadata.get('chunk_id', metadata.get('table_id', 0))
            page = metadata.get('page', 'unknown')
            
            # より詳細なID生成
            doc_id = f"{source}-{kind}-p{page}-c{chunk_id}"
            
            # バッチを跨いだ重複チェック
            if doc_id not in global_seen_ids:
                global_seen_ids.add(doc_id)
                processed_docs.append((doc, doc_id))
            else:
                print(f"⚠️ 重複スキップ: {doc_id}")
        
        print(f"✅ 重複除去後: {len(processed_docs)} ドキュメント")
        
        # バッチ処理
        for start in range(0, len(processed_docs), batch_size):
            batch = processed_docs[start:start+batch_size]
            
            embeddings, documents, metadatas, ids = [], [], [], []
            batch_seen_ids = set()  # バッチ内重複チェック
            
            for doc, doc_id in batch:
                # バッチ内でも重複チェック（念のため）
                if doc_id not in batch_seen_ids:
                    try:
                        emb = _embed_text_batch([doc["content"]])[0]
                        embeddings.append(emb)
                        documents.append(doc["content"])
                        metadatas.append(doc["metadata"])
                        ids.append(doc_id)
                        batch_seen_ids.add(doc_id)
                        
                    except Exception as e:
                        print(f"❌ 埋め込み生成エラー (ID: {doc_id}): {e}")
                else:
                    print(f"⚠️ バッチ内重複スキップ: {doc_id}")
            
            # バッチをコレクションに追加
            if embeddings:
                try:
                    collection.add(
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    print(f"✅ バッチ {start//batch_size + 1} 完了: {len(embeddings)} ドキュメント")
                    
                except Exception as e:
                    print(f"❌ バッチ追加エラー: {e}")
                    # 個別追加を試行
                    for i, (emb, doc, meta, doc_id) in enumerate(zip(embeddings, documents, metadatas, ids)):
                        try:
                            collection.add(
                                embeddings=[emb],
                                documents=[doc],
                                metadatas=[meta],
                                ids=[doc_id]
                            )
                        except Exception as e2:
                            print(f"❌ 個別追加失敗 (ID: {doc_id}): {e2}")
            
        # 永続化
        if persist_directory:
            try:
                client.persist()
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
    # 4) dict リストに整形
    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append({
            "content": doc,
            "metadata": meta,
            "distance": dist,
        })
    return hits