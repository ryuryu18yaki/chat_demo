from __future__ import annotations

from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.rag_vector import query_collection
from src.rag_preprocess import extract_images_from_pdf
from base64 import b64encode

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# プロンプト組み立て
# ---------------------------------------------------------------------------

def _build_prompt(question: str, contexts: List[str]) -> List[Dict[str, str]]:
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the provided context to answer the "
            "user's question. If the context is insufficient, say you don't know."
        ),
    }
    context_block = "\n\n".join(f"[Doc {i+1}]\n{ctx}" for i, ctx in enumerate(contexts))
    user_msg = {
        "role": "user",
        "content": (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\nAnswer in Japanese."
        ),
    }
    return [system_msg, user_msg]

# ---------------------------------------------------------------------------
# 回答生成
# ---------------------------------------------------------------------------

def generate_answer(
        *,
        prompt: str,
        question: str,
        collection,
        rag_files: List[Dict[str, Any]],
        top_k: int = 5,
        model: str = _DEFAULT_MODEL,
        max_context_chars: int = 6000,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        1) テキストのみでベクトル検索
        2) ヒットしたテキスト/表チャンクを contexts に集約
        3) 同ページの画像を PDF から切り出して files に追加
        4) GPT-4V へマルチモーダル入力として渡し回答を生成
        """
        client = OpenAI()

        # --- 1) テキスト/表のみを検索 ---
        hits = query_collection(
            collection=collection,
            query=question,
            n_results=top_k,
        )
        
        # --- 2) contexts を収集 ---
        contexts: List[str] = []
        total_len = 0
        for hit in hits:
            meta = hit.get("metadata", {})
            kind = meta.get("kind", "text")
            if kind not in ("text", "table"):
                continue
            content = hit.get("content", "")
            if total_len + len(content) > max_context_chars:
                break
            contexts.append(content)
            total_len += len(content)
        
        # --- 3) 画像を抽出 ---
        files: List[Dict[str, bytes]] = []
        placeholders: List[str] = []
        images_info: List[Dict[str, Any]] = []  # 🔥 画像情報を保存
        
        for hit in hits:
            meta = hit.get("metadata", {})
            if meta.get("kind") not in ("text", "table"):
                continue
            source = meta.get("source")
            page = meta.get("page")
            if not source or not page:
                continue
            # アップロード済みの PDF から抽出
            for f in rag_files:
                if f["name"] == source:
                    for img in extract_images_from_pdf(f["data"]):
                        if img["page"] == page:
                            # プレースホルダを追加
                            idx = len(placeholders) + 1
                            placeholders.append(f"[Image {idx}: {source} p{page} id={img['image_id']}]" )
                            files.append({
                                "name": f"{source}_p{page}_{img['image_id']}.png",
                                "data": img["bytes"],
                            })
                            
                            # 🔥 画像情報を保存（表示用）
                            images_info.append({
                                "name": f"{source}_p{page}_{img['image_id']}.png",
                                "data": img["bytes"],
                                "source": source,
                                "page": page,
                                "image_id": img["image_id"]
                            })
        
        # 画像のプレースホルダを contexts に追加
        contexts.extend(placeholders)
        
        # --- 4) プロンプト組み立て ---
        system_msg, user_msg = _build_prompt(question, contexts)
        
        # 画像を含むマルチモーダル user メッセージ形式に変換
        parts: List[Dict[str, Any]] = []
        for idx, file in enumerate(files, start=1):
            data_url = "data:image/png;base64," + b64encode(file["data"]).decode("utf-8")
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        # 最後にテキスト部分を追加
        parts.append({"type": "text", "text": user_msg["content"]})
        user_msg["content"] = parts
        
        # --- 5) Messages 組み立て ---
        messages: List[Dict[str, Any]] = []
        if chat_history:
            messages.append({"role": "system", "content": prompt})
            messages.extend(chat_history)
        messages.append(system_msg)
        messages.append(user_msg)
        
        # --- 6) API呼び出しパラメータを構築 ---
        params = {
            "model": model,
            "messages": messages,
        }
        
        # temperatureとmax_tokensが指定されている場合のみ追加
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # --- 7) GPT-4V 呼び出し ---
        resp = client.chat.completions.create(**params)
        
        # 🔥 画像情報も返すように修正
        return {
            "answer": resp.choices[0].message.content,
            "sources": hits,
            "images": images_info  # 画像情報を追加
        }