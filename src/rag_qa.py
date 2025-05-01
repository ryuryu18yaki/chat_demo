from __future__ import annotations

from typing import List, Dict, Any, Optional
from base64 import b64encode
from openai import OpenAI
from src.rag_vector import query_collection

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _build_prompt(question: str, contexts: List[str]) -> List[Dict[str, str]]:
    """system + user メッセージの形でプロンプトを組み立てる。"""
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the provided context to answer the "
            "user's question. If the context is insufficient, say you don't know."
        ),
    }

    context_block = "\n\n".join(f"[Doc {i+1}]\n{ctx}" for i, ctx in enumerate(contexts))
    user_content = (
        "Context:\n" + context_block + "\n\n"
        "Question: " + question + "\n\nAnswer in Japanese."
    )
    user_msg = {"role": "user", "content": user_content}
    return [system_msg, user_msg]


# ---------------------------------------------------------------------------
# メイン関数
# ---------------------------------------------------------------------------

def generate_answer(
    *,
    prompt: str,
    question: str,
    collection,
    top_k: int = 5,
    model: str = _DEFAULT_MODEL,
    max_context_chars: int = 6000,
    image_bytes: Optional[bytes] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """ベクトル DB から関連文書を取り出し、GPT で回答を生成する。

    Returns:
        {
            "answer": str,            # 生成した回答
            "sources": List[Dict],    # query_collection と同形式
        }
    """
    # 1) ベクトル検索
    hits = query_collection(collection=collection, query_text=question, top_k=top_k)

    # 2) 長すぎる場合はカット
    contexts: List[str] = []
    total_chars = 0
    for h in hits:
        if total_chars + len(h["content"]) > max_context_chars:
            break
        contexts.append(h["content"])
        total_chars += len(h["content"])

    # 3) プロンプト組み立て（画像 & 履歴を追加）
    base_msgs = _build_prompt(question, contexts)

    # ---- 画像パートを付与 ----
    if image_bytes:
        data_url = "data:image/png;base64," + b64encode(image_bytes).decode("utf-8")
        img_part = {"type": "image_url", "image_url": {"url": data_url}}
        text_part = {"type": "text", "text": base_msgs[1]["content"]}
        base_msgs[1]["content"] = [img_part, text_part]

    # ---- 過去履歴を先頭に挿入 ----
    if chat_history:
        messages = (
            [{"role": "system", "content": prompt}]
            + chat_history
            + base_msgs
        )
    else:
        messages = base_msgs
    if image_bytes:
        data_url = "data:image/png;base64," + b64encode(image_bytes).decode("utf-8")
        # user メッセージを list[dict] 形式へ差し替え
        img_part = {"type": "image_url", "image_url": {"url": data_url}}
        text_part = {"type": "text", "text": messages[1]["content"]}
        messages[1]["content"] = [img_part, text_part]

    # 4) OpenAI Chat 完了
    client = OpenAI()
    resp = client.chat.completions.create(model=model, messages=messages)
    answer = resp.choices[0].message.content

    return {"answer": answer, "sources": hits}
