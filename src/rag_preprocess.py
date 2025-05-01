from __future__ import annotations

from io import BytesIO
from typing import List, Dict, Any

# PDF テキスト抽出用（MIT License）
from pdfminer.high_level import extract_text  # type: ignore
from pdfminer.layout import LAParams  # type: ignore

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_txt",
    "chunk_text",
    "preprocess_files",
]

# ---------------------------------------------------------------------------
# 基本抽出関数
# ---------------------------------------------------------------------------

def extract_text_from_pdf(data: bytes) -> str:
    """PDF バイナリから全文テキストを取得する。

    Args:
        data: PDF ファイルのバイナリデータ。
    Returns:
        抽出したテキスト文字列。
    """
    with BytesIO(data) as buf:
        laparams = LAParams()  # デフォルト設定で十分
        text = extract_text(buf, laparams=laparams)
    return text


def extract_text_from_txt(data: bytes, encoding: str | None = None) -> str:
    """TXT バイナリを文字列へ。

    Args:
        data: テキストファイルのバイナリデータ。
        encoding: 明示的に指定する場合はエンコーディング名。
    Returns:
        デコード済み文字列。
    """
    if encoding is None:
        # 最低限の簡易判定（UTF‑8 → Shift‑JIS → Latin‑1）
        for enc in ("utf-8", "shift_jis", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("Failed to decode text file with common encodings")
    return data.decode(encoding)

# ---------------------------------------------------------------------------
# チャンク化ユーティリティ
# ---------------------------------------------------------------------------

def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """テキストを重複付きで分割するシンプル実装。

    Args:
        text: 入力テキスト。
        chunk_size: 1 チャンクあたりの最大文字数。
        overlap: 隣接チャンク同士の重複文字数。
    Returns:
        チャンク文字列のリスト。
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break                      # ← 最後まで到達したら終了
        start = end - overlap          # 次は overlap だけ戻る
    return chunks

# ---------------------------------------------------------------------------
# メイン: アップロードファイル → チャンク化テキスト
# ---------------------------------------------------------------------------

def preprocess_files(
    files: List[Dict[str, Any]],
    *,
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """`st.session_state.rag_files` 形式のリストを受け取り、
    チャンクごとにテキスト + メタデータ辞書を返す。

    メタデータには `source` と `chunk_id` を含め、後工程（埋め込み / 検索）で
    どのファイル由来かトレースできるようにする。
    """
    docs: List[Dict[str, Any]] = []

    for f in files:
        name: str = f["name"]
        mime: str = f["type"]
        data: bytes = f["data"]

        if mime == "text/plain" or name.lower().endswith(".txt"):
            text = extract_text_from_txt(data)
        elif mime == "application/pdf" or name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(data)
        else:
            # 画像やその他はここではスキップ（次フェーズで OCR を追加予定）
            continue

        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            docs.append({
                "content": chunk,
                "metadata": {
                    "source": name,
                    "chunk_id": idx,
                },
            })
    return docs
