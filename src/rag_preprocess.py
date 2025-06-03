# src/rag_preprocess.py

from __future__ import annotations
from io import BytesIO
from typing import List, Dict, Any
import hashlib

import pdfplumber       # pip install pdfplumber
from pdfminer.high_level import extract_text  # type: ignore
from pdfminer.layout import LAParams         # type: ignore
from PIL import Image

__all__ = [
    "extract_text_from_pdf",
    "extract_text_from_txt",
    "chunk_text",
    "extract_tables_from_pdf",
    "extract_images_from_pdf",
    "preprocess_files",
]

# ---------------------------------------------------------------------------
# 1) テキスト抽出
# ---------------------------------------------------------------------------
def extract_text_from_pdf(data: bytes) -> str:
    """PDF バイナリから全文テキストを取得する。"""
    with BytesIO(data) as buf:
        laparams = LAParams()
        text = extract_text(buf, laparams=laparams)
    return text

def extract_text_from_txt(data: bytes, encoding: str | None = None) -> str:
    """TXT バイナリを文字列へデコードする。"""
    if encoding is None:
        for enc in ("utf-8", "shift_jis", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("Failed to decode text file with common encodings")
    return data.decode(encoding)

# 🔥 修正版: ページ別テキスト抽出
def extract_text_from_pdf_by_pages(data: bytes) -> List[Dict[str, Any]]:
    """PDFからページ別にテキストを抽出"""
    pages_text = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():  # 空ページをスキップ
                pages_text.append({
                    "text": page_text,
                    "page": page_num
                })
    return pages_text

# ---------------------------------------------------------------------------
# 2) チャンク化ユーティリティ（修正版）
# ---------------------------------------------------------------------------
def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 80) -> List[str]:
    """テキストを重複付きで分割する。（オーバーラップを10%に調整）"""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    
    # 空のテキストや短いテキストの場合
    if not text.strip() or len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        
        if chunk:  # 空のチャンクをスキップ
            chunks.append(chunk)
        
        if end == len(text):
            break
        start = end - overlap
    
    return chunks

# 🔥 新機能: ユニークID生成
def generate_chunk_id(source: str, page: int, chunk_index: int, content: str) -> str:
    """ユニークなチャンクIDを生成"""
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{source}_p{page}_{chunk_index}_{content_hash}"

# ---------------------------------------------------------------------------
# 3) 表の抽出
# ---------------------------------------------------------------------------
def extract_tables_from_pdf(data: bytes) -> List[Dict[str, Any]]:
    """pdfplumber で表を抽出し、CSV ライクな文字列で返す。"""
    tables: List[Dict[str, Any]] = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for tbl_idx, table in enumerate(page.extract_tables(), start=1):
                # None を "" に置き換えてから結合
                lines: list[str] = []
                for row in table:
                    cells = [(cell if cell is not None else "") for cell in row]
                    lines.append(",".join(cells))
                csv_text = "\n".join(lines)
                
                if csv_text.strip():  # 空の表をスキップ
                    tables.append({
                        "text": csv_text,
                        "page": page_num,
                        "table_id": tbl_idx,
                    })
    return tables

# ---------------------------------------------------------------------------
# 4) 画像の抽出
# ---------------------------------------------------------------------------
def extract_images_from_pdf(data: bytes) -> List[Dict[str, Any]]:
    """pdfplumber で埋め込み画像を抽出し、バイト＋メタデータで返す。"""
    imgs: List[Dict[str, Any]] = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # ① ページ全体を PIL Image として取得
            page_img = page.to_image(resolution=150)
            pil_page = page_img.original
            for img_idx, img_meta in enumerate(page.images, start=1):
                x0, top, x1, bottom = (
                    img_meta["x0"], img_meta["top"],
                    img_meta["x1"], img_meta["bottom"]
                )
                # ② PIL Image に対して crop
                cropped = pil_page.crop((x0, top, x1, bottom))
                # PIL で PNG にエンコード
                buf = BytesIO()
                cropped.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                imgs.append({
                    "bytes": img_bytes,
                    "page": page_num,
                    "image_id": img_idx,
                    "width": cropped.width,
                    "height": cropped.height,
                })
    return imgs

# ---------------------------------------------------------------------------
# 5) メイン: ファイル→チャンク辞書リスト（大幅修正）
# ---------------------------------------------------------------------------
def preprocess_files(
    files: List[Dict[str, Any]],
    *,
    chunk_size: int = 800,
    overlap: int = 80,  # 10%のオーバーラップ
) -> List[Dict[str, Any]]:
    """
    files: List of {"name": str, "type": mime, "data": bytes}
    を受け取り、kind(text/table/image)ごとに docs を返す。
    
    🔥 修正点:
    - ページ別処理でメタデータの重複を防ぐ
    - ユニークIDの生成
    - 重複チャンクの除去
    """
    docs: List[Dict[str, Any]] = []
    seen_content = set()  # 重複除去用

    for f in files:
        name, mime, data = f["name"], f["type"], f["data"]
        
        # ファイルハッシュ（デバッグ用）
        file_hash = hashlib.md5(data).hexdigest()[:8]

        # 🔥 表のチャンク化（修正）
        if name.lower().endswith(".pdf"):
            tables = extract_tables_from_pdf(data)
            for tbl in tables:
                content = tbl["text"]
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                if content_hash not in seen_content:
                    unique_id = f"{name}_p{tbl['page']}_table{tbl['table_id']}_{content_hash[:8]}"
                    docs.append({
                        "content": content,
                        "metadata": {
                            "source": name,
                            "kind": "table",
                            "page": tbl["page"],
                            "table_id": tbl["table_id"],
                            "unique_id": unique_id,
                            "file_hash": file_hash,
                        }
                    })
                    seen_content.add(content_hash)

        # 🔥 プレーンテキストのチャンク化（ページ別処理）
        if mime == "text/plain" or name.lower().endswith(".txt"):
            # テキストファイルの場合（ページなし）
            text = extract_text_from_txt(data)
            if text.strip():
                for chunk_idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
                    content_hash = hashlib.md5(chunk.encode()).hexdigest()
                    
                    if content_hash not in seen_content:
                        unique_id = generate_chunk_id(name, 1, chunk_idx, chunk)
                        docs.append({
                            "content": chunk,
                            "metadata": {
                                "source": name,
                                "kind": "text",
                                "page": 1,  # テキストファイルは1ページとして扱う
                                "chunk_id": chunk_idx,
                                "unique_id": unique_id,
                                "file_hash": file_hash,
                            }
                        })
                        seen_content.add(content_hash)
                        
        elif mime == "application/pdf" or name.lower().endswith(".pdf"):
            # 🔥 PDFの場合：ページ別処理
            pages_data = extract_text_from_pdf_by_pages(data)
            
            for page_data in pages_data:
                page_num = page_data["page"]
                page_text = page_data["text"]
                
                if page_text.strip():  # 空ページをスキップ
                    chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        content_hash = hashlib.md5(chunk.encode()).hexdigest()
                        
                        if content_hash not in seen_content:
                            unique_id = generate_chunk_id(name, page_num, chunk_idx, chunk)
                            docs.append({
                                "content": chunk,
                                "metadata": {
                                    "source": name,
                                    "kind": "text",
                                    "page": page_num,  # 🔥 正確なページ番号
                                    "chunk_id": chunk_idx,
                                    "unique_id": unique_id,
                                    "file_hash": file_hash,
                                }
                            })
                            seen_content.add(content_hash)

    return docs