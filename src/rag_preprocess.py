# src/rag_preprocess.py

from __future__ import annotations
from io import BytesIO
from typing import List, Dict, Any

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

# ---------------------------------------------------------------------------
# 2) チャンク化ユーティリティ
# ---------------------------------------------------------------------------
def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """テキストを重複付きで分割する。"""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

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
# 5) メイン: ファイル→チャンク辞書リスト
# ---------------------------------------------------------------------------
def preprocess_files(
    files: List[Dict[str, Any]],
    *,
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    files: List of {"name": str, "type": mime, "data": bytes}
    を受け取り、kind(text/table/image)ごとに docs を返す。
    """
    docs: List[Dict[str, Any]] = []

    for f in files:
        name, mime, data = f["name"], f["type"], f["data"]

        # — 表のチャンク化 —
        if name.lower().endswith(".pdf"):
            for tbl in extract_tables_from_pdf(data):
                docs.append({
                    "content": tbl["text"],
                    "metadata": {
                        "source":    name,
                        "kind":      "table",
                        "page":      tbl["page"],
                        "table_id":  tbl["table_id"],
                    }
                })

        # # — 画像のチャンク化 —
        # if name.lower().endswith(".pdf"):
        #     for img in extract_images_from_pdf(data):
        #         docs.append({
        #             "content": "",  # テキストチャンクではない
        #             "metadata": {
        #                 "source":   name,
        #                 "kind":     "image",
        #                 "page":     img["page"],
        #                 "image_id": img["image_id"],
        #                 "width":    img["width"],
        #                 "height":   img["height"],
        #             },
        #             "image_bytes": img["bytes"],
        #         })

        # — プレーンテキストのチャンク化 —
        if mime == "text/plain" or name.lower().endswith(".txt"):
            text = extract_text_from_txt(data)
        elif mime == "application/pdf" or name.lower().endswith(".pdf"):
            laparams = LAParams()
            text = extract_text(BytesIO(data), laparams=laparams)
        else:
            continue

        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            docs.append({
                "content": chunk,
                "metadata": {
                    "source":   name,
                    "kind":     "text",
                    "chunk_id": idx,
                }
            })

    return docs