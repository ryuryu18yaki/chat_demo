# src/rag_preprocess.py

from __future__ import annotations
from io import BytesIO
from typing import List, Dict, Any
import hashlib

import pdfplumber       # pip install pdfplumber
from pdfminer.high_level import extract_text  # type: ignore
from pdfminer.layout import LAParams         # type: ignore
from pypdf import PdfReader
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
def extract_images_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    images, counter = [], 0

    for pnum, page in enumerate(reader.pages, start=1):
        for img_obj in page.images:
            counter += 1
            data = img_obj.data

            # --- Pillow で形式判定 ---
            try:
                fmt = Image.open(BytesIO(data)).format.lower()  # 'jpeg', 'png', ...
            except Exception:
                fmt = "png"  # 何かあればデフォルト
            ext = "jpg" if fmt == "jpeg" else fmt

            fname = f"page{pnum}_{counter:03}.{ext}"
            images.append(
                {
                    "page": pnum,
                    "image_id": f"{counter:03}",
                    "name": fname,
                    "bytes": data,
                    "width": getattr(img_obj, "width", None),
                    "height": getattr(img_obj, "height", None),
                }
            )
    return images

# ---------------------------------------------------------------------------
# 5) メイン: ファイル→チャンク辞書リスト（大幅修正）
# ---------------------------------------------------------------------------
def preprocess_files(
    files: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    files: List of {"name": str, "type": mime, "data": bytes, "equipment_name": str, "equipment_category": str}
    を受け取り、設備ごとにファイル別でテキストを保持して返す。
    
    Returns:
        Dict[equipment_name, {
            "files": Dict[filename, file_text],  # ファイル別テキスト保持
            "sources": List[str],  # 使用したファイル名のリスト
            "equipment_category": str,
            "total_files": int,
            "total_pages": int,
            "total_chars": int
        }]
    """
    equipment_data = {}  # 設備名をキーとする辞書
    
    print(f"📚 設備ごとファイル別保持処理開始 - ファイル数: {len(files)}")

    for f in files:
        name, mime, data = f["name"], f["type"], f["data"]
        equipment_name = f.get("equipment_name", "不明")
        equipment_category = f.get("equipment_category", "その他設備")
        
        print(f"📄 処理中: {name} → 設備: {equipment_name}")
        
        # 設備データの初期化
        if equipment_name not in equipment_data:
            equipment_data[equipment_name] = {
                "files": {},  # ファイル名 → テキストの辞書
                "sources": [],
                "equipment_category": equipment_category,
                "total_files": 0,
                "total_pages": 0,
                "total_chars": 0
            }
        
        # ファイルごとのテキスト抽出
        file_text = ""
        file_pages = 0
        
        # テキストファイルの処理
        if mime == "text/plain" or name.lower().endswith(".txt"):
            try:
                raw_text = extract_text_from_txt(data)
                file_text = f"=== ファイル: {name} ===\n{raw_text}"
                file_pages = 1  # テキストファイルは1ページとして扱う
                print(f"  ✅ TXTファイル処理完了 - 文字数: {len(file_text)}")
            except Exception as e:
                print(f"  ❌ TXTファイル処理エラー: {e}")
                continue
                
        # PDFファイルの処理
        elif mime == "application/pdf" or name.lower().endswith(".pdf"):
            try:
                # ページ別にテキストを抽出
                pages_data = extract_text_from_pdf_by_pages(data)
                
                # 全ページのテキストを結合（ファイル単位）
                page_texts = [f"=== ファイル: {name} ==="]  # ファイルヘッダー
                
                for page_data in pages_data:
                    page_num = page_data["page"]
                    page_text = page_data["text"].strip()
                    
                    if page_text:  # 空ページをスキップ
                        # ページ情報を含めてテキストを整形
                        formatted_page = f"\n--- ページ {page_num} ---\n{page_text}"
                        page_texts.append(formatted_page)
                        file_pages += 1
                
                file_text = "\n".join(page_texts)
                print(f"  ✅ PDFファイル処理完了 - ページ数: {file_pages}, 文字数: {len(file_text)}")
                
            except Exception as e:
                print(f"  ❌ PDFファイル処理エラー: {e}")
                continue
        
        else:
            print(f"  ⚠️ 未対応ファイル形式: {mime}")
            continue
        
        # 設備データに追加（ファイル別に保存）
        if file_text.strip():  # 空でない場合のみ追加
            equipment_data[equipment_name]["files"][name] = file_text
            equipment_data[equipment_name]["sources"].append(name)
            equipment_data[equipment_name]["total_files"] += 1
            equipment_data[equipment_name]["total_pages"] += file_pages
            equipment_data[equipment_name]["total_chars"] += len(file_text)
    
    # 結果サマリーを出力
    print(f"\n📋 設備ごとファイル別保持処理完了")
    for equipment_name, data in equipment_data.items():
        print(f"🔧 設備: {equipment_name}")
        print(f"   ファイル数: {data['total_files']}")
        print(f"   ページ数: {data['total_pages']}")
        print(f"   総文字数: {data['total_chars']}")
        print(f"   ソース: {', '.join(data['sources'])}")
        print()
    
    return equipment_data