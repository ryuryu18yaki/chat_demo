# src/rag_preprocess.py - テーブル対応版（重複はそのまま）

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
    "format_table_as_text",
    "extract_structured_content"
]

# ---------------------------------------------------------------------------
# 1) テキスト抽出（既存）
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
# 2) チャンク化ユーティリティ（既存）
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

def generate_chunk_id(source: str, page: int, chunk_index: int, content: str) -> str:
    """ユニークなチャンクIDを生成"""
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{source}_p{page}_{chunk_index}_{content_hash}"

# ---------------------------------------------------------------------------
# 3) 表の抽出（強化版）
# ---------------------------------------------------------------------------
def extract_tables_from_pdf(data: bytes) -> List[Dict[str, Any]]:
    """pdfplumber で表を抽出し、構造化されたデータとして返す。"""
    tables: List[Dict[str, Any]] = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()
            
            for tbl_idx, table in enumerate(page_tables, start=1):
                if not table or len(table) == 0:
                    continue
                    
                # テーブルデータの処理
                processed_table = []
                headers = []
                
                # ヘッダー行の検出（最初の行をヘッダーとして扱う）
                if table[0]:
                    headers = [str(cell).strip() if cell else f"列{i+1}" for i, cell in enumerate(table[0])]
                
                # データ行の処理
                for row_idx, row in enumerate(table[1:] if headers else table, start=1):
                    if not row:
                        continue
                    processed_row = [str(cell).strip() if cell else "" for cell in row]
                    if any(processed_row):  # 空行をスキップ
                        processed_table.append(processed_row)
                
                if processed_table:
                    # CSV形式のテキスト生成
                    csv_lines = []
                    if headers:
                        csv_lines.append(",".join(f'"{h}"' for h in headers))
                    
                    for row in processed_table:
                        csv_lines.append(",".join(f'"{cell}"' for cell in row))
                    
                    csv_text = "\n".join(csv_lines)
                    
                    # 構造化されたテーブル情報
                    table_info = {
                        "page": page_num,
                        "table_id": tbl_idx,
                        "headers": headers,
                        "data": processed_table,
                        "csv_text": csv_text,
                        "row_count": len(processed_table),
                        "col_count": len(headers) if headers else (len(processed_table[0]) if processed_table else 0),
                        "formatted_text": format_table_as_text(headers, processed_table)
                    }
                    
                    tables.append(table_info)
                    print(f"📊 テーブル抽出: ページ{page_num}, テーブル{tbl_idx} - {len(processed_table)}行×{table_info['col_count']}列")
    
    return tables

def format_table_as_text(headers: List[str], data: List[List[str]]) -> str:
    """テーブルデータを読みやすいテキスト形式に変換"""
    if not data:
        return ""
    
    lines = []
    
    # ヘッダー情報
    if headers:
        lines.append("【テーブル項目】")
        for i, header in enumerate(headers):
            lines.append(f"  {i+1}. {header}")
        lines.append("")
    
    # データ行の処理
    lines.append("【テーブルデータ】")
    for row_idx, row in enumerate(data, start=1):
        lines.append(f"行{row_idx}:")
        
        if headers and len(row) == len(headers):
            # ヘッダーと対応付けて表示
            for header, value in zip(headers, row):
                if value.strip():  # 空でない値のみ
                    lines.append(f"  ・{header}: {value}")
        else:
            # ヘッダーなしまたは列数不一致の場合
            for col_idx, value in enumerate(row):
                if value.strip():
                    lines.append(f"  ・列{col_idx+1}: {value}")
        lines.append("")
    
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 4) 画像の抽出（既存）
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
# 5) 構造化コンテンツ抽出
# ---------------------------------------------------------------------------
def extract_structured_content(data: bytes, filename: str) -> Dict[str, Any]:
    """PDFから構造化されたコンテンツ（テキスト+テーブル）を抽出"""
    
    result = {
        "text_content": [],
        "table_content": [],
        "total_text_chars": 0,
        "total_tables": 0
    }
    
    # テキスト抽出
    try:
        pages_text = extract_text_from_pdf_by_pages(data)
        result["text_content"] = pages_text
        result["total_text_chars"] = sum(len(page["text"]) for page in pages_text)
        print(f"📄 テキスト抽出完了: {filename} - {len(pages_text)}ページ, {result['total_text_chars']}文字")
    except Exception as e:
        print(f"❌ テキスト抽出エラー: {filename} - {e}")
    
    # テーブル抽出
    try:
        tables = extract_tables_from_pdf(data)
        result["table_content"] = tables
        result["total_tables"] = len(tables)
        print(f"📊 テーブル抽出完了: {filename} - {len(tables)}テーブル")
    except Exception as e:
        print(f"❌ テーブル抽出エラー: {filename} - {e}")
    
    return result

# ---------------------------------------------------------------------------
# 6) メイン: ファイル→設備辞書リスト（テーブル対応版）
# ---------------------------------------------------------------------------
def preprocess_files(
    files: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    files: List of {"name": str, "type": mime, "data": bytes, "equipment_name": str, "equipment_category": str}
    を受け取り、設備ごとにファイル別でテキスト+テーブルを保持して返す。
    
    Returns:
        Dict[equipment_name, {
            "files": Dict[filename, file_content],  # ファイル別コンテンツ保持
            "sources": List[str],  # 使用したファイル名のリスト
            "equipment_category": str,
            "total_files": int,
            "total_pages": int,
            "total_chars": int,
            "total_tables": int,  # テーブル数
            "table_info": List[Dict]  # テーブル詳細情報
        }]
    """
    equipment_data = {}  # 設備名をキーとする辞書
    
    print(f"📚 設備ごとファイル別保持処理開始（テーブル対応版） - ファイル数: {len(files)}")

    for f in files:
        name, mime, data = f["name"], f["type"], f["data"]
        equipment_name = f.get("equipment_name", "不明")
        equipment_category = f.get("equipment_category", "その他設備")
        
        print(f"📄 処理中: {name} → 設備: {equipment_name}")
        
        # 設備データの初期化
        if equipment_name not in equipment_data:
            equipment_data[equipment_name] = {
                "files": {},  # ファイル名 → コンテンツの辞書
                "sources": [],
                "equipment_category": equipment_category,
                "total_files": 0,
                "total_pages": 0,
                "total_chars": 0,
                "total_tables": 0,  # テーブル数
                "table_info": []    # テーブル詳細情報
            }
        
        # ファイルごとのコンテンツ抽出
        file_content = ""
        file_pages = 0
        file_tables = 0
        
        # テキストファイルの処理
        if mime == "text/plain" or name.lower().endswith(".txt"):
            try:
                raw_text = extract_text_from_txt(data)
                file_content = f"=== ファイル: {name} ===\n{raw_text}"
                file_pages = 1  # テキストファイルは1ページとして扱う
                print(f"  ✅ TXTファイル処理完了 - 文字数: {len(file_content)}")
            except Exception as e:
                print(f"  ❌ TXTファイル処理エラー: {e}")
                continue
                
        # PDFファイルの処理（テーブル対応版）
        elif mime == "application/pdf" or name.lower().endswith(".pdf"):
            try:
                # 構造化コンテンツ抽出
                structured = extract_structured_content(data, name)
                
                # ファイルヘッダー
                content_parts = [f"=== ファイル: {name} ==="]
                
                # テキストコンテンツの処理
                for page_data in structured["text_content"]:
                    page_num = page_data["page"]
                    page_text = page_data["text"].strip()
                    
                    if page_text:  # 空ページをスキップ
                        formatted_page = f"\n--- ページ {page_num} ---\n{page_text}"
                        content_parts.append(formatted_page)
                        file_pages += 1
                
                # テーブルコンテンツの処理
                if structured["table_content"]:
                    content_parts.append(f"\n=== テーブル情報 ({len(structured['table_content'])}個) ===")
                    
                    for table in structured["table_content"]:
                        table_header = f"\n--- ページ{table['page']} テーブル{table['table_id']} ({table['row_count']}行×{table['col_count']}列) ---"
                        content_parts.append(table_header)
                        content_parts.append(table["formatted_text"])
                        
                        # テーブル情報を設備データに追加
                        table_info = {
                            "source_file": name,
                            "page": table["page"],
                            "table_id": table["table_id"],
                            "headers": table["headers"],
                            "row_count": table["row_count"],
                            "col_count": table["col_count"],
                            "formatted_text": table["formatted_text"]
                        }
                        equipment_data[equipment_name]["table_info"].append(table_info)
                        file_tables += 1
                
                file_content = "\n".join(content_parts)
                
                print(f"  ✅ PDFファイル処理完了 - ページ数: {file_pages}, テーブル数: {file_tables}, 文字数: {len(file_content)}")
                
            except Exception as e:
                print(f"  ❌ PDFファイル処理エラー: {e}")
                continue
        
        else:
            print(f"  ⚠️ 未対応ファイル形式: {mime}")
            continue
        
        # 設備データに追加（ファイル別に保存）
        if file_content.strip():  # 空でない場合のみ追加
            equipment_data[equipment_name]["files"][name] = file_content
            equipment_data[equipment_name]["sources"].append(name)
            equipment_data[equipment_name]["total_files"] += 1
            equipment_data[equipment_name]["total_pages"] += file_pages
            equipment_data[equipment_name]["total_chars"] += len(file_content)
            equipment_data[equipment_name]["total_tables"] += file_tables
    
    # 結果サマリーを出力
    print(f"\n📋 設備ごとファイル別保持処理完了（テーブル対応版）")
    for equipment_name, data in equipment_data.items():
        print(f"🔧 設備: {equipment_name}")
        print(f"   ファイル数: {data['total_files']}")
        print(f"   ページ数: {data['total_pages']}")
        print(f"   テーブル数: {data['total_tables']}")
        print(f"   総文字数: {data['total_chars']}")
        print(f"   ソース: {', '.join(data['sources'])}")
        print()
    
    return equipment_data