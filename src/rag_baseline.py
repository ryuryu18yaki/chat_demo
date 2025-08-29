# rag_baseline.py
from typing import List, Optional, Dict, Tuple, Iterable
import io, json, re
import pdfplumber

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings


# ========== 1) 文書読み込み・分割 =================================
def _bytes_to_text(data: bytes, file_name: str) -> str:
    """
    TXT/JSONなどテキスト系のbytes -> str 変換（UTF-8前提、失敗時はignore）
    """
    text = data.decode("utf-8", errors="ignore")
    if file_name.lower().endswith(".json"):
        try:
            # JSONは整形のうえ文字列化（検索性を少し上げる）
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except Exception:
            pass
    return text

def file_dicts_to_documents(file_dicts: list[dict]) -> list["Document"]:
    """
    Google Drive から取得した file_dicts（bytes含む）を LangChain Document に変換。
    - PDF: pdfplumberでページごとに抽出（1ページ=1ドキュメント）
    - TXT/JSON: 1ファイル=1ドキュメント（整形方針をPDF側と合わせ、ヘッダー等を付与）
    - equipment_classifier の結果などメタデータをそのまま付与
    """
    # 依存をここでimport（ヘルパー化しないため）

    docs: list[Document] = []

    for f in file_dicts:
        name = f.get("name")
        mime = f.get("type")
        size = f.get("size")
        data = f.get("data", b"")
        equip = f.get("equipment_name")
        category = f.get("equipment_category")
        jurisdiction_tag = f.get("jurisdiction_tag")

        # ---- 共通メタデータ -----------------------------------------
        meta = {
            "source": name,
            "mime_type": mime,
            "size": size,
            "equipment_name": equip,
            "equipment_category": category,
        }
        if jurisdiction_tag:
            meta["jurisdiction_tag"] = jurisdiction_tag

        # ---- ページ番号表記を残すか？（簡易ロジック：議事録/ログ系は外す） ----
        lower_name = (name or "").lower()
        include_pages = not (("議事録" in (name or "")) or ("minutes" in lower_name) or ("log" in lower_name))

        # ---- TXT/JSON（1ドキュメント） ------------------------------
        if (mime == "text/plain" or lower_name.endswith(".txt") or lower_name.endswith(".json")):
            # デコード（簡易：utf-8→cp932→ignore）
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = data.decode("cp932")
                except UnicodeDecodeError:
                    text = data.decode("utf-8", errors="ignore")

            # JSONは整形
            if lower_name.endswith(".json"):
                try:
                    text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
                except Exception:
                    # 壊れたJSONは原文のまま
                    pass

            # 改行・空行整形
            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            text = re.sub(r"\n{3,}", "\n\n", text)

            # ヘッダー付与（PDFと合わせる）
            header = f"=== ファイル: {name} ==="
            content = f"{header}\n{text}" if text else header

            # 1ページ扱いに統一（後段の処理互換のため）
            doc = Document(
                page_content=content,
                metadata={**meta, "page": 1}
            )
            docs.append(doc)
            continue

        # ---- PDF（pdfplumberで1ページ=1Doc） ------------------------
        if (mime == "application/pdf" or lower_name.endswith(".pdf")):
            # ページ番号検出パターン（簡易）
            # 例: "12", "12 / 45", "12/45", "Page 12", "p.12"
            pat_single_num = re.compile(r"^\s*\d+\s*$", re.IGNORECASE)
            pat_fraction   = re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.IGNORECASE)
            pat_page_word  = re.compile(r"^\s*(page|p\.)\s*\d+\s*$", re.IGNORECASE)

            try:
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    for idx, page in enumerate(pdf.pages, start=1):
                        # テキスト抽出（必要に応じて tolerance 調整）
                        page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                        page_text = page_text.replace("\r\n", "\n").replace("\r", "\n").strip()

                        # 余分な空行圧縮
                        page_text = re.sub(r"\n{3,}", "\n\n", page_text)

                        # ページ番号を含めない設定なら、末尾・先頭に出がちな番号行を簡易で除去
                        if not include_pages and page_text:
                            cleaned_lines = []
                            for ln in page_text.splitlines():
                                ln_stripped = ln.strip()
                                if pat_single_num.match(ln_stripped) or pat_fraction.match(ln_stripped) or pat_page_word.match(ln_stripped):
                                    # ページ番号っぽい行はスキップ
                                    continue
                                cleaned_lines.append(ln)
                            page_text = "\n".join(cleaned_lines).strip()

                        # 空ページはスキップ（必要なら保持に変更）
                        if not page_text:
                            continue

                        # ページヘッダー（include_pages が True の場合のみページ明示）
                        header = f"=== ファイル: {name} ==="
                        page_header = f"--- ページ {idx} ---" if include_pages else ""
                        content = f"{header}\n{page_header}\n{page_text}".strip()

                        doc = Document(
                            page_content=content,
                            metadata={**meta, "page": idx}
                        )
                        docs.append(doc)
            except Exception as e:
                # 壊れたPDF等はスキップ（ログは上位で拾う想定）
                # print(f"pdf parse error for {name}: {e}")
                pass

            continue

        # ---- その他形式はスキップ -------------------------------
        # 追加対応したい場合はここに分岐を足す（docx, csv, xlsx, etc.）
        continue

    return docs

def filter_file_dicts_by_name(
    file_dicts: list[dict],
    *,
    include: Iterable[str] | None = None,   # 例: ["pdf", "消防"]
    exclude: Iterable[str] | None = None,   # 例: ["旧暗黙知", "draft"]
    casefold: bool = True,
) -> list[dict]:
    """
    file_dicts をファイル名で選別する。
    - include が指定されていれば、その文字列を含むものだけ採用
    - exclude は常に除外（文字列を含むものを除外）
    - 単純な部分一致で判定
    """
    def _norm(s: str) -> str:
        return s.casefold() if casefold else s

    inc = [p for p in (include or [])]
    exc = [p for p in (exclude or [])]

    def match(name: str, patterns: list[str]) -> bool:
        if not patterns:
            return False
        n = _norm(name)
        for p in patterns:
            q = _norm(p)
            if q in n:
                return True
        return False

    out: list[dict] = []
    for f in file_dicts:
        name = f.get("name", "")
        # include 条件：指定があれば、少なくとも1つマッチが必要
        if inc and not match(name, inc):
            continue
        # exclude 条件：1つでもマッチしたら除外
        if exc and match(name, exc):
            continue
        out.append(f)
    return out

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """汎用の再帰スプリッタでチャンク化。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(documents)


# ========== 2) 埋め込み・インメモリFAISS ==========================

def make_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """OpenAIの多言語埋め込み（日本語OK）。"""
    return OpenAIEmbeddings(model=model)

def build_faiss_in_memory(chunks: List[Document], embeddings):
    """FAISS をメモリ上に構築（保存しない）。"""
    if not chunks:
        raise ValueError("No documents to index: chunks is empty.")
    return FAISS.from_documents(chunks, embedding=embeddings)


# ========== 3) Retriever（検索器） ================================
def make_retriever(
    vectorstore,
    k: int = 3,
    *,
    use_mmr: bool = False,
    fetch_k: int = 20,
    metadata_filter: Optional[Dict] = None,
):
    """類似度検索が基本。多様性が必要なら MMR をON。"""
    if use_mmr:
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                **({"filter": metadata_filter} if metadata_filter else {}),
            },
        )
    return vectorstore.as_retriever(
        search_kwargs={"k": k, **({"filter": metadata_filter} if metadata_filter else {})}
    )


# ========== 4) 便利ヘルパ（任意） ================================

def build_rag_retriever_from_file_dicts(
    file_dicts: list[dict],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 3,
    use_mmr: bool = False,
) -> Tuple[any, dict]:
    """
    file_dicts -> Documents -> チャンク -> 埋め込み -> インメモリFAISS -> Retriever
    をまとめて作るユーティリティ。stats には件数を返す。
    """
    docs = file_dicts_to_documents(file_dicts)
    # チャンク切りを最適化
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    emb = make_embeddings()
    vs = build_faiss_in_memory(chunks, emb)
    # ここまでが妥当では？
    retriever = make_retriever(vs, k=k, use_mmr=use_mmr)
    stats = {"docs": len(docs), "chunks": len(chunks), "k": k, "mmr": use_mmr}
    # retrieverはここでいいのか？初期のロードでここまではダメな気がする
    return retriever, stats

def get_context_text(
    retriever,
    query: str,
    *,
    k: Optional[int] = None,
    max_chars: int = 3500,
    join_delim: str = "\n---\n",
) -> str:
    """
    取得文書をプレーンテキスト連結して返す。
    既存プロンプトの {context} へそのまま差し込める想定。
    """
    kwargs = {}
    if k is not None:
        # 一部の retriever 実装は .invoke で動くが、LangChain標準では get_relevant_documents が安定
        docs = retriever.get_relevant_documents(query=query, **kwargs)[:k]
    else:
        docs = retriever.get_relevant_documents(query=query, **kwargs)

    parts: List[str] = []
    for d in docs:
        header = f"[source: {d.metadata.get('source')}, page: {d.metadata.get('page')}]"
        parts.append(header + "\n" + d.page_content.strip())

    text = join_delim.join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text
