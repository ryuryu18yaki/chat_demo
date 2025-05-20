from pathlib import Path
from src.rag_preprocess import preprocess_files
from src.rag_vector import save_docs_to_chroma

def initialize_chroma_from_input(input_dir: str, persist_dir: str | None, collection_name: str = "default") -> dict:
    """
    ベクトルDBを構築し、collection と PDFファイル群を返す
    """
    input_path = Path(input_dir)
    files = list(input_path.glob("**/*.*"))

    # PDFを読み込んで doc 化
    file_dicts = []
    for f in files:
        file_dicts.append({
            "name": f.name,
            "type": "application/pdf",
            "size": f.stat().st_size,
            "data": f.read_bytes()
        })

    docs = preprocess_files(file_dicts)
    collection = save_docs_to_chroma(docs=docs, collection_name=collection_name, persist_directory=persist_dir)

    return {
        "collection": collection,
        "rag_files": file_dicts  # ← PDFファイルのバイナリも保持
    }