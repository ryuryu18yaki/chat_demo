# src/startup_loader.py (シンプル版 - ChromaDB不使用)

from pathlib import Path
from src.rag_preprocess import preprocess_files
from src.equipment_classifier import extract_equipment_from_filename, get_equipment_category

def initialize_equipment_data(input_dir: str) -> dict:
    """
    設備データを初期化し、辞書として返す（ChromaDB不使用）
    
    Args:
        input_dir: 入力ディレクトリ
        
    Returns:
        {
            "equipment_data": Dict[設備名, 設備データ],
            "file_list": List[ファイル情報],
            "equipment_list": List[設備名],
            "category_list": List[カテゴリ名]
        }
    """
    input_path = Path(input_dir)
    files = list(input_path.glob("**/*.*"))

    print(f"📂 ファイル読み込み開始 - ディレクトリ: {input_dir}")
    print(f"📁 発見ファイル数: {len(files)}")

    # ファイルを読み込んで設備メタデータ付きの辞書作成
    file_dicts = []
    for f in files:
        # ファイル名から設備名を抽出
        equipment_name = extract_equipment_from_filename(f.name)
        equipment_category = get_equipment_category(equipment_name)
        
        file_dict = {
            "name": f.name,
            "type": "application/pdf" if f.suffix.lower() == ".pdf" else "text/plain",
            "size": f.stat().st_size,
            "data": f.read_bytes(),
            # 🔥 設備メタデータを追加
            "equipment_name": equipment_name,
            "equipment_category": equipment_category
        }
        file_dicts.append(file_dict)
        
        print(f"📄 読み込み: {f.name} → 設備: {equipment_name} (カテゴリ: {equipment_category})")

    # 設備ごとに全文結合処理
    print(f"\n🔄 設備ごと全文結合処理開始...")
    equipment_data = preprocess_files(file_dicts)

    # 設備一覧とカテゴリ一覧を生成
    equipment_list = list(equipment_data.keys())
    category_list = list(set(data["equipment_category"] for data in equipment_data.values()))

    print(f"\n✅ 初期化完了")
    print(f"📊 統計情報:")
    print(f"   - 処理ファイル数: {len(file_dicts)}")
    print(f"   - 設備数: {len(equipment_list)}")
    print(f"   - カテゴリ数: {len(category_list)}")
    
    for equipment_name in sorted(equipment_list):
        data = equipment_data[equipment_name]
        total_chars = data.get('total_chars', 0)
        print(f"   - {equipment_name}: {data['total_files']}ファイル, {data['total_pages']}ページ, {total_chars}文字")

    return {
        "equipment_data": equipment_data,  # メインデータ: Dict[設備名, 設備データ]
        "file_list": file_dicts,          # 元ファイル情報（互換性のため）
        "equipment_list": sorted(equipment_list),  # 設備名一覧
        "category_list": sorted(category_list)     # カテゴリ一覧
    }

def get_equipment_names(equipment_data: dict) -> list:
    """利用可能な設備名一覧を取得"""
    return sorted(equipment_data.keys())

def get_equipment_by_category(equipment_data: dict, category: str) -> list:
    """指定カテゴリの設備名一覧を取得"""
    return [
        name for name, data in equipment_data.items() 
        if data["equipment_category"] == category
    ]

def get_equipment_full_text(equipment_data: dict, equipment_name: str, selected_files: list = None) -> str:
    """指定設備の全文を取得（ファイル選択対応）"""
    if equipment_name not in equipment_data:
        print(f"⚠️ 設備が見つかりません: {equipment_name}")
        return ""
    
    eq_data = equipment_data[equipment_name]
    files_dict = eq_data["files"]
    
    if selected_files is None:
        # 全ファイル使用
        selected_files = eq_data["sources"]
    
    # 選択されたファイルのテキストを結合
    selected_texts = []
    for file_name in selected_files:
        if file_name in files_dict:
            selected_texts.append(files_dict[file_name])
        else:
            print(f"⚠️ ファイルが見つかりません: {file_name}")
    
    return "\n\n".join(selected_texts)

def get_equipment_files(equipment_data: dict, equipment_name: str) -> dict:
    """指定設備のファイル辞書を取得"""
    if equipment_name in equipment_data:
        return equipment_data[equipment_name]["files"]
    else:
        print(f"⚠️ 設備が見つかりません: {equipment_name}")
        return {}

def get_equipment_file_text(equipment_data: dict, equipment_name: str, file_name: str) -> str:
    """指定設備の特定ファイルのテキストを取得"""
    if equipment_name not in equipment_data:
        print(f"⚠️ 設備が見つかりません: {equipment_name}")
        return ""
    
    files_dict = equipment_data[equipment_name]["files"]
    if file_name not in files_dict:
        print(f"⚠️ ファイルが見つかりません: {file_name}")
        return ""
    
    return files_dict[file_name]

def get_equipment_info(equipment_data: dict, equipment_name: str) -> dict:
    """指定設備の詳細情報を取得"""
    return equipment_data.get(equipment_name, {})

# 互換性のための関数（旧コードで使用されている可能性）
def initialize_chroma_from_input(input_dir: str, persist_dir: str | None, collection_name: str = "default") -> dict:
    """
    旧関数の互換性維持（廃止予定）
    新しいinitialize_equipment_dataを呼び出す
    """
    print("⚠️ initialize_chroma_from_input は廃止予定です。initialize_equipment_data を使用してください。")
    
    result = initialize_equipment_data(input_dir)
    
    # 旧形式に合わせて戻り値を調整
    return {
        "collection": None,  # 使用しない
        "rag_files": result["file_list"],
        "equipment_data": result["equipment_data"]  # 新しく追加
    }