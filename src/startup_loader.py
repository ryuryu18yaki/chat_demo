# src/startup_loader.py (最新コードベース + 管轄統合版)
from streamlit import secrets
from pathlib import Path

from src.rag_preprocess import preprocess_files, apply_text_replacements_from_fixmap
from src.equipment_classifier import extract_equipment_from_filename, get_equipment_category
from src.fire_department_classifier import classify_files_by_jurisdiction, get_jurisdiction_stats  # 🔥 追加
from src.gdrive_simple import download_files_from_drive, download_fix_files_from_drive
from src.building_manager import initialize_building_manager, get_building_manager
from src.logging_utils import init_logger
logger = init_logger()

def initialize_equipment_data(input_dir: str = "rag_data") -> dict:
    logger.info("🚨🚨🚨 NEW_FUNCTION: 関数呼び出し - input_dir='%s'", input_dir)
    
    # Google Driveからの読み込み判定
    if input_dir.startswith("gdrive:"):
        logger.info("🚨🚨🚨 NEW_FUNCTION: Google Driveモード")
        folder_id = input_dir.replace("gdrive:", "")
        logger.info("📂 Google Driveから読み込み - フォルダID: %s", folder_id)
        
        try:
            logger.info("🚨🚨🚨 gdrive_simple import開始")
            logger.info("🚨🚨🚨 download_files_from_drive 呼び出し開始")
            file_dicts = download_files_from_drive(folder_id)
            logger.info("🚨🚨🚨 download_files_from_drive 結果: %dファイル", len(file_dicts))
            
            if not file_dicts:
                logger.warning("⚠️ Google Driveからファイルが読み込めませんでした")
                return _create_empty_result()
        except Exception as e:
            logger.error("❌ Google Drive読み込み失敗: %s", e, exc_info=True)
            return _create_empty_result()
    else:
        logger.info("🚨🚨🚨 NEW_FUNCTION: ローカルモード - input_dir: %s", input_dir)
        
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ ディレクトリが存在しません: {input_dir}")
            return _create_empty_result()
        
        files = list(input_path.glob("**/*.*"))
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
                "equipment_name": equipment_name,
                "equipment_category": equipment_category
            }
            file_dicts.append(file_dict)
            
            print(f"📄 読み込み: {f.name} → 設備: {equipment_name} (カテゴリ: {equipment_category})")

    # 🔥 管轄別分類処理を追加
    logger.info("🚨 管轄別分類処理開始...")
    try:
        jurisdiction_classified = classify_files_by_jurisdiction(file_dicts)
        jurisdiction_stats = get_jurisdiction_stats(jurisdiction_classified)
        
        logger.info("🔥 管轄別分類結果:")
        logger.info(f"   - 東京消防庁: {jurisdiction_stats['東京消防庁_ファイル数']}ファイル")
        logger.info(f"   - 丸の内消防署: {jurisdiction_stats['丸の内消防署_ファイル数']}ファイル")
        logger.info(f"   - 一般消防資料: {jurisdiction_stats['一般消防資料_ファイル数']}ファイル")
        logger.info(f"   - 設備ファイル: {jurisdiction_stats['設備ファイル数']}ファイル")
    except Exception as e:
        logger.error(f"❌ 管轄分類処理失敗: {e}")
        # フォールバック：管轄分類なしで継続
        jurisdiction_classified = {
            "jurisdictions": {"東京消防庁": [], "丸の内消防署": []},
            "general_fire": [],
            "equipment_files": file_dicts
        }
        jurisdiction_stats = {
            "東京消防庁_ファイル数": 0,
            "丸の内消防署_ファイル数": 0,
            "一般消防資料_ファイル数": 0,
            "設備ファイル数": len(file_dicts),
            "消防関連総数": 0
        }

    # 🔥 階層的な設備データ構築
    # 1. 基本設備データ（一般設備のみ）
    base_files = jurisdiction_classified["equipment_files"]
    base_equipment_data = preprocess_files(base_files)
    
    # 2. 一般消防資料を独立設備として追加
    general_fire_files = jurisdiction_classified["general_fire"]
    if general_fire_files:
        general_fire_processed = preprocess_files(general_fire_files)
        # 一般消防資料をまとめて一つの設備として扱う
        if general_fire_processed:
            combined_general_fire = {
                "equipment_category": "消防設備",
                "total_files": sum(data["total_files"] for data in general_fire_processed.values()),
                "total_pages": sum(data["total_pages"] for data in general_fire_processed.values()),
                "total_chars": sum(data["total_chars"] for data in general_fire_processed.values()),
                "sources": [],
                "files": {}
            }
            for equipment_name, data in general_fire_processed.items():
                combined_general_fire["sources"].extend(data["sources"])
                combined_general_fire["files"].update(data["files"])
            
            base_equipment_data["一般消防資料"] = combined_general_fire

    # 3. 🔥東京消防庁の階層的設備作成
    tokyo_files = jurisdiction_classified["jurisdictions"]["東京消防庁"]
    if tokyo_files:
        # 基本設備 + 一般消防 + 東京消防庁
        combined_files = base_files + general_fire_files + tokyo_files
        tokyo_all_data = preprocess_files(combined_files)
        
        if tokyo_all_data:
            combined_tokyo = {
                "equipment_category": "消防設備",
                "total_files": sum(data["total_files"] for data in tokyo_all_data.values()),
                "total_pages": sum(data["total_pages"] for data in tokyo_all_data.values()),
                "total_chars": sum(data["total_chars"] for data in tokyo_all_data.values()),
                "sources": [],
                "files": {}
            }
            for equipment_name, data in tokyo_all_data.items():
                combined_tokyo["sources"].extend(data["sources"])
                combined_tokyo["files"].update(data["files"])
            
            base_equipment_data["🔥東京消防庁"] = combined_tokyo

    # 4. 🔥丸の内消防署の階層的設備作成
    marunouchi_files = jurisdiction_classified["jurisdictions"]["丸の内消防署"]
    if marunouchi_files:
        # 基本設備 + 一般消防 + 東京消防庁 + 丸の内
        combined_files = base_files + general_fire_files + tokyo_files + marunouchi_files
        marunouchi_all_data = preprocess_files(combined_files)
        
        if marunouchi_all_data:
            combined_marunouchi = {
                "equipment_category": "消防設備",
                "total_files": sum(data["total_files"] for data in marunouchi_all_data.values()),
                "total_pages": sum(data["total_pages"] for data in marunouchi_all_data.values()),
                "total_chars": sum(data["total_chars"] for data in marunouchi_all_data.values()),
                "sources": [],
                "files": {}
            }
            for equipment_name, data in marunouchi_all_data.items():
                combined_marunouchi["sources"].extend(data["sources"])
                combined_marunouchi["files"].update(data["files"])
            
            base_equipment_data["🔥丸の内消防署"] = combined_marunouchi

    # 最終的な設備データ
    equipment_data = base_equipment_data

    # 🔥 ビル情報マネージャーを初期化（file_dictsを使用）
    logger.info(f"\n🏢 ビル情報マネージャー初期化中...")
    logger.info("🔍 file_dicts 詳細情報:")
    logger.info("   - file_dicts 型: %s", type(file_dicts))
    logger.info("   - file_dicts 長さ: %d", len(file_dicts) if file_dicts else 0)
    
    if file_dicts:
        logger.info("   - 最初の3ファイル:")
        for i, file_dict in enumerate(file_dicts[:3]):
            name = file_dict.get("name", "N/A")
            size = file_dict.get("size", 0)
            logger.info("     %d. %s (%d bytes)", i+1, name, size)
    
    building_manager = initialize_building_manager(file_dicts)
    
    if building_manager.available:
        building_count = len(building_manager.get_building_list())
        logger.info(f"✅ ビル情報初期化完了: {building_count}件のビル情報")
    else:
        logger.warning("⚠️ ビル情報の初期化に失敗しました")

    # 設備一覧とカテゴリ一覧を生成
    equipment_list = list(equipment_data.keys())
    category_list = list(set(data["equipment_category"] for data in equipment_data.values()))

    print(f"\n✅ 初期化完了（管轄統合版）")
    print(f"📊 統計情報:")
    print(f"   - 処理ファイル数: {len(file_dicts)}")
    print(f"   - 設備数: {len(equipment_list)}")
    print(f"   - カテゴリ数: {len(category_list)}")
    
    # 🔥 管轄統計を表示
    print(f"🔥 管轄別資料:")
    print(f"   - 東京消防庁: {jurisdiction_stats['東京消防庁_ファイル数']}ファイル")
    print(f"   - 丸の内消防署: {jurisdiction_stats['丸の内消防署_ファイル数']}ファイル")
    print(f"   - 一般消防資料: {jurisdiction_stats['一般消防資料_ファイル数']}ファイル")
    print(f"   - 消防関連総数: {jurisdiction_stats['消防関連総数']}ファイル")
    
    # ビル情報統計を追加
    building_manager = get_building_manager()
    if building_manager and building_manager.available:
        building_count = len(building_manager.get_building_list())
        print(f"   - ビル情報数: {building_count}")
        print(f"   - 利用可能ビル: {', '.join(building_manager.get_building_list()[:5])}...")  # 最初の5件のみ表示
    
    for equipment_name in sorted(equipment_list):
        data = equipment_data[equipment_name]
        total_chars = data.get('total_chars', 0)
        print(f"   - {equipment_name}: {data['total_files']}ファイル, {data['total_pages']}ページ, {total_chars}文字")

    return {
        "equipment_data": equipment_data,
        "file_list": file_dicts,
        "equipment_list": sorted(equipment_list),
        "category_list": sorted(category_list),
        "building_manager": building_manager if 'building_manager' in locals() else None,
        # 🔥 管轄関連データを追加
        "jurisdiction_classified": jurisdiction_classified,
        "jurisdiction_stats": jurisdiction_stats
    }

def _create_empty_result() -> dict:
    """空の結果を返す"""
    return {
        "equipment_data": {},
        "file_list": [],
        "equipment_list": [],
        "category_list": [],
        "fixes_files": {},  # 🔥 追加
        "building_manager": None,  # 🔥 追加
        # 🔥 管轄関連の空データを追加
        "jurisdiction_classified": {
            "jurisdictions": {"東京消防庁": [], "丸の内消防署": []},
            "general_fire": [],
            "equipment_files": []
        },
        "jurisdiction_stats": {
            "東京消防庁_ファイル数": 0,
            "丸の内消防署_ファイル数": 0,
            "一般消防資料_ファイル数": 0,
            "設備ファイル数": 0,
            "消防関連総数": 0
        }
    }

# 🔥 ビル情報関連の便利関数を追加
def get_available_buildings() -> list:
    """利用可能なビル一覧を取得"""
    manager = get_building_manager()
    return manager.get_building_list() if manager and manager.available else []

def get_building_info_for_prompt(building_name: str = None) -> str:
    """ビル情報をプロンプト用にフォーマット"""
    manager = get_building_manager()
    if manager and manager.available:
        return manager.format_building_info_for_prompt(building_name)
    return "【ビル情報】利用可能なビル情報がありません。"

# 既存の関数は変更なし
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