# src/startup_loader.py (シンプル管轄版)
from streamlit import secrets
from pathlib import Path

from src.rag_preprocess import preprocess_files, apply_text_replacements_from_fixmap
from src.equipment_classifier import extract_equipment_from_filename, get_equipment_category
from src.fire_department_classifier import classify_files_by_jurisdiction, get_jurisdiction_stats, extract_fire_department_info  # 🔥 追加
from src.gdrive_simple import download_files_from_drive, download_fix_files_from_drive
from src.building_manager import initialize_building_manager, get_building_manager
from src.logging_utils import init_logger
from src.rag_baseline import filter_file_dicts_by_name, build_rag_retriever_from_file_dicts
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

    # 🔥 シンプル版: 基本設備データのみ作成（管轄統合なし）
    # 管轄関係なく、全ファイルから設備データを作成
    equipment_data = preprocess_files(file_dicts)

    # ビル情報マネージャーを初期化
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

    print(f"\n✅ 初期化完了（シンプル管轄版）")
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
        print(f"   - 利用可能ビル: {', '.join(building_manager.get_building_list()[:5])}...")
    
    for equipment_name in sorted(equipment_list):
        data = equipment_data[equipment_name]
        total_chars = data.get('total_chars', 0)
        print(f"   - {equipment_name}: {data['total_files']}ファイル, {data['total_pages']}ページ, {total_chars}文字")
    
    # 🔥 各ファイルにタグを付与
    logger.info("🏷️ ファイルタグ付け処理開始...")
    
    for file_dict in file_dicts:
        filename = file_dict.get("name", "")
        
        # 管轄タグの決定
        fire_info = extract_fire_department_info(filename)  # fire_department_classifier.py使用
        
        if fire_info["jurisdiction"] == "丸の内消防署":
            file_dict["jurisdiction_tag"] = "🔥丸の内消防署"
        elif fire_info["jurisdiction"] == "東京消防庁":
            file_dict["jurisdiction_tag"] = "🔥東京消防庁"
        elif fire_info["is_general"]:
            file_dict["jurisdiction_tag"] = "📄一般消防資料"
        else:
            file_dict["jurisdiction_tag"] = "📄一般設備資料"  # デフォルト
        
        logger.info(f"🏷️ {filename} → タグ: {file_dict['jurisdiction_tag']}")
    
    # 既存の設備データ作成処理（変更なし）
    equipment_data = preprocess_files(file_dicts)
    
    # 🔥 設備データの各ファイルにもタグ情報を追加
    for equipment_name, eq_data in equipment_data.items():
        # ファイルソース情報にタグを追加
        tagged_sources = []
        for source_file in eq_data["sources"]:
            # 元のファイルからタグ情報を取得
            original_file = next((f for f in file_dicts if f["name"] == source_file), None)
            if original_file:
                tag = original_file.get("jurisdiction_tag", "📄一般設備資料")
                tagged_sources.append({
                    "name": source_file,
                    "tag": tag
                })
        
        eq_data["tagged_sources"] = tagged_sources
    
    # 既存の戻り値に加えて、タグ統計も追加
    tag_stats = get_tag_statistics(file_dicts)

    try:
        # secrets で ON/OFF したい場合（なければデフォルト True）
        rag_enabled = True
        try:
            rag_enabled = bool(secrets.get("RAG_MODE", True))
        except Exception:
            pass

        rag_retriever = None
        rag_stats = {}

        if rag_enabled and file_dicts:
            # ファイルをフィルタリング（「ビルマスター」を除外）
            filtered_file_dicts = filter_file_dicts_by_name(
                file_dicts,
                exclude=["ビルマスター"]
            )
            
            logger.info(f"📂 フィルタリング前: {len(file_dicts)}ファイル -> フィルタリング後: {len(filtered_file_dicts)}ファイル")

            if filtered_file_dicts:
                rag_retriever, rag_stats = build_rag_retriever_from_file_dicts(
                    filtered_file_dicts,
                    chunk_size=1000,
                    chunk_overlap=200,
                    k=3,
                    use_mmr=False,
                )

                logger.info("✅ RAG retriever 構築完了: %s", rag_stats)
            else:
                logger.info("ℹ️ フィルタリング後にファイルが0件のため RAG 構築をスキップ")

        else:
            logger.info("ℹ️ RAGは無効化または file_dicts が空のためスキップ")

    except Exception as e:
        logger.error("❌ RAG 構築に失敗しました: %s", e, exc_info=True)
        rag_retriever, rag_stats = None, {}
    
    return {
        "equipment_data": equipment_data,
        "file_list": file_dicts,
        "equipment_list": sorted(equipment_list),
        "category_list": sorted(category_list),
        "building_manager": building_manager,
        "tag_stats": tag_stats,
        "rag_retriever": rag_retriever,
        "rag_stats": rag_stats,
    }

def get_tag_statistics(file_dicts: list) -> dict:
    """ファイルのタグ統計を取得"""
    stats = {}
    for file_dict in file_dicts:
        tag = file_dict.get("jurisdiction_tag", "📄一般設備資料")
        stats[tag] = stats.get(tag, 0) + 1
    return stats

# 🔥 新規関数: 管轄に基づいてフィルタされたファイルリストを取得
def get_filtered_files_by_jurisdiction(equipment_name: str, selected_jurisdiction: str = None) -> list:
    """
    指定された管轄に基づいて、設備のファイルリストをフィルタリング
    
    Args:
        equipment_name: 設備名
        selected_jurisdiction: 選択された管轄 ("🔥東京消防庁" | "🔥丸の内消防署" | None)
        
    Returns:
        フィルタされたファイル名リスト
    """
    import streamlit as st
    
    equipment_data = st.session_state.get("equipment_data", {})
    if equipment_name not in equipment_data:
        return []
    
    eq_data = equipment_data[equipment_name]
    tagged_sources = eq_data.get("tagged_sources", [])
    
    if not selected_jurisdiction:
        # 管轄指定なし → 一般設備資料のみ
        allowed_tags = ["📄一般設備資料"]
    elif selected_jurisdiction == "🔥東京消防庁":
        # 東京消防庁 → 一般設備資料 + 一般消防資料 + 東京消防庁
        allowed_tags = ["📄一般設備資料", "📄一般消防資料", "🔥東京消防庁"]
    elif selected_jurisdiction == "🔥丸の内消防署":
        # 丸の内消防署 → 一般設備資料 + 一般消防資料 + 東京消防庁 + 丸の内消防署
        allowed_tags = ["📄一般設備資料", "📄一般消防資料", "🔥東京消防庁", "🔥丸の内消防署"]
    else:
        allowed_tags = ["📄一般設備資料"]
    
    # フィルタリング
    filtered_files = []
    for source in tagged_sources:
        if source["tag"] in allowed_tags:
            filtered_files.append(source["name"])
    
    return filtered_files

# 🔥 新規追加: プロンプト生成時に管轄資料を取得する関数
def get_jurisdiction_content_for_equipment(equipment_name: str, selected_jurisdiction: str = None) -> str:
    """
    指定された設備と管轄に応じて、追加すべき管轄資料のテキストを取得
    
    Args:
        equipment_name: 選択された設備名
        selected_jurisdiction: 選択された管轄 ("東京消防庁" | "丸の内消防署" | None)
        
    Returns:
        管轄固有の資料テキスト（ない場合は空文字）
    """
    if not selected_jurisdiction:
        return ""
    
    import streamlit as st
    jurisdiction_classified = st.session_state.get("jurisdiction_classified", {})
    
    if not jurisdiction_classified:
        return ""
    
    # 階層的に取得
    jurisdiction_files = []
    
    if selected_jurisdiction == "東京消防庁":
        # 一般消防資料 + 東京消防庁資料
        jurisdiction_files.extend(jurisdiction_classified.get("general_fire", []))
        jurisdiction_files.extend(jurisdiction_classified.get("jurisdictions", {}).get("東京消防庁", []))
        
    elif selected_jurisdiction == "丸の内消防署":
        # 一般消防資料 + 東京消防庁資料 + 丸の内資料
        jurisdiction_files.extend(jurisdiction_classified.get("general_fire", []))
        jurisdiction_files.extend(jurisdiction_classified.get("jurisdictions", {}).get("東京消防庁", []))
        jurisdiction_files.extend(jurisdiction_classified.get("jurisdictions", {}).get("丸の内消防署", []))
    
    if not jurisdiction_files:
        return ""
    
    # 選択された設備に関連する管轄ファイルのみフィルタリング
    relevant_files = []
    for file_dict in jurisdiction_files:
        file_equipment = file_dict.get("equipment_name", "")
        # 設備名が一致するか、またはその他の場合は含める
        if file_equipment == equipment_name or file_equipment == "その他":
            relevant_files.append(file_dict)
    
    if not relevant_files:
        return ""
    
    # ファイルからテキストを抽出して結合
    # 注意: ここでは簡易的な処理。実際にはpreprocess_filesと同様の処理が必要
    jurisdiction_texts = []
    for file_dict in relevant_files:
        file_name = file_dict.get("name", "")
        jurisdiction_texts.append(f"=== {file_name} ===")
        # 実際のテキスト抽出処理は省略（必要に応じて実装）
    
    return "\n\n".join(jurisdiction_texts)

def _create_empty_result() -> dict:
    """空の結果を返す"""
    return {
        "equipment_data": {},
        "file_list": [],
        "equipment_list": [],
        "category_list": [],
        "fixes_files": {},
        "building_manager": None,
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

# 既存の関数は変更なし
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
        "equipment_data": result["equipment_data"]
    }