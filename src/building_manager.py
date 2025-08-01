# src/building_manager.py（簡素化版）

import json
from typing import Dict, List, Any, Optional
from src.logging_utils import init_logger

logger = init_logger()

class BuildingManager:
    """三菱地所ビルマスター.jsonを管理するクラス"""
    
    def __init__(self, file_dicts: List[Dict[str, Any]]):
        """
        初期化
        
        Args:
            file_dicts: download_files_from_drive の結果
        """
        self.building_data: Dict[str, Any] = {}
        self.building_list: List[str] = []
        self.available = False
        
        if file_dicts:
            self._load_building_data(file_dicts)
    
    def _load_building_data(self, file_dicts: List[Dict[str, Any]]):
        """ファイルからビルマスターJSONを読み込み"""
        
        logger.info("🔍 ビルマスター検索開始 - ファイル数: %d", len(file_dicts))
        
        # デバッグ: 全ファイル名を表示
        for i, file_dict in enumerate(file_dicts):
            filename = file_dict.get("name", "")
            logger.info("🔍 ファイル %d: %s", i+1, filename)
        
        # 三菱地所ビルマスター.jsonを探す
        building_master_file = None
        for file_dict in file_dicts:
            filename = file_dict.get("name", "")
            logger.info("🔍 チェック中: %s", filename)
            
            # 🔥 文字コード詳細デバッグ
            logger.info("🔍 ファイル名の文字コード詳細:")
            for i, char in enumerate(filename):
                logger.info("  文字 %d: '%s' (ord=%d)", i, char, ord(char))
            
            # 🔥 検索文字列の文字コード
            search_text = "三菱地所ビルマスター"
            logger.info("🔍 検索文字列の文字コード詳細:")
            for i, char in enumerate(search_text):
                logger.info("  文字 %d: '%s' (ord=%d)", i, char, ord(char))
            
            # 🔥 修正: 非常に緩い検索条件
            filename_lower = filename.lower()
            
            # 条件: "ビルマスター" が含まれているだけでOK
            contains_master = "ビルマスター" in filename
            
            logger.info("🔍   - 'ビルマスター' 含有: %s", contains_master)
            
            # 🔥 さらに詳細デバッグ
            if "ビルマスター" in filename:
                logger.info("✅ ビルマスターファイル発見: %s", filename)
                building_master_file = file_dict
                break
            else:
                logger.info("❌ マッチしない: %s", filename)
                
                # 🔥 文字ごとの部分一致も試行
                if "ビル" in filename:
                    logger.info("🔍 'ビル' は含まれています")
                if "マスター" in filename:
                    logger.info("🔍 'マスター' は含まれています")
                if ".json" in filename:
                    logger.info("🔍 '.json' は含まれています")
                
                # 🔥 最終手段: ファイル名に json が含まれていれば対象とする
                if ".json" in filename.lower():
                    logger.info("🎯 JSONファイルとして強制採用: %s", filename)
                    building_master_file = file_dict
                    break
        
        if not building_master_file:
            logger.warning("⚠️ 三菱地所ビルマスター.json が見つかりません")
            logger.warning("📝 検索条件: ファイル名に'三菱地所ビルマスター'を含み、'.json'で終わるファイル")
            
            # 🔥 追加: JSONファイルの一覧を表示
            json_files = [f.get("name", "") for f in file_dicts if f.get("name", "").endswith(".json")]
            logger.warning("📝 利用可能なJSONファイル: %s", json_files)
            return
        
        try:
            # JSONデータを読み込み
            file_data = building_master_file.get("data", b"")
            logger.info("📄 JSONデータサイズ: %d bytes", len(file_data))
            
            if len(file_data) == 0:
                logger.error("❌ JSONファイルが空です")
                return
            
            # 文字エンコーディングを試行
            try:
                json_text = file_data.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("⚠️ UTF-8デコードに失敗、shift_jisを試行")
                json_text = file_data.decode("shift_jis")
            
            logger.info("📄 JSON文字列長: %d", len(json_text))
            logger.info("📄 JSON先頭100文字: %s", json_text[:100])
            
            json_data = json.loads(json_text)
            logger.info("📄 JSON解析成功")
            
            self.building_data = json_data
            
            # 🔥 JSONデータ構造を詳細に調査
            logger.info("📊 JSONデータ構造調査:")
            logger.info("  - データ型: %s", type(json_data))
            
            if isinstance(json_data, dict):
                logger.info("  - 辞書のキー数: %d", len(json_data))
                logger.info("  - キー一覧: %s", list(json_data.keys())[:10])  # 最初の10個のキー
                
                # 🔥 修正: 辞書形式でも各ビルの詳細情報から名前を取得
                self.building_list = []
                for key, building_info in json_data.items():
                    logger.info("  - キー '%s' の値の型: %s", key, type(building_info))
                    
                    if isinstance(building_info, dict):
                        logger.info("    - 内包するキー: %s", list(building_info.keys())[:5])
                        
                        # 優先順位: 略称 → toko建物コード → キー名
                        if "略称" in building_info and building_info["略称"]:
                            name = building_info["略称"]
                            logger.info("    - 略称を使用: %s", name)
                        elif "toko建物コード" in building_info and building_info["toko建物コード"]:
                            name = building_info["toko建物コード"]
                            logger.info("    - toko建物コードを使用: %s", name)
                        elif "toko建物コードNo." in building_info and building_info["toko建物コードNo."]:
                            name = f"ビル{building_info['toko建物コードNo.']}"
                            logger.info("    - toko建物コードNo.を使用: %s", name)
                        else:
                            name = key  # フォールバック: キー名
                            logger.info("    - キー名を使用: %s", name)
                        
                        self.building_list.append(name)
                    else:
                        # 値が辞書でない場合はキー名を使用
                        self.building_list.append(key)
                        logger.warning("    - 値が辞書でないためキー名を使用: %s", key)
                
                logger.info("📄 辞書形式のJSONデータ - 抽出されたビル名数: %d", len(self.building_list))
                
            elif isinstance(json_data, list):
                logger.info("  - リストの要素数: %d", len(json_data))
                
                # 最初の要素の構造を確認
                if len(json_data) > 0:
                    first_item = json_data[0]
                    logger.info("  - 最初の要素の型: %s", type(first_item))
                    if isinstance(first_item, dict):
                        logger.info("  - 最初の要素のキー: %s", list(first_item.keys()))
                        # 略称があるかチェック
                        if "略称" in first_item:
                            logger.info("  - 最初の要素の略称: %s", first_item["略称"])
                
                # ビル情報のリストの場合
                self.building_list = []
                for i, building in enumerate(json_data):
                    if isinstance(building, dict):
                        # 略称を最優先、なければtoko建物コード、なければインデックス
                        if "略称" in building and building["略称"]:
                            name = building["略称"]
                        elif "toko建物コード" in building and building["toko建物コード"]:
                            name = building["toko建物コード"]
                        elif "toko建物コードNo." in building and building["toko建物コードNo."]:
                            name = f"ビル{building['toko建物コードNo.']}"
                        else:
                            name = f"ビル{i+1}"
                        
                        self.building_list.append(name)
                        logger.info("  - ビル %d: %s", i+1, name)
                    else:
                        logger.warning("  - 要素 %d は辞書ではありません: %s", i, type(building))
                
                logger.info("📄 リスト形式のJSONデータ - 要素数: %d", len(self.building_list))
                
            else:
                logger.warning("⚠️ 予期しないJSONデータ形式: %s", type(json_data))
                logger.info("📊 データの内容（最初の200文字）: %s", str(json_data)[:200])
                return
            
            self.available = True
            logger.info("✅ ビルマスターデータ読み込み成功: %d件のビル情報", len(self.building_list))
            logger.info("📋 ビル一覧: %s", self.building_list[:10])  # 最初の10件を表示
            
        except json.JSONDecodeError as e:
            logger.error("❌ JSON解析エラー: %s", e)
            logger.error("❌ JSON文字列の一部: %s", json_text[:200] if 'json_text' in locals() else "取得できません")
            self.available = False
        except Exception as e:
            logger.error("❌ ビルマスターデータ読み込み失敗: %s", e)
            self.available = False
    
    def get_building_list(self) -> List[str]:
        """利用可能なビル一覧を取得"""
        return self.building_list.copy()
    
    def get_building_info(self, building_name: str = None) -> Optional[Dict[str, Any]]:
        """
        指定されたビルの情報を取得
        
        Args:
            building_name: ビル名（Noneの場合は最初のビル）
            
        Returns:
            ビル情報辞書またはNone
        """
        if not self.available:
            logger.warning("🔍 get_building_info: データが利用できません")
            return None
        
        logger.info("🔍 get_building_info: 検索対象='%s'", building_name)
        logger.info("🔍 利用可能なビル一覧: %s", self.building_list)
        
        if isinstance(self.building_data, dict):
            if building_name is None:
                # 最初のビルを返す
                first_value = next(iter(self.building_data.values())) if self.building_data else None
                logger.info("🔍 最初のビルを返却: %s", first_value is not None)
                return first_value
            
            # 🔥 複数の方法でビル情報を検索
            logger.info("🔍 辞書形式での検索開始")
            
            # 方法1: 直接キーで検索
            if building_name in self.building_data:
                logger.info("✅ 直接キーで発見: %s", building_name)
                return self.building_data[building_name]
            
            # 方法2: 各ビル情報の略称・コードで検索
            for key, building_info in self.building_data.items():
                if isinstance(building_info, dict):
                    # 略称での検索
                    if "略称" in building_info and building_info["略称"] == building_name:
                        logger.info("✅ 略称で発見: key='%s', 略称='%s'", key, building_info["略称"])
                        return building_info
                    
                    # toko建物コードでの検索
                    if "toko建物コード" in building_info and building_info["toko建物コード"] == building_name:
                        logger.info("✅ toko建物コードで発見: key='%s', コード='%s'", key, building_info["toko建物コード"])
                        return building_info
                    
                    # toko建物コードNo.での検索
                    if "toko建物コードNo." in building_info:
                        expected_name = f"ビル{building_info['toko建物コードNo.']}"
                        if expected_name == building_name:
                            logger.info("✅ toko建物コードNo.で発見: key='%s', No.='%s'", key, building_info["toko建物コードNo."])
                            return building_info
            
            # 方法3: 部分一致検索
            for key, building_info in self.building_data.items():
                if isinstance(building_info, dict):
                    # 略称での部分一致
                    if "略称" in building_info and building_info["略称"]:
                        if building_name in building_info["略称"] or building_info["略称"] in building_name:
                            logger.info("🔍 略称部分一致で発見: key='%s', 略称='%s'", key, building_info["略称"])
                            return building_info
            
            logger.warning("❌ 辞書形式で見つかりません: %s", building_name)
            return None
        
        elif isinstance(self.building_data, list):
            if building_name is None:
                # 最初のビルを返す
                first_item = self.building_data[0] if self.building_data else None
                logger.info("🔍 最初のビルを返却: %s", first_item is not None)
                return first_item
            
            # 🔥 リスト形式での検索
            logger.info("🔍 リスト形式での検索開始")
            
            for i, building in enumerate(self.building_data):
                if isinstance(building, dict):
                    # 略称での検索
                    if "略称" in building and building["略称"] == building_name:
                        logger.info("✅ 略称で発見: index=%d, 略称='%s'", i, building["略称"])
                        return building
                    
                    # toko建物コードでの検索
                    if "toko建物コード" in building and building["toko建物コード"] == building_name:
                        logger.info("✅ toko建物コードで発見: index=%d, コード='%s'", i, building["toko建物コード"])
                        return building
                    
                    # toko建物コードNo.での検索
                    if "toko建物コードNo." in building:
                        expected_name = f"ビル{building['toko建物コードNo.']}"
                        if expected_name == building_name:
                            logger.info("✅ toko建物コードNo.で発見: index=%d, No.='%s'", i, building["toko建物コードNo."])
                            return building
            
            logger.warning("❌ リスト形式で見つかりません: %s", building_name)
            return None
        
        logger.warning("❌ 未対応のデータ形式")
        return None
    
    def get_all_buildings_info(self) -> Dict[str, Any]:
        """全ビル情報を取得"""
        if not self.available:
            return {}
        
        if isinstance(self.building_data, dict):
            return self.building_data.copy()
        elif isinstance(self.building_data, list):
            # リスト形式の場合は略称をキーとした辞書に変換
            return {
                building.get("略称", f"ビル{i+1}"): building
                for i, building in enumerate(self.building_data)
            }
        
        return {}
    
    def format_building_info_for_prompt(self, building_name: str = None) -> str:
        """
        ビル情報をプロンプト用にフォーマット
        
        Args:
            building_name: ビル名（Noneの場合は全ビル情報）
            
        Returns:
            フォーマットされたビル情報文字列
        """
        if not self.available:
            return "【ビル情報】利用可能なビル情報がありません。"
        
        if building_name:
            # 特定のビル情報
            building_info = self.get_building_info(building_name)
            if not building_info:
                return f"【ビル情報】指定されたビル「{building_name}」の情報が見つかりません。"
            
            return self._format_single_building(building_name, building_info)
        
        else:
            # 全ビル情報
            all_buildings = self.get_all_buildings_info()
            if not all_buildings:
                return "【ビル情報】利用可能なビル情報がありません。"
            
            formatted_parts = []
            for bldg_name, bldg_info in all_buildings.items():
                formatted_parts.append(self._format_single_building(bldg_name, bldg_info))
            
            return "\n\n".join(formatted_parts)
    
    def _format_single_building(self, building_name: str, building_info: Dict[str, Any]) -> str:
        """単一ビル情報をフォーマット"""
        
        lines = [f"【ビル情報：{building_name}】"]
        
        # 基本情報
        if "toko建物コードNo." in building_info:
            lines.append(f"- 建物コードNo.: {building_info['toko建物コードNo.']}")
        if "toko建物コード" in building_info:
            lines.append(f"- 建物コード: {building_info['toko建物コード']}")
        if "略称" in building_info:
            lines.append(f"- 略称: {building_info['略称']}")
        
        # 基準階プラン
        if "基準階プラン" in building_info:
            plan = building_info["基準階プラン"]
            lines.append("- 基準階プラン:")
            if "床面積" in plan:
                lines.append(f"  - 床面積: {plan['床面積']}")
            if "天井高" in plan:
                lines.append(f"  - 天井高: {plan['天井高']}")
            if "OAフロア" in plan:
                lines.append(f"  - OAフロア: {plan['OAフロア']}")
        
        # 基準階材料
        if "基準階材料" in building_info:
            materials = building_info["基準階材料"]
            lines.append("- 基準階材料:")
            
            # 自動火災報知設備
            if "自動火災報知設備(基準階)" in materials:
                fire_alarm = materials["自動火災報知設備(基準階)"]
                lines.append("  - 自動火災報知設備(基準階):")
                if "メーカー" in fire_alarm:
                    lines.append(f"    - メーカー: {fire_alarm['メーカー']}")
                if "感知器種別" in fire_alarm:
                    lines.append(f"    - 感知器種別: {fire_alarm['感知器種別']}")
            
            # 非常放送
            if "非常放送(基準階)" in materials:
                broadcast = materials["非常放送(基準階)"]
                lines.append("  - 非常放送(基準階):")
                if "メーカー" in broadcast:
                    lines.append(f"    - メーカー: {broadcast['メーカー']}")
                if "スピーカー種別" in broadcast:
                    lines.append(f"    - スピーカー種別: {broadcast['スピーカー種別']}")
            
            # 非常照明
            if "非常照明(基準階)" in materials:
                emergency_light = materials["非常照明(基準階)"]
                lines.append("  - 非常照明(基準階):")
                if "メーカー" in emergency_light:
                    lines.append(f"    - メーカー: {emergency_light['メーカー']}")
                if "照明器具種別" in emergency_light:
                    lines.append(f"    - 照明器具種別: {emergency_light['照明器具種別']}")
            
            # 誘導灯
            if "誘導灯(基準階)" in materials:
                guide_light = materials["誘導灯(基準階)"]
                lines.append("  - 誘導灯(基準階):")
                if "メーカー" in guide_light:
                    lines.append(f"    - メーカー: {guide_light['メーカー']}")
                if "型式" in guide_light:
                    lines.append(f"    - 型式: {guide_light['型式']}")
        
        # 概要
        if "概要" in building_info:
            summary = building_info["概要"]
            lines.append("- 概要:")
            if "所在地" in summary:
                lines.append(f"  - 所在地: {summary['所在地']}")
            if "用途区分(消防)" in summary:
                lines.append(f"  - 用途区分(消防): {summary['用途区分(消防)']}")
        
        return "\n".join(lines)
    
    def search_building_by_keyword(self, keyword: str) -> List[str]:
        """
        キーワードでビルを検索
        
        Args:
            keyword: 検索キーワード
            
        Returns:
            マッチしたビル名のリスト
        """
        if not self.available:
            return []
        
        keyword_lower = keyword.lower()
        matched_buildings = []
        
        all_buildings = self.get_all_buildings_info()
        for building_name, building_info in all_buildings.items():
            # ビル名で検索
            if keyword_lower in building_name.lower():
                matched_buildings.append(building_name)
                continue
            
            # 略称で検索
            if "略称" in building_info and keyword_lower in building_info["略称"].lower():
                matched_buildings.append(building_name)
                continue
            
            # 所在地で検索
            if ("概要" in building_info and 
                "所在地" in building_info["概要"] and 
                keyword_lower in building_info["概要"]["所在地"].lower()):
                matched_buildings.append(building_name)
        
        return matched_buildings

# グローバルインスタンス管理
_building_manager: Optional[BuildingManager] = None

def get_building_manager() -> Optional[BuildingManager]:
    """BuildingManagerのインスタンスを取得"""
    return _building_manager

def initialize_building_manager(file_dicts: List[Dict[str, Any]]) -> BuildingManager:
    """BuildingManagerを初期化"""
    global _building_manager
    _building_manager = BuildingManager(file_dicts)
    return _building_manager

def format_all_buildings_for_prompt() -> str:
    """全ビル情報をプロンプト用にフォーマット（便利関数）"""
    manager = get_building_manager()
    if manager:
        return manager.format_building_info_for_prompt()
    return "【ビル情報】利用可能なビル情報がありません。"

def format_building_for_prompt(building_name: str) -> str:
    """特定ビル情報をプロンプト用にフォーマット（便利関数）"""
    manager = get_building_manager()
    if manager:
        return manager.format_building_info_for_prompt(building_name)
    return f"【ビル情報】ビル「{building_name}」の情報が見つかりません。"