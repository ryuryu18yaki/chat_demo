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
        
        # 三菱地所ビルマスター.jsonを探す
        building_master_file = None
        for file_dict in file_dicts:
            filename = file_dict.get("name", "")
            if "三菱地所ビルマスター" in filename and filename.endswith(".json"):
                building_master_file = file_dict
                break
        
        if not building_master_file:
            logger.warning("⚠️ 三菱地所ビルマスター.json が見つかりません")
            return
        
        try:
            # JSONデータを読み込み
            file_data = building_master_file.get("data", b"")
            json_data = json.loads(file_data.decode("utf-8"))
            self.building_data = json_data
            
            # ビル一覧を生成（キーまたは略称から）
            if isinstance(json_data, dict):
                # ビル名をキーとした辞書の場合
                self.building_list = list(json_data.keys())
            elif isinstance(json_data, list):
                # ビル情報のリストの場合
                self.building_list = [
                    building.get("略称", f"ビル{i+1}")
                    for i, building in enumerate(json_data)
                ]
            
            self.available = True
            logger.info("✅ ビルマスターデータ読み込み成功: %d件のビル情報", len(self.building_list))
            
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
            return None
        
        if isinstance(self.building_data, dict):
            if building_name is None:
                # 最初のビルを返す
                return next(iter(self.building_data.values())) if self.building_data else None
            return self.building_data.get(building_name)
        
        elif isinstance(self.building_data, list):
            if building_name is None:
                # 最初のビルを返す
                return self.building_data[0] if self.building_data else None
            
            # 略称で検索
            for building in self.building_data:
                if building.get("略称") == building_name:
                    return building
        
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