import re
from typing import Dict, List, Optional

def extract_fire_department_info(filename: str) -> Dict[str, Optional[str]]:
    """
    ファイル名から消防庁の管轄情報を抽出する
    
    Args:
        filename: ファイル名
    
    Returns:
        {
            "jurisdiction": "東京消防庁" | "丸の内消防署" | None,
            "is_fire_dept": True/False,
            "is_general": True/False  # どの管轄でも使用する資料かどうか
        }
    """
    
    # ファイル名を正規化
    normalized_name = filename.lower()
    
    result = {
        "jurisdiction": None,
        "is_fire_dept": False,
        "is_general": False
    }
    
    # 1. 丸の内消防署の判定（優先度高）
    marunouchi_patterns = [
        r"丸の内",
        r"marunouchi"
    ]
    
    for pattern in marunouchi_patterns:
        if re.search(pattern, normalized_name):
            result["jurisdiction"] = "丸の内消防署"
            result["is_fire_dept"] = True
            result["is_general"] = False
            return result
    
    # 2. 東京消防庁の判定
    tokyo_fire_patterns = [
        r"東京消防庁",
        r"tokyo.*消防",
        r"消防庁.*東京"
    ]
    
    for pattern in tokyo_fire_patterns:
        if re.search(pattern, normalized_name):
            result["jurisdiction"] = "東京消防庁"
            result["is_fire_dept"] = True
            result["is_general"] = False
            return result
    
    # 3. 一般的な消防関連キーワード（どの管轄でも使用）
    general_fire_patterns = [
        r"消防法",
        r"防災設備.*設置.*基準",
        r"消防設備.*基準",
        r"火災報知.*設置.*基準",
        r"消防.*法令",
        r"防災.*ハンドブック"
    ]
    
    for pattern in general_fire_patterns:
        if re.search(pattern, normalized_name):
            result["jurisdiction"] = None  # 管轄指定なし
            result["is_fire_dept"] = True
            result["is_general"] = True  # どの管轄でも使用可能
            return result
    
    return result

def classify_files_by_jurisdiction(file_dicts: List[Dict]) -> Dict:
    """
    ファイルリストを管轄別に分類
    
    Args:
        file_dicts: ファイル辞書のリスト
        
    Returns:
        {
            "jurisdictions": {
                "東京消防庁": [files...],
                "丸の内消防署": [files...]
            },
            "general_fire": [files...],     # どの管轄でも使用する消防関連ファイル
            "equipment_files": [files...]   # 一般的な設備ファイル
        }
    """
    classified = {
        "jurisdictions": {
            "東京消防庁": [],
            "丸の内消防署": []
        },
        "general_fire": [],    # どの管轄でも使用する消防関連
        "equipment_files": []  # 一般的な設備ファイル
    }
    
    for file_dict in file_dicts:
        filename = file_dict.get("name", "")
        fire_info = extract_fire_department_info(filename)
        
        # ファイルに分類情報を追加
        file_dict_with_info = file_dict.copy()
        file_dict_with_info.update(fire_info)
        
        if fire_info["is_fire_dept"]:
            if fire_info["is_general"]:
                # どの管轄でも使用する消防関連資料
                classified["general_fire"].append(file_dict_with_info)
            elif fire_info["jurisdiction"]:
                # 特定管轄の資料
                jurisdiction = fire_info["jurisdiction"]
                classified["jurisdictions"][jurisdiction].append(file_dict_with_info)
            else:
                # 消防関連だが分類不明 → 一般設備として扱う
                classified["equipment_files"].append(file_dict_with_info)
        else:
            # 消防関連ではない一般設備ファイル
            classified["equipment_files"].append(file_dict_with_info)
    
    return classified

def get_jurisdiction_stats(classified_data: Dict) -> Dict:
    """
    管轄別ファイルの統計情報
    """
    jurisdictions = classified_data["jurisdictions"]
    
    return {
        "東京消防庁_ファイル数": len(jurisdictions["東京消防庁"]),
        "丸の内消防署_ファイル数": len(jurisdictions["丸の内消防署"]),
        "一般消防資料_ファイル数": len(classified_data["general_fire"]),
        "設備ファイル数": len(classified_data["equipment_files"]),
        "消防関連総数": (
            len(jurisdictions["東京消防庁"]) + 
            len(jurisdictions["丸の内消防署"]) + 
            len(classified_data["general_fire"])
        )
    }

def get_files_for_jurisdiction(classified_data: Dict, selected_jurisdiction: str) -> List[Dict]:
    """
    指定された管轄に応じたファイル一覧を取得
    
    Args:
        classified_data: classify_files_by_jurisdiction()の戻り値
        selected_jurisdiction: "東京消防庁" | "丸の内消防署" | None
        
    Returns:
        対象ファイルのリスト
    """
    result_files = []
    
    # 常に一般設備ファイルは含める
    result_files.extend(classified_data["equipment_files"])
    
    # 常に一般消防資料も含める（どの管轄でも使用）
    result_files.extend(classified_data["general_fire"])
    
    # 指定された管轄の資料を追加
    if selected_jurisdiction and selected_jurisdiction in classified_data["jurisdictions"]:
        result_files.extend(classified_data["jurisdictions"][selected_jurisdiction])
    
    return result_files

# テスト用
if __name__ == "__main__":
    test_files = [
        "東京消防庁_自動火災報知設備.pdf",
        "丸の内_防災設備点検.pdf", 
        "消防法による防災設備の設置基準.pdf",  # 一般消防資料
        "Panasonic照明カタログ.pdf",  # 一般設備
        "非常放送設備マニュアル.pdf"  # 一般設備
    ]
    
    print("=== 管轄別ファイル分類テスト ===")
    for filename in test_files:
        fire_info = extract_fire_department_info(filename)
        print(f"{filename}")
        print(f"  → 管轄: {fire_info['jurisdiction']}")
        print(f"  → 消防関連: {fire_info['is_fire_dept']}")
        print(f"  → 一般資料: {fire_info['is_general']}")
        print()