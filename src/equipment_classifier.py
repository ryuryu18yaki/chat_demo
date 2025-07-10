# src/equipment_classifier.py

import re
from typing import Dict, List, Optional

def extract_equipment_from_filename(filename: str) -> Optional[str]:
    """
    ファイル名から設備名を抽出する
    
    Args:
        filename: ファイル名（例：「消防法による防災設備の設置基準.pdf」）
    
    Returns:
        設備名（例：「自動火災報知設備」）または None
    """
    
    # 設備名のマッピング辞書（優先度順・実際のファイル名に基づく）
    equipment_patterns = {
        # 自動火災報知設備
        "自動火災報知設備": [
            r"PanasonicWEBカタログ設置基準",
            r"日本火災報知器工業会 自動火災報知設備の設置基準",
            r"東京消防庁 自動火災報知設備",
            r"能美防災 警報設備早見表",
            r"カタログ",
            r"煙感知",
            r"熱感知",
            r"防災設備"
        ],
        
        # 非常放送設備
        "非常放送設備": [
            r"TOA 非常用放送設備マニュアル",
            r"TOA 非常用・業務用放送設備システム",
            r"UNI-PEX 非常放送設備",
            r"スピーカ",
            r"放送設備"
        ],
        
        # 誘導灯設備
        "誘導灯設備": [
            r"Panasonic誘導灯4P(区分～免除～設置基準～設置)",
            r"東京消防庁 誘導灯及び誘導標識",
            r"誘導灯",
            r"避難誘導"
        ],
        
        # 非常照明設備  
        "非常照明設備": [
            r"岩崎電気株式会社 ライティング講座（照明講座）",
            r"アイリスオーヤマ 非常灯（非常用照明器具）の設置基準",
            r"設置基準について.*非常灯.*LED非常用照明器具.*施設用照明.*アイリスオーヤマ",
            r"非常用照明器具",
            r"防災照明",
            r"岩崎電気",
            r"非常灯",
            r"LED非常用照明器具",
            r"施設用照明",
            r"アイリスオーヤマ",
            r"非常照明",
            r"非常用照明",
            r"照明設計"
        ]
    }
    
    # ファイル名を正規化（拡張子除去、小文字化）
    normalized_name = filename.lower()
    if '.' in normalized_name:
        normalized_name = normalized_name.rsplit('.', 1)[0]
    
    # パターンマッチング（優先度順）
    for equipment_name, patterns in equipment_patterns.items():
        for pattern in patterns:
            if re.search(pattern, normalized_name):
                return equipment_name
    
    # マッチしない場合は「その他」
    return "その他"

def get_equipment_category(equipment_name: str) -> str:
    """
    設備名からカテゴリを取得
    
    Args:
        equipment_name: 設備名
        
    Returns:
        カテゴリ名
    """
    categories = {
        "消防設備": [
            "自動火災報知設備",
            "非常放送設備", 
            "誘導灯設備",
            "非常照明設備"
        ]
    }
    
    for category, equipments in categories.items():
        if equipment_name in equipments:
            return category
    
    return "その他設備"

# 使用例とテスト
if __name__ == "__main__":
    test_files = [
        "消防法による防災設備の設置基準.pdf",
        "自動火災報知設備の設置基準.pdf",
        "警報設備早見表.pdf",
        "非常放送設備マニュアル.pdf", 
        "非常放送設備.pdf",
        "照明設計資料.pdf",
        "誘導灯及び誘導標識.pdf",
        "非常用照明器具 * 防災照明 * 岩崎電気.pdf",
        "設置基準について｜非常灯(LED非常用照明器具)｜施設用照明｜アイリスオーヤマ.pdf",
        "その他の資料.pdf"
    ]
    
    for filename in test_files:
        equipment = extract_equipment_from_filename(filename)
        category = get_equipment_category(equipment)
        print(f"{filename} → 設備: {equipment}, カテゴリ: {category}")