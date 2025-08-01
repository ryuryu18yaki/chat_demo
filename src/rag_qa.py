# src/rag_qa.py - AWS Bedrock Claude + Azure OpenAI GPT版

from __future__ import annotations
from typing import List, Dict, Any, Optional
import boto3
import json
import os

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Azure OpenAI関連のインポートを追加
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

# 🔥 ビル情報マネージャーのインポートを追加
from src.building_manager import get_building_manager

# ---------------------------------------------------------------------------
# AWS Bedrock設定
# ---------------------------------------------------------------------------
def create_bedrock_client():
    """AWS Bedrock クライアントを作成"""
    if STREAMLIT_AVAILABLE:
        try:
            aws_access_key_id = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
            aws_secret_access_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
            aws_region = st.secrets.get("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
        except:
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
    else:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS Bedrock の設定が不足しています。環境変数を確認してください。")
    
    return boto3.client(
        'bedrock-runtime',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

# ---------------------------------------------------------------------------
# Azure OpenAI設定
# ---------------------------------------------------------------------------
def create_azure_client():
    """Azure OpenAI クライアントを作成"""
    if not AZURE_OPENAI_AVAILABLE:
        raise ValueError("Azure OpenAI ライブラリがインストールされていません。pip install openai を実行してください。")
    
    if STREAMLIT_AVAILABLE:
        try:
            azure_endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
            azure_api_key = st.secrets.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))
            azure_api_version = st.secrets.get("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
        except:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    else:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    if not azure_endpoint or not azure_api_key:
        raise ValueError("Azure OpenAI の設定が不足しています。環境変数を確認してください。")
    
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version
    )

# ---------------------------------------------------------------------------
# モデル管理
# ---------------------------------------------------------------------------

# Claude用のモデル名マッピング
CLAUDE_MODEL_MAPPING = {
    "claude-4-sonnet": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-3.7": "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"
}

# Azure OpenAI用のモデル名マッピングを追加
AZURE_MODEL_MAPPING = {
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o"
}

def get_claude_model_name(model_name: str) -> str:
    """Claude表示名をBedrockモデルIDに変換"""
    return CLAUDE_MODEL_MAPPING.get(model_name, model_name)

def call_claude_bedrock(client, model_id: str, messages: List[Dict], max_tokens: int = None, temperature: float = None):
    """AWS Bedrock Converse API経由でClaudeを呼び出し"""
    
    # メッセージ形式をConverse APIに合わせて変換
    system_prompts = []
    conversation_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_prompts.append({"text": msg["content"]})
        else:
            conversation_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })
    
    # Converse API用のパラメータを構築
    converse_params = {
        "modelId": model_id,
        "messages": conversation_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens or 4096  # max_tokensが指定されていればそれを使用、なければ4096
        }
    }
    
    # temperatureが指定されている場合のみ設定
    if temperature is not None and temperature != 0.0:
        converse_params["inferenceConfig"]["temperature"] = temperature
    
    # システムプロンプトがある場合は追加
    if system_prompts:
        converse_params["system"] = system_prompts
    
    # Converse API呼び出し
    response = client.converse(**converse_params)
    
    return response['output']['message']['content'][0]['text']

def call_azure_gpt(client, model_name: str, messages: List[Dict], max_tokens: int = None, temperature: float = None):
    """Azure OpenAI経由でGPTを呼び出し"""
    formatted_messages = []
    
    for msg in messages:
        if msg["role"] in ["system", "user", "assistant"]:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    api_params = {
        "model": AZURE_MODEL_MAPPING.get(model_name, model_name),
        "messages": formatted_messages,
        "max_tokens": max_tokens or 4096  # Noneの場合は4096をデフォルトに
    }
    
    if temperature is not None and temperature != 0.0:
        api_params["temperature"] = temperature
    
    response = client.chat.completions.create(**api_params)
    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "claude-4-sonnet"

# ---------------------------------------------------------------------------
# 回答生成（設備全文投入版）
# ---------------------------------------------------------------------------

def generate_answer_with_equipment(
        *,
        prompt: str,
        question: str,
        equipment_data: Dict[str, Dict[str, Any]],
        target_equipment: str,
        selected_files: Optional[List[str]] = None,
        model: str = _DEFAULT_MODEL,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_building_info: bool = True,  # 🔥 新規追加
        target_building: Optional[str] = None,  # 🔥 新規追加
    ) -> Dict[str, Any]:
    """
    指定された設備の選択ファイルをプロンプトに投入してAIで回答を生成
    
    Args:
        prompt: システムプロンプト
        question: ユーザーの質問
        equipment_data: 設備データ辞書（preprocess_filesの出力）
        target_equipment: 対象設備名
        selected_files: 使用するファイル名のリスト（Noneなら全ファイル）
        model: 使用するモデル
        chat_history: チャット履歴
        temperature: 温度パラメータ
        max_tokens: 最大トークン数
        include_building_info: ビル情報を含めるかどうか  # 🔥 新規追加
        target_building: 対象ビル名（Noneなら全ビル情報）  # 🔥 新規追加
    
    Returns:
        回答結果辞書
    """
    
    # --- 1) 指定設備のデータを取得 ---
    if target_equipment not in equipment_data:
        available_equipment = list(equipment_data.keys())
        raise ValueError(f"設備 '{target_equipment}' が見つかりません。利用可能な設備: {available_equipment}")
    
    equipment_info = equipment_data[target_equipment]
    available_files = equipment_info["files"]  # ファイル名 → テキストの辞書
    all_sources = equipment_info["sources"]
    
    # 🔥 選択されたファイルのみを結合
    if selected_files is not None:
        print(f"🔧 使用設備: {target_equipment}")
        print(f"📄 選択ファイル: {', '.join(selected_files)}")
        print(f"📄 利用可能ファイル: {', '.join(all_sources)}")
        
        # 選択されたファイルのテキストを結合
        selected_texts = []
        actual_sources = []
        
        for file_name in selected_files:
            if file_name in available_files:
                selected_texts.append(available_files[file_name])
                actual_sources.append(file_name)
            else:
                print(f"⚠️ ファイルが見つかりません: {file_name}")
        
        if not selected_texts:
            raise ValueError(f"選択されたファイル（{', '.join(selected_files)}）が設備データに見つかりません。")
        
        combined_text = "\n\n".join(selected_texts)
        sources = actual_sources
        
        print(f"📝 結合後文字数: {len(combined_text)}")
        
    else:
        # 全ファイル使用
        selected_texts = list(available_files.values())
        combined_text = "\n\n".join(selected_texts)
        sources = all_sources
        
        print(f"🔧 使用設備: {target_equipment}")
        print(f"📄 全ファイル使用: {', '.join(sources)}")
        print(f"📝 結合後文字数: {len(combined_text)}")
    
    # 🔥 --- 2) ビル情報を取得してプロンプトに追加 ---
    building_info_text = ""
    if include_building_info:
        building_manager = get_building_manager()
        if building_manager and building_manager.available:
            if target_building:
                building_info_text = building_manager.format_building_info_for_prompt(target_building)
                print(f"🏢 対象ビル情報: {target_building}")
            else:
                building_info_text = building_manager.format_building_info_for_prompt()
                building_count = len(building_manager.get_building_list())
                print(f"🏢 全ビル情報使用: {building_count}件")
        else:
            print("⚠️ ビル情報が利用できません")
            building_info_text = "【ビル情報】利用可能なビル情報がありません。"
    
    # --- 3) プロンプト組み立て ---
    context_parts = []
    
    # ビル情報を最初に配置
    if building_info_text:
        context_parts.append(building_info_text)
    
    # 設備資料情報
    equipment_context = f"""
【参考資料】設備: {target_equipment} (カテゴリ: {equipment_info['equipment_category']})
使用ファイル: {', '.join(sources)}
使用ファイル数: {len(sources)}/{len(all_sources)}
【注意事項】
**暗黙知メモに関して、ページ番号などの情報は出力を禁止します。**

【資料内容】
{combined_text}
"""
    context_parts.append(equipment_context)
    
    # 全体のコンテキストを結合
    full_context = "\n\n".join(context_parts)
    
    system_msg = {
        "role": "system",
        "content": prompt
    }
    
    user_msg = {
        "role": "user", 
        "content": f"{full_context}\n\n【質問】\n{question}\n\n上記のビル情報と資料を参考に、日本語で回答してください。"
    }
    
    # --- 4) Messages 組み立て ---
    messages: List[Dict[str, Any]] = []
    
    # チャット履歴があれば追加
    if chat_history:
        messages.append(system_msg)
        # 安全な履歴のみ追加
        safe_history = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in chat_history
            if isinstance(m, dict) and m.get("role") and m.get("content")
        ]
        messages.extend(safe_history[:-1])  # 最後の質問は除く（新しい質問で上書き）
        messages.append(user_msg)
    else:
        messages = [system_msg, user_msg]
    
    # --- 5) AI モデル呼び出し ---
    try:
        print(f"🤖 API呼び出し開始 - モデル: {model}")
        
        if model.startswith("gpt"):
            # Azure OpenAI GPT
            azure_client = create_azure_client()
            answer = call_azure_gpt(
                azure_client,
                model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            # AWS Bedrock Claude
            bedrock_client = create_bedrock_client()
            model_id = get_claude_model_name(model)
            answer = call_claude_bedrock(
                bedrock_client,
                model_id, 
                messages,
                max_tokens=max_tokens,
                temperature=temperature if temperature != 0.0 else None
            )
        
        print(f"✅ 回答生成完了 - 回答文字数: {len(answer)}")
        
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")
        raise
    
    # 🔥 結果にビル情報も含める
    result = {
        "answer": answer,
        "used_equipment": target_equipment,
        "equipment_info": equipment_info,
        "sources": sources,
        "selected_files": selected_files,
        "context_length": len(full_context),  # 🔥 ビル情報込みの長さ
        "building_info_included": include_building_info,  # 🔥 新規追加
        "target_building": target_building,  # 🔥 新規追加
        "images": []  # 現バージョンでは画像は対応しない
    }
    
    return result

# 🔥 ビル情報なしでの回答生成関数も追加
def generate_answer_without_rag(
        *,
        prompt: str,
        question: str,
        model: str = _DEFAULT_MODEL,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_building_info: bool = True,  # 🔥 新規追加
        target_building: Optional[str] = None,  # 🔥 新規追加
    ) -> Dict[str, Any]:
    """
    設備資料なしでビル情報のみを使用して回答生成
    """
    
    # 🔥 ビル情報を取得
    building_info_text = ""
    if include_building_info:
        building_manager = get_building_manager()
        if building_manager and building_manager.available:
            if target_building:
                building_info_text = building_manager.format_building_info_for_prompt(target_building)
                print(f"🏢 対象ビル情報: {target_building}")
            else:
                building_info_text = building_manager.format_building_info_for_prompt()
                building_count = len(building_manager.get_building_list())
                print(f"🏢 全ビル情報使用: {building_count}件")
        else:
            print("⚠️ ビル情報が利用できません")
            building_info_text = "【ビル情報】利用可能なビル情報がありません。"
    
    # API呼び出しパラメータを準備
    messages = []
    
    # システムプロンプト
    system_msg = {
        "role": "system",
        "content": prompt
    }
    messages.append(system_msg)
    
    # チャット履歴があれば追加
    if chat_history and len(chat_history) > 1:
        safe_history = [
            {"role": m.get("role"), "content": m.get("content")}
            for m in chat_history[:-1]  # 最後の質問は除く
            if isinstance(m, dict) and m.get("role") and m.get("content")
        ]
        messages.extend(safe_history)
    
    # 現在の質問（ビル情報付き）
    if building_info_text:
        question_with_building = f"{building_info_text}\n\n【質問】\n{question}\n\n上記のビル情報を参考に、あなたの知識に基づいて回答してください。"
    else:
        question_with_building = f"【質問】\n{question}\n\nビル情報は利用せず、あなたの知識に基づいて回答してください。"
    
    user_msg = {
        "role": "user",
        "content": question_with_building
    }
    messages.append(user_msg)
    
    # API呼び出しパラメータ
    api_params = {
        "max_tokens": max_tokens or 4096,
        "temperature": temperature or 0.0
    }
    
    # モデルに応じてAPI呼び出し
    try:
        print(f"🤖 ビル情報のみでの回答生成開始 - モデル: {model}")
        
        if model.startswith("gpt"):
            # Azure OpenAI GPT
            azure_client = create_azure_client()
            answer = call_azure_gpt(
                azure_client,
                model,
                messages,
                max_tokens=api_params["max_tokens"],
                temperature=api_params["temperature"]
            )
        else:
            # AWS Bedrock Claude
            bedrock_client = create_bedrock_client()
            model_id = get_claude_model_name(model)
            answer = call_claude_bedrock(
                bedrock_client,
                model_id,
                messages,
                max_tokens=api_params["max_tokens"],
                temperature=api_params["temperature"] if api_params["temperature"] != 0.0 else None
            )
        
        print(f"✅ ビル情報のみでの回答生成完了 - 回答文字数: {len(answer)}")
        
        return {
            "answer": answer,
            "used_equipment": "なし（ビル情報のみ使用）",
            "equipment_info": {},
            "sources": [],
            "selected_files": [],
            "context_length": len(building_info_text),
            "building_info_included": include_building_info,
            "target_building": target_building,
            "images": []
        }
        
    except Exception as e:
        print(f"❌ ビル情報のみでの回答生成エラー: {e}")
        raise

# ---------------------------------------------------------------------------
# 質問から設備を自動推定する関数
# ---------------------------------------------------------------------------

def detect_equipment_from_question(question: str, available_equipment: List[str]) -> Optional[str]:
    """
    質問文から対象設備を推定
    
    Args:
        question: ユーザーの質問文
        available_equipment: 利用可能な設備名のリスト
        
    Returns:
        推定された設備名または None
    """
    # 質問文を正規化
    question_lower = question.lower()
    
    # 設備名が直接含まれているかチェック
    for equipment in available_equipment:
        equipment_keywords = equipment.replace("設備", "").split("・")
        
        for keyword in equipment_keywords:
            if keyword in question_lower:
                print(f"🎯 自動推定: '{keyword}' → {equipment}")
                return equipment
    
    # キーワードマッチング
    equipment_keywords = {
        "自動火災報知設備": ["火災", "感知器", "煙", "熱", "報知", "警報"],
        "非常放送設備": ["放送", "スピーカ", "アナウンス"],
        "誘導灯設備": ["誘導灯", "避難", "誘導"],
        "非常照明設備": ["非常照明", "非常灯", "照明"]
    }
    
    for equipment, keywords in equipment_keywords.items():
        if equipment in available_equipment:
            for keyword in keywords:
                if keyword in question_lower:
                    print(f"🎯 キーワード推定: '{keyword}' → {equipment}")
                    return equipment
    
    print("❓ 設備を自動推定できませんでした")
    return None

# 🔥 質問からビルを自動推定する関数を追加
def detect_building_from_question(question: str) -> Optional[str]:
    """
    質問文から対象ビルを推定
    
    Args:
        question: ユーザーの質問文
        
    Returns:
        推定されたビル名または None
    """
    building_manager = get_building_manager()
    if not building_manager or not building_manager.available:
        return None
    
    # 利用可能なビル一覧を取得
    available_buildings = building_manager.get_building_list()
    
    # 質問文を正規化
    question_lower = question.lower()
    
    # 各ビルについてキーワード検索
    for building_name in available_buildings:
        # ビル名で直接検索
        if building_name.lower() in question_lower:
            print(f"🏢 ビル名推定: '{building_name}'")
            return building_name
        
        # キーワード検索を実行
        matched_buildings = building_manager.search_building_by_keyword(building_name)
        if matched_buildings:
            print(f"🏢 キーワード推定: '{building_name}' → {matched_buildings[0]}")
            return matched_buildings[0]
    
    print("❓ ビルを自動推定できませんでした")
    return None

# ---------------------------------------------------------------------------
# 互換性維持（旧関数）
# ---------------------------------------------------------------------------

def generate_answer(*args, **kwargs):
    """旧関数の互換性維持 - 廃止予定"""
    raise NotImplementedError(
        "generate_answer は廃止されました。generate_answer_with_equipment を使用してください。"
    )