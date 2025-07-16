# src/rag_qa.py - AWS Bedrock Claude版

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

# Claude用のモデル名マッピング
CLAUDE_MODEL_MAPPING = {
    "claude-4-sonnet": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-3.7": "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"
}

def get_claude_model_name(model_name: str) -> str:
    """Claude表示名をBedrockモデルIDに変換"""
    return CLAUDE_MODEL_MAPPING.get(model_name, model_name)

def call_claude_bedrock(client, model_id: str, messages: List[Dict], temperature: float = None):
    """AWS Bedrock Converse API経由でClaudeを呼び出し（max_tokensはモデル上限）"""
    
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
    
    # Converse API用のパラメータを構築（max_tokensは指定しない＝モデル上限）
    converse_params = {
        "modelId": model_id,
        "messages": conversation_messages,
        "inferenceConfig": {}
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
    ) -> Dict[str, Any]:
    """
    指定された設備の選択ファイルをプロンプトに投入してClaudeで回答を生成
    
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
    
    Returns:
        回答結果辞書
    """
    # AWS Bedrock クライアントを作成
    client = create_bedrock_client()

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
    
    # --- 2) プロンプト組み立て ---
    equipment_context = f"""
【参考資料】設備: {target_equipment} (カテゴリ: {equipment_info['equipment_category']})
使用ファイル: {', '.join(sources)}
使用ファイル数: {len(sources)}/{len(all_sources)}

【資料内容】
{combined_text}
"""
    
    system_msg = {
        "role": "system",
        "content": prompt
    }
    
    user_msg = {
        "role": "user", 
        "content": f"{equipment_context}\n\n【質問】\n{question}\n\n上記の資料を参考に、日本語で回答してください。"
    }
    
    # --- 3) Messages 組み立て ---
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
    
    # --- 5) AWS Bedrock 呼び出し ---
    try:
        model_id = get_claude_model_name(model)
        print(f"🤖 API呼び出し開始 - モデル: {model_id}")
        answer = call_claude_bedrock(
            client,
            model_id, 
            messages,
            temperature=temperature if temperature != 0.0 else None
        )
        print(f"✅ 回答生成完了 - 回答文字数: {len(answer)}")
        
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")
        raise
    
    return {
        "answer": answer,
        "used_equipment": target_equipment,
        "equipment_info": equipment_info,
        "sources": sources,
        "selected_files": selected_files,  # 🔥 選択ファイル情報を追加
        "context_length": len(combined_text),
        "images": []  # 現バージョンでは画像は対応しない
    }

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

# ---------------------------------------------------------------------------
# 互換性維持（旧関数）
# ---------------------------------------------------------------------------

def generate_answer(*args, **kwargs):
    """旧関数の互換性維持 - 廃止予定"""
    raise NotImplementedError(
        "generate_answer は廃止されました。generate_answer_with_equipment を使用してください。"
    )