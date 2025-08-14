# src/langchain_models.py

import os
from typing import Union, Optional
import boto3
from langchain_aws import ChatBedrock  # ← AWS Bedrock用
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from src.logging_utils import init_logger
logger = init_logger()

# Claude用のモデル名マッピング（Bedrock用）
CLAUDE_MODEL_MAPPING = {
    "claude-4-sonnet": "anthropic.claude-sonnet-4-20250514-v1:0",  
    "claude-3.7": "anthropic.claude-3-7-sonnet-20250219-v1:0",     
}

# Azure OpenAI用のモデル名マッピング
AZURE_MODEL_MAPPING = {
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o"
}

class ModelManager:
    """LangChain用のモデル管理クラス"""
    
    @staticmethod
    def get_credentials() -> dict:
        """認証情報を取得"""
        credentials = {}
        
        if STREAMLIT_AVAILABLE:
            # === AWS Bedrock設定 ===
            try:
                credentials["aws_access_key_id"] = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
                credentials["aws_secret_access_key"] = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
                credentials["aws_region"] = st.secrets.get("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
            except:
                credentials["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
                credentials["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
                credentials["aws_region"] = os.getenv("AWS_REGION", "us-east-1")
            
            # === Azure OpenAI設定 ===
            try:
                credentials["azure_endpoint"] = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
                credentials["azure_api_key"] = st.secrets.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))
                credentials["azure_api_version"] = st.secrets.get("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
            except:
                credentials["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
                credentials["azure_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
                credentials["azure_api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        else:
            # Streamlit環境外では環境変数から取得
            credentials["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
            credentials["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
            credentials["aws_region"] = os.getenv("AWS_REGION", "us-east-1")
            credentials["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
            credentials["azure_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
            credentials["azure_api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        return credentials
    
    @staticmethod
    def create_claude_model(
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> ChatBedrock:
        """Claude (AWS Bedrock経由) モデルを作成"""
        credentials = ModelManager.get_credentials()
        
        if not credentials["aws_access_key_id"] or not credentials["aws_secret_access_key"]:
            raise ValueError("AWS Bedrock の設定が不足しています。Streamlit SecretsのAWS認証情報を確認してください。")
        
        # Boto3セッションを作成（認証情報を明示的に設定）
        session = boto3.Session(
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"],
            region_name=credentials["aws_region"]
        )
        
        # Bedrock Runtimeクライアントを作成
        bedrock_client = session.client('bedrock-runtime')
        
        # モデル名をBedrock用に変換
        bedrock_model_id = CLAUDE_MODEL_MAPPING.get(model_name, "anthropic.claude-sonnet-4-20250514-v1:0")
        
        # ChatBedrockのパラメータ
        model_kwargs = {
            "model_id": bedrock_model_id,
            "client": bedrock_client,
            "model_kwargs": {
                "temperature": temperature,
            }
        }
        
        if max_tokens is not None:
            model_kwargs["model_kwargs"]["max_tokens"] = max_tokens
        
        logger.info(f"🤖 Claude Bedrock model作成: {bedrock_model_id}, temp={temperature}, max_tokens={max_tokens}, region={credentials['aws_region']}")
        
        return ChatBedrock(**model_kwargs)
    
    @staticmethod
    def create_azure_gpt_model(
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> AzureChatOpenAI:
        """Azure OpenAI GPT モデルを作成"""
        credentials = ModelManager.get_credentials()
        
        if not credentials["azure_endpoint"] or not credentials["azure_api_key"]:
            raise ValueError("Azure OpenAI の設定が不足しています。Streamlit SecretsまたはSecrets.tomlを確認してください。")
        
        # モデル名をAzureデプロイメント名に変換
        azure_deployment = AZURE_MODEL_MAPPING.get(model_name, model_name)
        
        model_kwargs = {
            "azure_deployment": azure_deployment,
            "temperature": temperature,
            "azure_endpoint": credentials["azure_endpoint"],
            "api_key": credentials["azure_api_key"],
            "api_version": credentials["azure_api_version"]
        }
        
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        
        logger.info(f"🤖 Azure GPT LangChain model作成: {azure_deployment}, temp={temperature}, max_tokens={max_tokens}")
        
        return AzureChatOpenAI(**model_kwargs)
    
    @staticmethod
    def get_chat_model(
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> BaseChatModel:
        """
        モデル名に基づいて適切なChatModelを返す
        
        Args:
            model_name: モデル名 (claude-4-sonnet, gpt-4o等)
            temperature: 温度パラメータ
            max_tokens: 最大トークン数
            
        Returns:
            LangChainのChatModel
        """
        logger.info(f"🎯 LangChain ChatModel作成開始: model={model_name}")
        
        if model_name.startswith("claude"):
            return ModelManager.create_claude_model(model_name, temperature, max_tokens)
        elif model_name.startswith("gpt"):
            return ModelManager.create_azure_gpt_model(model_name, temperature, max_tokens)
        else:
            raise ValueError(f"サポートされていないモデル: {model_name}")

# 便利関数
def get_chat_model(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> BaseChatModel:
    """ModelManager.get_chat_modelの便利関数"""
    return ModelManager.get_chat_model(model_name, temperature, max_tokens)

# 互換性テスト用関数
def test_model_creation():
    """モデル作成のテスト"""
    try:
        logger.info("🧪 Claude Bedrock model test...")
        claude_model = get_chat_model("claude-4-sonnet", temperature=0.1, max_tokens=100)
        logger.info("✅ Claude Bedrock model 作成成功")
        
        logger.info("🧪 GPT model test...")
        gpt_model = get_chat_model("gpt-4o", temperature=0.1, max_tokens=100)
        logger.info("✅ GPT model 作成成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model作成テスト失敗: {e}")
        return False

if __name__ == "__main__":
    test_model_creation()