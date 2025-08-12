# src/langchain_chains.py

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.langchain_models import get_chat_model
from src.logging_utils import init_logger
logger = init_logger()

class ChainManager:
    """LangChain用のチェーン管理クラス - シンプル版"""
    
    @staticmethod
    def create_equipment_knowledge(inputs: dict) -> str:
        """設備資料のKnowledge Contents生成"""
        equipment_content = inputs.get("equipment_content", "")
        if not equipment_content:
            return "設備資料情報はありません。"
        return equipment_content
    
    @staticmethod
    def create_building_knowledge(inputs: dict) -> str:
        """ビル情報のKnowledge Contents生成"""
        building_content = inputs.get("building_content", "")
        if not building_content:
            return "ビル情報はありません。"
        return building_content
    
    @staticmethod
    def create_chat_history_messages(chat_history: Optional[List[Dict[str, str]]]) -> List:
        """チャット履歴をLangChainのメッセージ形式に変換"""
        if not chat_history:
            return []
        
        messages = []
        for msg in chat_history:
            if not isinstance(msg, dict) or not msg.get("role") or not msg.get("content"):
                continue
                
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
        
        return messages
    
    @staticmethod
    def create_unified_chain(
        model_name: str,
        system_prompt: str,
        mode: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ):
        """統一されたチェーンテンプレート
        
        プロンプト構造:
        === System Message ===
        （各モード専用プロンプト）
        === Knowledge Contents ===
        （各モードの追加資料情報）
        === Chat History ===
        （ある場合は会話履歴）
        === Human Message ===
        【質問】（ユーザーの質問）
        上記の情報を参考に、日本語で回答してください。
        """
        chat_model = get_chat_model(model_name, temperature, max_tokens)
        
        if mode == "暗黙知法令チャットモード":
            # 設備資料ありの場合
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "=== Knowledge Contents ===\n{knowledge_contents}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【質問】\n{question}\n\n上記の情報を参考に、日本語で回答してください。")
            ])
            
            knowledge_generator = RunnableLambda(ChainManager.create_equipment_knowledge)
            
        elif mode == "ビルマスタ質問モード":
            # ビル情報ありの場合
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "=== Knowledge Contents ===\n{knowledge_contents}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【質問】\n{question}\n\n上記の情報を参考に、日本語で回答してください。")
            ])
            
            knowledge_generator = RunnableLambda(ChainManager.create_building_knowledge)
            
        else:  # 質疑応答書添削モード
            # Knowledge Contentsなしの場合
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【質問】\n{question}\n\n上記の情報を参考に、日本語で回答してください。")
            ])
            
            knowledge_generator = None
        
        # チェーン構築
        if knowledge_generator:
            chain = (
                {
                    "question": lambda x: x["question"],
                    "knowledge_contents": knowledge_generator,
                    "chat_history": lambda x: ChainManager.create_chat_history_messages(x.get("chat_history"))
                }
                | prompt
                | chat_model
                | StrOutputParser()
            )
        else:
            chain = (
                {
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: ChainManager.create_chat_history_messages(x.get("chat_history"))
                }
                | prompt
                | chat_model
                | StrOutputParser()
            )
        
        logger.info(f"✅ Unified Chain 作成完了: model={model_name}, mode={mode}")
        return chain

# === 統一インターフェース ===

def generate_unified_answer(
    *,
    prompt: str,
    question: str,
    model: str = "claude-4-sonnet",
    mode: str = "暗黙知法令チャットモード",
    equipment_content: Optional[str] = None,
    building_content: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    統一された回答生成関数
    
    Args:
        prompt: システムプロンプト
        question: ユーザーの質問
        model: 使用するモデル名
        mode: モード名
        equipment_content: 設備資料内容（暗黙知法令チャットモードで使用）
        building_content: ビル情報内容（ビルマスタ質問モードで使用）
        chat_history: チャット履歴
        temperature: 温度パラメータ
        max_tokens: 最大トークン数
        
    Returns:
        回答結果辞書
    """
    
    logger.info(f"🚀 統一回答生成開始: model={model}, mode={mode}")
    
    # 統一チェーンを作成
    chain = ChainManager.create_unified_chain(model, prompt, mode, temperature, max_tokens)
    
    # 入力データ準備
    chain_input = {
        "question": question,
        "chat_history": chat_history[:-1] if chat_history and len(chat_history) > 1 else None
    }
    
    # モード別のコンテンツを追加
    if mode == "暗黙知法令チャットモード" and equipment_content:
        chain_input["equipment_content"] = equipment_content
    elif mode == "ビルマスタ質問モード" and building_content:
        chain_input["building_content"] = building_content
    
    # チェーン実行
    try:
        answer = chain.invoke(chain_input)
        
        # 結果構築
        result = {
            "answer": answer,
            "mode": mode,
            "langchain_used": True
        }
        
        logger.info(f"✅ 統一回答生成完了: mode={mode}, 回答文字数={len(answer)}")
        return result
        
    except Exception as e:
        logger.error(f"❌ 統一回答生成エラー: {e}", exc_info=True)
        raise

# === 後方互換性のための関数 ===

def generate_smart_answer_with_langchain(
    *,
    prompt: str,
    question: str,
    model: str = "claude-4-sonnet",
    equipment_data: Optional[Dict[str, Dict[str, Any]]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    既存のapp.pyから呼び出される関数（後方互換性のため）
    
    注意: この関数は既存のロジックを維持しつつ、
    実際の設備選択やビル情報取得はapp.py側で行われる前提です。
    """
    
    # この関数は既存のapp.pyとの互換性を保つため、
    # 実際の処理はapp.py側で行うことを想定
    # ここでは基本的な回答生成のみ実行
    
    return generate_unified_answer(
        prompt=prompt,
        question=question,
        model=model,
        mode="暗黙知法令チャットモード",  # デフォルト
        chat_history=chat_history,
        temperature=temperature,
        max_tokens=max_tokens
    )

# テスト用関数
def test_chain_creation():
    """チェーン作成のテスト"""
    try:
        logger.info("🧪 統一チェーンテスト開始...")
        
        # 各モードのテスト
        modes = ["暗黙知法令チャットモード", "質疑応答書添削モード", "ビルマスタ質問モード"]
        
        for mode in modes:
            chain = ChainManager.create_unified_chain(
                "claude-4-sonnet",
                f"あなたは{mode}の専門家です。",
                mode,
                temperature=0.0
            )
            logger.info(f"✅ {mode} Chain 作成成功")
        
        logger.info("🧪 統一回答生成テスト...")
        result = generate_unified_answer(
            prompt="テスト用プロンプト",
            question="テスト質問",
            mode="質疑応答書添削モード"
        )
        logger.info("✅ 統一回答生成テスト成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ テスト失敗: {e}")
        return False

if __name__ == "__main__":
    test_chain_creation()