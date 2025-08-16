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
    def create_combined_knowledge(inputs: dict) -> str:
        """設備資料とビル情報を組み合わせたKnowledge Contents生成"""
        equipment_content = inputs.get("equipment_content", "")
        building_content = inputs.get("building_content", "")
        
        knowledge_parts = []
        
        if equipment_content:
            knowledge_parts.append(f"=== 設備資料情報 ===\n{equipment_content}")
        
        if building_content:
            knowledge_parts.append(f"=== ビル情報 ===\n{building_content}")
        
        if not knowledge_parts:
            return "関連資料情報はありません。一般知識に基づいて回答してください。"
        
        return "\n\n".join(knowledge_parts)
    
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
        """統一されたチェーンテンプレート - モード別プロンプト構成対応"""
        chat_model = get_chat_model(model_name, temperature, max_tokens)
        
        if mode == "質疑応答書添削モード":
            # Knowledge Contentsなしの場合
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【添削依頼】\n{question}\n\n上記の内容について、質疑応答書として適切な形式で添削・改善提案をお願いします。")
            ])
            knowledge_generator = None
            
        elif mode == "暗黙知法令チャットモード":
            # 暗黙知法令チャットモード専用構成
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "=== 設備資料情報 ===\n{equipment_content}\n\n=== ビル情報 ===\n{building_content}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【技術的質問】\n{question}\n\n上記の設備資料とビル情報を参考に、建築電気設備設計の観点から詳細に回答してください。")
            ])
            knowledge_generator = RunnableLambda(ChainManager.create_separate_knowledge)
            
        elif mode == "ビルマスタ質問モード":
            # ビルマスタ質問モード専用構成
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "=== ビルマスター情報 ===\n{building_content}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【ビル情報に関する質問】\n{question}\n\nビルマスターデータに記載されている情報のみを使用して、正確に回答してください。")
            ])
            knowledge_generator = RunnableLambda(ChainManager.create_building_knowledge)
            
        else:
            # デフォルト（既存の統一構成を維持）
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{knowledge_contents}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【質問】\n{question}\n\n上記の資料情報を参考に、日本語で回答してください。")
            ])
            knowledge_generator = RunnableLambda(ChainManager.create_combined_knowledge)
        
        # 🔥 修正: チェーン構築を統一（新しいフィールド対応）
        if knowledge_generator:
            chain = (
                {
                    "question": lambda x: x["question"],
                    "equipment_content": lambda x: x.get("equipment_content", ""),
                    "building_content": lambda x: x.get("building_content", ""),
                    "target_building_content": lambda x: x.get("target_building_content", ""),  # 🔥 新規追加
                    "other_buildings_content": lambda x: x.get("other_buildings_content", ""),   # 🔥 新規追加
                    "knowledge_contents": knowledge_generator,  # 従来モード用
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

    @staticmethod
    def create_building_knowledge(inputs: dict) -> dict:
        """ビルマスタ質問モード用：新しいプロンプト構造対応"""
        result = inputs.copy()
        
        # 新しいフィールドを取得
        target_building_content = inputs.get("target_building_content", "")
        other_buildings_content = inputs.get("other_buildings_content", "")
        building_content = inputs.get("building_content", "")  # 後方互換性
        
        # 新しいプロンプト構造を構築
        if target_building_content and other_buildings_content:
            # 特定ビル + 他のビル（新機能）
            formatted_content = f"==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\n{other_buildings_content}"
            
        elif target_building_content and not other_buildings_content:
            # 特定ビルのみ
            formatted_content = f"==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\nその他のビル情報はありません。"
            
        elif other_buildings_content and not target_building_content:
            # 全ビル情報（対象ビルとその他の区別なし）
            formatted_content = f"==現在の対象ビル==\n対象ビルは指定されていません。\n\n==その他のビル==\n{other_buildings_content}"
            
        elif building_content:
            # 後方互換性：従来のbuilding_contentを使用
            formatted_content = f"==現在の対象ビル==\n対象ビルは指定されていません。\n\n==その他のビル==\n{building_content}"
            
        else:
            # ビル情報なし
            formatted_content = "==現在の対象ビル==\n対象ビルは指定されていません。\n\n==その他のビル==\nビル情報はありません。"
        
        result["building_content"] = formatted_content
        return result

    @staticmethod
    def create_separate_knowledge(inputs: dict) -> dict:
        """暗黙知法令チャットモード用：設備とビルを分離表示"""
        result = inputs.copy()
        
        equipment_content = inputs.get("equipment_content", "")
        target_building_content = inputs.get("target_building_content", "")
        other_buildings_content = inputs.get("other_buildings_content", "")
        building_content = inputs.get("building_content", "")  # 後方互換性
        
        # 設備情報
        result["equipment_content"] = equipment_content if equipment_content else "設備資料情報はありません。"
        
        # ビル情報（ビルマスタモードと同じ構造を使用）
        if target_building_content and other_buildings_content:
            # 特定ビル + 他のビル
            formatted_building = f"==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\n{other_buildings_content}"
            
        elif target_building_content and not other_buildings_content:
            # 特定ビルのみ
            formatted_building = f"==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\nその他のビル情報はありません。"
            
        elif building_content:
            # 後方互換性：従来の構造
            formatted_building = building_content
            
        else:
            # ビル情報なし
            formatted_building = "ビル情報はありません。"
        
        result["building_content"] = formatted_building
        return result

    # 🔥 統一的なcomplete_prompt構築のための新しい関数
    @staticmethod
    def create_building_prompt_content(inputs: dict) -> str:
        """complete_prompt構築用：ビル情報のフォーマット"""
        target_building_content = inputs.get("target_building_content", "")
        other_buildings_content = inputs.get("other_buildings_content", "")
        building_content = inputs.get("building_content", "")
        
        if target_building_content and other_buildings_content:
            return f"=== ビルマスター情報 ===\n==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\n{other_buildings_content}"
            
        elif target_building_content and not other_buildings_content:
            return f"=== ビルマスター情報 ===\n==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\nその他のビル情報はありません。"
            
        elif other_buildings_content and not target_building_content:
            return f"=== ビルマスター情報 ===\n==現在の対象ビル==\n対象ビルは指定されていません。\n\n==その他のビル==\n{other_buildings_content}"
            
        elif building_content:
            return f"=== ビルマスター情報 ===\n{building_content}"
            
        else:
            return "=== ビルマスター情報 ===\nビル情報はありません。"

# === 統一インターフェース ===

def generate_unified_answer(
    *,
    prompt: str,
    question: str,
    model: str = "claude-4-sonnet",
    mode: str = "暗黙知法令チャットモード",
    equipment_content: Optional[str] = None,
    building_content: Optional[str] = None,
    target_building_content: Optional[str] = None,  # 🔥 新規追加
    other_buildings_content: Optional[str] = None,   # 🔥 新規追加
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    統一された回答生成関数
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
    if mode != "質疑応答書添削モード":
        chain_input["equipment_content"] = equipment_content or ""
        chain_input["building_content"] = building_content or ""
        chain_input["target_building_content"] = target_building_content or ""  # 🔥 新規追加
        chain_input["other_buildings_content"] = other_buildings_content or ""   # 🔥 新規追加
    
    # チェーン実行
    try:
        answer = chain.invoke(chain_input)
        
        # 🔥 修正: モード別のcomplete_prompt構築（新しい構造対応）
        full_prompt_parts = []
        
        # システムプロンプト
        full_prompt_parts.append(f"=== System Message ===\n{prompt}")
        
        # モード別のKnowledge Contents構築
        if mode == "暗黙知法令チャットモード":
            # 設備とビル情報を分離表示
            equipment_content = chain_input.get("equipment_content", "")
            if equipment_content:
                full_prompt_parts.append(f"=== 設備資料情報 ===\n{equipment_content}")
            else:
                full_prompt_parts.append("=== 設備資料情報 ===\n設備資料情報はありません。")
            
            # ビル情報（新しい構造対応）
            building_prompt = ChainManager.create_building_prompt_content(chain_input)
            full_prompt_parts.append(building_prompt)
                
        elif mode == "ビルマスタ質問モード":
            # ビル情報のみ（新しい構造）
            building_prompt = ChainManager.create_building_prompt_content(chain_input)
            full_prompt_parts.append(building_prompt)
                
        elif mode != "質疑応答書添削モード":
            # その他のモード（従来の統一構造）
            equipment_content = chain_input.get("equipment_content", "")
            building_content = chain_input.get("building_content", "")
            
            knowledge_parts = []
            if equipment_content:
                knowledge_parts.append(f"=== 設備資料情報 ===\n{equipment_content}")
            if building_content:
                knowledge_parts.append(f"=== ビル情報 ===\n{building_content}")
            
            if knowledge_parts:
                full_prompt_parts.append(f"=== Knowledge Contents ===\n" + "\n\n".join(knowledge_parts))
            else:
                full_prompt_parts.append("=== Knowledge Contents ===\n関連資料情報はありません。")
        
        # チャット履歴
        original_chat_history = chat_history[:-1] if chat_history and len(chat_history) > 1 else None
        if original_chat_history:
            full_prompt_parts.append("=== Chat History ===")
            for msg in original_chat_history:
                if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                    role = msg["role"].capitalize()
                    full_prompt_parts.append(f"{role}: {msg['content']}")
        
        # 現在の質問（モード別の接頭辞付き）
        if mode == "暗黙知法令チャットモード":
            full_prompt_parts.append(f"=== Human Message ===\n【技術的質問】\n{question}\n\n上記の設備資料とビル情報を参考に、建築電気設備設計の観点から詳細に回答してください。")
        elif mode == "質疑応答書添削モード":
            full_prompt_parts.append(f"=== Human Message ===\n【添削依頼】\n{question}\n\n上記の内容について、質疑応答書として適切な形式で添削・改善提案をお願いします。")
        elif mode == "ビルマスタ質問モード":
            full_prompt_parts.append(f"=== Human Message ===\n【ビル情報に関する質問】\n{question}\n\nビルマスターデータに記載されている情報のみを使用して、正確に回答してください。")
        else:
            full_prompt_parts.append(f"=== Human Message ===\n【質問】\n{question}\n\n上記の資料情報を参考に、日本語で回答してください。")
        
        # 完全なプロンプトを結合
        complete_prompt = "\n\n".join(full_prompt_parts)
        
        # 結果構築
        result = {
            "answer": answer,
            "mode": mode,
            "langchain_used": True,
            "complete_prompt": complete_prompt  # 🔥 新しい構造に対応
        }
        
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
    mode: str = "暗黙知法令チャットモード",
    equipment_content: Optional[str] = None,
    building_content: Optional[str] = None,
    target_building_content: Optional[str] = None,  # 🔥 新規追加
    other_buildings_content: Optional[str] = None,   # 🔥 新規追加
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    既存のapp.pyから呼び出される関数（後方互換性のため）
    """
    
    return generate_unified_answer(
        prompt=prompt,
        question=question,
        model=model,
        mode=mode,
        equipment_content=equipment_content,
        building_content=building_content,
        target_building_content=target_building_content,  # 🔥 新規追加
        other_buildings_content=other_buildings_content,   # 🔥 新規追加
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