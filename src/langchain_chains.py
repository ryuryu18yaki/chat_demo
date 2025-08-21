# src/langchain_chains.py (最小限の変更を加えた最終版)

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
# ▼ 変更点：JSONパーサーをインポートします
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.langchain_models import get_chat_model
from src.logging_utils import init_logger
logger = init_logger()

class ChainManager:
    """LangChain用のチェーン管理クラス - シンプル版"""
    # =================================================================
    # ▼ 変更点
    # このクラス内の既存のメソッドは一切変更しません。
    # =================================================================
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
                    "target_building_content": lambda x: x.get("target_building_content", ""),
                    "other_buildings_content": lambda x: x.get("other_buildings_content", ""),
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

    @staticmethod
    def create_building_knowledge(inputs: dict) -> dict:
        """ビルマスタ質問モード用：新しいプロンプト構造対応"""
        result = inputs.copy()
        
        target_building_content = inputs.get("target_building_content", "")
        other_buildings_content = inputs.get("other_buildings_content", "")
        building_content = inputs.get("building_content", "")
        
        if target_building_content and other_buildings_content:
            formatted_content = f"==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\n{other_buildings_content}"
        elif target_building_content and not other_buildings_content:
            formatted_content = f"==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\nその他のビル情報はありません。"
        elif other_buildings_content and not target_building_content:
            formatted_content = f"==現在の対象ビル==\n対象ビルは指定されていません。\n\n==その他のビル==\n{other_buildings_content}"
        elif building_content:
            formatted_content = f"==現在の対象ビル==\n対象ビルは指定されていません。\n\n==その他のビル==\n{building_content}"
        else:
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
        building_content = inputs.get("building_content", "")
        
        result["equipment_content"] = equipment_content if equipment_content else "設備資料情報はありません。"
        
        if target_building_content and other_buildings_content:
            formatted_building = f"==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\n{other_buildings_content}"
        elif target_building_content and not other_buildings_content:
            formatted_building = f"==現在の対象ビル==\n{target_building_content}\n\n==その他のビル==\nその他のビル情報はありません。"
        elif building_content:
            formatted_building = building_content
        else:
            formatted_building = "ビル情報はありません。"
        
        result["building_content"] = formatted_building
        return result

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

# =================================================================
# ▼ 変更点
# generate_unified_answer と generate_smart_answer_with_langchain を書き換え、
# タイトル生成機能を追加します。
# =================================================================

def get_actual_prompt_from_template(
    prompt_template: ChatPromptTemplate,
    inputs: dict,
    mode: str
) -> str:
    """プロンプトテンプレートから実際の送信内容を取得"""
    try:
        # knowledge_generator処理が必要な場合
        if mode == "暗黙知法令チャットモード":
            processed_inputs = ChainManager.create_separate_knowledge(inputs)
        elif mode == "ビルマスタ質問モード":
            processed_inputs = ChainManager.create_building_knowledge(inputs)
        elif mode != "質疑応答書添削モード":
            processed_inputs = inputs.copy()
            processed_inputs["knowledge_contents"] = ChainManager.create_combined_knowledge(inputs)
        else:
            processed_inputs = inputs
        
        # チャット履歴をメッセージ形式に変換
        processed_inputs["chat_history"] = ChainManager.create_chat_history_messages(
            inputs.get("chat_history")
        )
        
        # プロンプトテンプレートを適用
        messages = prompt_template.format_messages(**processed_inputs)
        
        # メッセージを文字列に変換
        prompt_parts = []
        for msg in messages:
            role = getattr(msg, 'type', 'unknown').upper()
            content = getattr(msg, 'content', str(msg))
            prompt_parts.append(f"=== {role} ===\n{content}")
        
        complete_prompt = "\n\n" + ("="*50 + "\n\n").join(prompt_parts)
        
        logger.info(f"🔥 Generated actual prompt: {len(complete_prompt)} characters")
        return complete_prompt
        
    except Exception as e:
        logger.error(f"❌ Prompt generation failed: {e}")
        return f"=== ERROR ===\nプロンプト生成に失敗: {str(e)}"

def generate_unified_answer(
    *,
    prompt: str,
    question: str,
    model: str = "claude-4-sonnet",
    mode: str = "暗黙知法令チャットモード",
    equipment_content: Optional[str] = None,
    building_content: Optional[str] = None,
    target_building_content: Optional[str] = None,
    other_buildings_content: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    generate_title: bool = False # ★タイトル生成フラグを追加
) -> Dict[str, Any]:
    """
    統一された回答生成関数。generate_titleフラグに応じて動作を切り替える。
    """
    logger.info(f"🚀 統一回答生成開始: model={model}, mode={mode}, generate_title={generate_title}")
    
    # 既存のチェーン作成ロジックを呼び出す
    # ★ generate_title が True の場合、元のプロンプトにJSON指示を追加する
    final_prompt = prompt
    output_parser = StrOutputParser()
    if generate_title:
        json_instruction = """
【重要：出力形式】
あなたの回答と、この会話のタイトルを考え、必ず以下のJSON形式で出力してください。他のテキストは一切含めないでください。
{{
  "answer": "ここにユーザーへの回答本文を入れてください。",
  "title": "ここに30文字程度の会話のタイトルを入れてください。"
}}"""
        final_prompt = prompt + "\n\n" + json_instruction
        output_parser = JsonOutputParser()

    # ★ 既存の create_unified_chain を呼び出すが、末尾のパーサーだけを差し替える
    # この方法では create_unified_chain の中身を書き換える必要があり、元のコードの変更が大きくなるため、
    # ここでチェーンのロジックを再定義するのが最も安全です。元のロジックは完全にコピーします。
    
    chat_model = get_chat_model(model, temperature, max_tokens)
    
    # 元の create_unified_chain の中身をここに展開
    if mode == "質疑応答書添削モード":
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", final_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "【添削依頼】\n{question}\n\n上記の内容について、質疑応答書として適切な形式で添削・改善提案をお願いします。")
        ])
        chain = (
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: ChainManager.create_chat_history_messages(x.get("chat_history"))
            }
            | prompt_template
            | chat_model
            | output_parser
        )
    else: # 暗黙知法令、ビルマスタ、その他デフォルトモード
        # どのモードでも knowledge_generator を使う想定で汎用化
        if mode == "暗黙知法令チャットモード":
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", final_prompt),
                ("human", "=== 設備資料情報 ===\n{equipment_content}\n\n=== ビル情報 ===\n{building_content}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【技術的質問】\n{question}\n\n上記の設備資料とビル情報を参考に、建築電気設備設計の観点から詳細に回答してください。")
            ])
            knowledge_generator = RunnableLambda(ChainManager.create_separate_knowledge)
        elif mode == "ビルマスタ質問モード":
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", final_prompt),
                ("human", "=== ビルマスター情報 ===\n{building_content}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【ビル情報に関する質問】\n{question}\n\nビルマスターデータに記載されている情報のみを使用して、正確に回答してください。")
            ])
            knowledge_generator = RunnableLambda(ChainManager.create_building_knowledge)
        else: # デフォルト
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", final_prompt),
                ("human", "{knowledge_contents}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "【質問】\n{question}\n\n上記の資料情報を参考に、日本語で回答してください。")
            ])
            knowledge_generator = RunnableLambda(ChainManager.create_combined_knowledge)
        
        chain = (
            {
                "question": lambda x: x["question"],
                "equipment_content": lambda x: x.get("equipment_content", ""),
                "building_content": lambda x: x.get("building_content", ""),
                "target_building_content": lambda x: x.get("target_building_content", ""),
                "other_buildings_content": lambda x: x.get("other_buildings_content", ""),
                "knowledge_contents": knowledge_generator,
                "chat_history": lambda x: ChainManager.create_chat_history_messages(x.get("chat_history"))
            }
            | prompt_template
            | chat_model
            | output_parser
        )

    # 入力データ準備
    chain_input = {
        "question": question,
        "chat_history": chat_history[:-1] if chat_history and len(chat_history) > 1 else None,
        "equipment_content": equipment_content or "",
        "building_content": building_content or "",
        "target_building_content": target_building_content or "",
        "other_buildings_content": other_buildings_content or ""
    }

    try:
        actual_complete_prompt = get_actual_prompt_from_template(
            prompt_template, chain_input, mode
        )
    except Exception as e:
        logger.error(f"❌ Prompt extraction failed: {e}")
        actual_complete_prompt = f"=== SYSTEM ===\n{final_prompt}\n\n=== HUMAN ===\n{question}"
    
    # チェーン実行と結果の整形
    try:
        response = chain.invoke(chain_input)
        
        # complete_prompt の構築ロジックは元のコードから省略（必要なら後で復活可能）
        
        if generate_title:
            return {
                "answer": response.get("answer", "応答の取得に失敗しました。"),
                "title": response.get("title"),
                "langchain_used": True,
                "complete_prompt": actual_complete_prompt
            }
        else:
            return {
                "answer": str(response),
                "title": None,
                "langchain_used": True,
                "complete_prompt": actual_complete_prompt
            }
        
    except Exception as e:
        logger.error(f"❌ 統一回答生成エラー: {e}", exc_info=True)
        # 既存のコードに合わせてエラーを再発生させる
        raise

def generate_smart_answer_with_langchain(
    *,
    prompt: str,
    question: str,
    model: str = "claude-4-sonnet",
    mode: str = "暗黙知法令チャットモード",
    equipment_content: Optional[str] = None,
    building_content: Optional[str] = None,
    target_building_content: Optional[str] = None,
    other_buildings_content: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    generate_title: bool = False # ★app.pyから渡されるフラグ
) -> Dict[str, Any]:
    """
    既存のapp.pyから呼び出される関数（後方互換性のため）
    ★ generate_title フラグを下の関数に渡す役割を追加
    """
    # 既存のコードでは generate_unified_answer を呼び出しているので、その構造を維持
    # generate_title フラグを渡すように変更
    response_dict = generate_unified_answer(
        prompt=prompt,
        question=question,
        model=model,
        mode=mode,
        equipment_content=equipment_content,
        building_content=building_content,
        target_building_content=target_building_content,
        other_buildings_content=other_buildings_content,
        chat_history=chat_history,
        temperature=temperature,
        max_tokens=max_tokens,
        generate_title=generate_title # ★フラグを渡す
    )
    
    # 既存のコードは generate_unified_answer の戻り値をそのまま返していたので、
    # その構造を模倣するが、新しいキー 'title' を含める
    return response_dict

# =================================================================
# ▼ 変更点
# この関数は不要になるため、完全に削除します。
# =================================================================
# def generate_chat_title_with_llm(...):

# =================================================================
# ▼ 変更点
# このテスト関数は古い構成に基づいているため、一旦コメントアウトするか削除します。
# =================================================================
# def test_chain_creation():
# if __name__ == "__main__":
#     test_chain_creation()