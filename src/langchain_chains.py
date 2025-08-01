# src/langchain_chains.py

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from src.langchain_models import get_chat_model
from src.logging_utils import init_logger
logger = init_logger()

class ChainManager:
    """LangChain用のチェーン管理クラス"""
    
    @staticmethod
    def create_equipment_context(inputs: dict) -> str:
        """設備コンテキストを生成する関数"""
        question = inputs["question"]
        target_equipment = inputs.get("target_equipment")
        selected_files = inputs.get("selected_files")
        equipment_data = inputs.get("equipment_data")
        
        if not target_equipment or not equipment_data:
            return "設備資料なしでの一般的な回答を生成します。"
        
        if target_equipment not in equipment_data:
            available_equipment = list(equipment_data.keys())
            return f"設備 '{target_equipment}' が見つかりません。利用可能な設備: {', '.join(available_equipment)}"
        
        equipment_info = equipment_data[target_equipment]
        available_files = equipment_info["files"]
        all_sources = equipment_info["sources"]
        
        # 選択されたファイルのみを結合
        if selected_files is not None:
            logger.info(f"🔧 使用設備: {target_equipment}")
            logger.info(f"📄 選択ファイル: {', '.join(selected_files)}")
            
            selected_texts = []
            actual_sources = []
            
            for file_name in selected_files:
                if file_name in available_files:
                    selected_texts.append(available_files[file_name])
                    actual_sources.append(file_name)
                else:
                    logger.warning(f"⚠️ ファイルが見つかりません: {file_name}")
            
            if not selected_texts:
                return f"選択されたファイル（{', '.join(selected_files)}）が設備データに見つかりません。"
            
            combined_text = "\n\n".join(selected_texts)
            sources = actual_sources
            
        else:
            # 全ファイル使用
            selected_texts = list(available_files.values())
            combined_text = "\n\n".join(selected_texts)
            sources = all_sources
        
        logger.info(f"📝 結合後文字数: {len(combined_text)}")
        
        # 設備コンテキストを構築
        equipment_context = f"""
            【参考資料】設備: {target_equipment} (カテゴリ: {equipment_info['equipment_category']})
            使用ファイル: {', '.join(sources)}
            使用ファイル数: {len(sources)}/{len(all_sources)}
            【注意事項】
            **暗黙知メモに関して、ページ番号などの情報は出力を禁止します。**

            【資料内容】
            {combined_text}
            """
        
        return equipment_context
    
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
    def create_simple_qa_chain(
        model_name: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ):
        """シンプルなQ&Aチェーンを作成（設備資料なし）"""
        
        # ChatModelを作成
        chat_model = get_chat_model(model_name, temperature, max_tokens)
        
        # プロンプトテンプレートを作成
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}")
        ])
        
        # チェーンを構築
        chain = (
            {
                "question": RunnablePassthrough(),
                "chat_history": lambda x: ChainManager.create_chat_history_messages(x.get("chat_history"))
            }
            | prompt
            | chat_model
            | StrOutputParser()
        )
        
        logger.info(f"✅ Simple QA Chain 作成完了: model={model_name}")
        return chain
    
    @staticmethod
    def create_equipment_qa_chain(
        model_name: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ):
        """設備資料付きQ&Aチェーンを作成"""
        
        # ChatModelを作成
        chat_model = get_chat_model(model_name, temperature, max_tokens)
        
        # プロンプトテンプレートを作成
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{equipment_context}\n\n【質問】\n{question}\n\n上記の資料を参考に、日本語で回答してください。")
        ])
        
        # 設備コンテキスト生成関数をRunnableLambdaでラップ
        context_generator = RunnableLambda(ChainManager.create_equipment_context)
        
        # チェーンを構築
        chain = (
            {
                "question": lambda x: x["question"],
                "equipment_context": context_generator,
                "chat_history": lambda x: ChainManager.create_chat_history_messages(x.get("chat_history"))
            }
            | prompt
            | chat_model
            | StrOutputParser()
        )
        
        logger.info(f"✅ Equipment QA Chain 作成完了: model={model_name}")
        return chain

class SmartAnswerGenerator:
    """
    スマートな回答生成クラス
    設備の自動推定、ファイル選択、資料なし処理を全て統合
    """
    
    def __init__(self, equipment_data: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Args:
            equipment_data: 設備データ（preprocess_filesの出力）
        """
        self.equipment_data = equipment_data
    
    def determine_target_equipment(
        self, 
        question: str,
        selection_mode: str = "manual",
        manual_equipment: Optional[str] = None
    ) -> Optional[str]:
        """
        設備を決定する（既存のapp.pyロジックと同じ）
        
        Args:
            question: ユーザーの質問
            selection_mode: "manual", "auto", "category"
            manual_equipment: 手動選択された設備名
            
        Returns:
            決定された設備名 または None
        """
        if selection_mode == "auto":
            # 自動推定
            if not self.equipment_data:
                logger.warning("⚠️ 設備データがないため自動推定をスキップ")
                return None
            
            # 設備推定のインポートを関数内で行う（循環インポート回避）
            try:
                from src.rag_qa import detect_equipment_from_question
                available_equipment = list(self.equipment_data.keys())
                target_equipment = detect_equipment_from_question(question, available_equipment)
                
                if target_equipment:
                    logger.info(f"🤖 自動推定された設備: {target_equipment}")
                else:
                    logger.info("❓ 質問文から設備を推定できませんでした")
                
                return target_equipment
            except ImportError as e:
                logger.error(f"❌ 設備推定機能のインポートエラー: {e}")
                return None
        
        elif selection_mode in ["manual", "category"]:
            # 手動選択
            if manual_equipment and self.equipment_data and manual_equipment in self.equipment_data:
                logger.info(f"🔧 手動選択された設備: {manual_equipment}")
                return manual_equipment
            else:
                logger.info("⚠️ 設備が選択されていないか、設備データに存在しません")
                return None
        
        else:
            logger.warning(f"⚠️ 不明な選択モード: {selection_mode}")
            return None
    
    def get_selected_files(
        self, 
        target_equipment: str
    ) -> Optional[List[str]]:
        """
        選択されたファイルを取得（Streamlit session_state から）
        
        Args:
            target_equipment: 対象設備名
            
        Returns:
            選択されたファイル名のリスト または None
        """
        if not STREAMLIT_AVAILABLE or not target_equipment:
            return None
        
        try:
            selected_files_key = f"selected_files_{target_equipment}"
            selected_files = st.session_state.get(selected_files_key)
            
            if selected_files:
                logger.info(f"📄 使用ファイル: {len(selected_files)}個のファイルを使用")
                return selected_files
            else:
                logger.warning("⚠️ 使用するファイルが選択されていません")
                return None
                
        except Exception as e:
            logger.error(f"❌ ファイル選択取得エラー: {e}")
            return None
    
    def generate_answer(
        self,
        *,
        prompt: str,
        question: str,
        model: str = "claude-4-sonnet",
        selection_mode: str = "manual",
        manual_equipment: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        force_no_equipment: bool = False
    ) -> Dict[str, Any]:
        """
        統一された回答生成メソッド
        
        Args:
            prompt: システムプロンプト
            question: ユーザーの質問
            model: 使用するモデル名
            selection_mode: 設備選択モード ("manual", "auto", "category")
            manual_equipment: 手動選択された設備名
            chat_history: チャット履歴
            temperature: 温度パラメータ
            max_tokens: 最大トークン数
            force_no_equipment: 強制的に設備資料なしモードにする
            
        Returns:
            Dict[str, Any]: 回答結果辞書
        """
        
        logger.info(f"🚀 Smart Answer Generation開始: model={model}, mode={selection_mode}")
        
        processing_mode = "no_equipment"  # デフォルト
        target_equipment = None
        selected_files = None
        
        # === 1. 設備決定フェーズ ===
        if not force_no_equipment and self.equipment_data:
            target_equipment = self.determine_target_equipment(
                question, selection_mode, manual_equipment
            )
            
            if target_equipment:
                # === 2. ファイル選択フェーズ ===
                selected_files = self.get_selected_files(target_equipment)
                
                if selected_files:
                    processing_mode = "equipment_with_files"
                    logger.info("📋 処理モード: 設備資料あり")
                else:
                    # ファイルが選択されていない場合は設備なしモードに変更
                    logger.info("📋 ファイル未選択のため設備なしモードに変更")
                    target_equipment = None
                    processing_mode = "no_equipment"
        
        # === 3. 回答生成フェーズ ===
        try:
            if processing_mode == "equipment_with_files":
                # 設備資料ありモード
                chain = ChainManager.create_equipment_qa_chain(model, prompt, temperature, max_tokens)
                
                chain_input = {
                    "question": question,
                    "target_equipment": target_equipment,
                    "selected_files": selected_files,
                    "equipment_data": self.equipment_data,
                    "chat_history": chat_history[:-1] if chat_history and len(chat_history) > 1 else None
                }
                
                answer = chain.invoke(chain_input)
                
                # 結果情報を準備
                equipment_info = self.equipment_data.get(target_equipment, {})
                sources = selected_files if selected_files else equipment_info.get("sources", [])
                
                # コンテキスト長を計算
                context_length = 0
                if target_equipment in self.equipment_data:
                    eq_data = self.equipment_data[target_equipment]
                    available_files = eq_data["files"]
                    if selected_files:
                        context_length = sum(len(available_files.get(f, "")) for f in selected_files if f in available_files)
                    else:
                        context_length = sum(len(text) for text in available_files.values())
                
                result = {
                    "answer": answer,
                    "used_equipment": target_equipment,
                    "equipment_info": equipment_info,
                    "sources": sources,
                    "selected_files": selected_files or [],
                    "context_length": context_length,
                    "images": [],
                    "langchain_used": True,
                    "processing_mode": processing_mode
                }
                
            else:
                # 設備資料なしモード
                chain = ChainManager.create_simple_qa_chain(model, prompt, temperature, max_tokens)
                
                chain_input = {
                    "question": f"【質問】\n{question}\n\n設備資料は利用せず、あなたの知識に基づいて回答してください。",
                    "chat_history": chat_history[:-1] if chat_history and len(chat_history) > 1 else None
                }
                
                answer = chain.invoke(chain_input)
                
                result = {
                    "answer": answer,
                    "used_equipment": "なし（一般知識による回答）",
                    "equipment_info": {},
                    "sources": [],
                    "selected_files": [],
                    "context_length": 0,
                    "images": [],
                    "langchain_used": True,
                    "processing_mode": processing_mode
                }
            
            logger.info(f"✅ Smart Answer Generation完了: mode={processing_mode}, 回答文字数={len(result['answer'])}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Smart Answer Generation エラー: {e}", exc_info=True)
            raise

# === 便利関数（app.pyから呼び出し用） ===

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
    app.pyから呼び出す統一インターフェース
    設備選択、ファイル選択、資料なし処理を全て自動で処理
    
    Args:
        prompt: システムプロンプト
        question: ユーザーの質問
        model: 使用するモデル名
        equipment_data: 設備データ
        chat_history: チャット履歴
        temperature: 温度パラメータ
        max_tokens: 最大トークン数
        
    Returns:
        回答結果辞書
    """
    
    # Streamlit session_stateから設定を取得
    selection_mode = "manual"
    manual_equipment = None
    
    if STREAMLIT_AVAILABLE:
        try:
            selection_mode = st.session_state.get("selection_mode", "manual")
            manual_equipment = st.session_state.get("selected_equipment")
        except Exception as e:
            logger.warning(f"⚠️ session_state取得エラー: {e}")
    
    # SmartAnswerGeneratorで処理
    generator = SmartAnswerGenerator(equipment_data)
    
    return generator.generate_answer(
        prompt=prompt,
        question=question,
        model=model,
        selection_mode=selection_mode,
        manual_equipment=manual_equipment,
        chat_history=chat_history,
        temperature=temperature,
        max_tokens=max_tokens
    )

# === 後方互換性のための関数 ===

def generate_answer_with_langchain(
    *,
    prompt: str,
    question: str,
    model: str = "claude-4-sonnet",
    target_equipment: Optional[str] = None,
    selected_files: Optional[List[str]] = None,
    equipment_data: Optional[Dict[str, Dict[str, Any]]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    既存のgenerate_answer_with_equipment互換関数（後方互換性のため）
    """
    
    logger.info(f"🚀 LangChain回答生成開始（互換モード）: model={model}, equipment={target_equipment}")
    
    try:
        if target_equipment and equipment_data:
            # 設備資料ありモード
            chain = ChainManager.create_equipment_qa_chain(model, prompt, temperature, max_tokens)
            
            chain_input = {
                "question": question,
                "target_equipment": target_equipment,
                "selected_files": selected_files,
                "equipment_data": equipment_data,
                "chat_history": chat_history[:-1] if chat_history and len(chat_history) > 1 else None
            }
            
            answer = chain.invoke(chain_input)
            
            # 結果情報を準備
            equipment_info = equipment_data.get(target_equipment, {})
            sources = selected_files if selected_files else equipment_info.get("sources", [])
            
            # コンテキスト長を計算
            context_length = 0
            if target_equipment in equipment_data:
                eq_data = equipment_data[target_equipment]
                available_files = eq_data["files"]
                if selected_files:
                    context_length = sum(len(available_files.get(f, "")) for f in selected_files if f in available_files)
                else:
                    context_length = sum(len(text) for text in available_files.values())
            
            result = {
                "answer": answer,
                "used_equipment": target_equipment,
                "equipment_info": equipment_info,
                "sources": sources,
                "selected_files": selected_files or [],
                "context_length": context_length,
                "images": [],
                "langchain_used": True
            }
            
        else:
            # 設備資料なしモード
            chain = ChainManager.create_simple_qa_chain(model, prompt, temperature, max_tokens)
            
            chain_input = {
                "question": f"【質問】\n{question}\n\n設備資料は利用せず、あなたの知識に基づいて回答してください。",
                "chat_history": chat_history[:-1] if chat_history and len(chat_history) > 1 else None
            }
            
            answer = chain.invoke(chain_input)
            
            result = {
                "answer": answer,
                "used_equipment": "なし（一般知識による回答）",
                "equipment_info": {},
                "sources": [],
                "selected_files": [],
                "context_length": 0,
                "images": [],
                "langchain_used": True
            }
        
        logger.info(f"✅ LangChain回答生成完了（互換モード）: 回答文字数={len(result['answer'])}")
        return result
        
    except Exception as e:
        logger.error(f"❌ LangChain回答生成エラー（互換モード）: {e}", exc_info=True)
        raise

# テスト用関数
def test_chain_creation():
    """チェーン作成のテスト"""
    try:
        logger.info("🧪 Simple QA Chain test...")
        simple_chain = ChainManager.create_simple_qa_chain(
            "claude-4-sonnet",
            "あなたは親切なアシスタントです。",
            temperature=0.0
        )
        logger.info("✅ Simple QA Chain 作成成功")
        
        logger.info("🧪 Equipment QA Chain test...")
        equipment_chain = ChainManager.create_equipment_qa_chain(
            "claude-4-sonnet",
            "あなたは建築設備の専門家です。",
            temperature=0.0
        )
        logger.info("✅ Equipment QA Chain 作成成功")
        
        logger.info("🧪 Smart Answer Generator test...")
        generator = SmartAnswerGenerator()
        logger.info("✅ Smart Answer Generator 作成成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Chain作成テスト失敗: {e}")
        return False

if __name__ == "__main__":
    test_chain_creation()