import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any
import time, functools
import os

from src.rag_preprocess import preprocess_files
from src.rag_vector import save_docs_to_chroma
from src.rag_qa import generate_answer
from src.startup_loader import initialize_chroma_from_input
from src.logging_utils import init_logger
from src.sheets_manager import log_to_sheets, get_sheets_manager, send_prompt_to_model_comparison

import yaml
import streamlit_authenticator as stauth
import uuid

import threading
import queue
from typing import Optional
import atexit
import copy

st.set_page_config(page_title="GPT + RAG Chatbot", page_icon="💬", layout="wide")

logger = init_logger()

# Azure OpenAI設定を追加
def setup_azure_openai():
    """Azure OpenAI設定"""
    # 環境変数から取得（Streamlit Secretsでも可能）
    try:
        azure_endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
        azure_key = st.secrets.get("AZURE_OPENAI_KEY", os.getenv("AZURE_OPENAI_KEY"))
    except:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_KEY")
    
    if not azure_endpoint or not azure_key:
        st.error("Azure OpenAI の設定が不足しています。環境変数またはSecrets.tomlを確認してください。")
        st.stop()
    
    return OpenAI(
        api_key=azure_key,
        base_url=f"{azure_endpoint}/openai",
        default_query={"api-version": "2025-04-01-preview"},
        default_headers={"api-key": azure_key}
    )

# Azure用のモデル名マッピング
AZURE_MODEL_MAPPING = {
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini", 
    "gpt-4.1-nano": "gpt-4.1-nano",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini"
}

def get_azure_model_name(model_name: str) -> str:
    """OpenAIモデル名をAzureデプロイメント名に変換"""
    return AZURE_MODEL_MAPPING.get(model_name, model_name)

# =====  認証設定の読み込み ============================================================
with open('./config.yaml') as file:
    config = yaml.safe_load(file)

# 認証インスタンスの作成
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ===== post_log関数を修正 =====
def post_log(
        input_text: str,
        output_text: str,
        prompt: str,
        send_to_model_comparison: bool = False,
        user_info: dict = None,
        chat_messages: list = None,
    ):
        """Google Sheetsに直接ログを保存（gspread使用）- セッション状態対応版"""
        
        try:
            logger.info("🔍 post_log start — attempting to log conversation")
            
            # sheets_managerの状態確認
            logger.info("🔍 Step 1: Getting sheets manager...")
            try:
                manager = get_sheets_manager()
                logger.info("🔍 Step 2: Manager obtained — type=%s", type(manager).__name__)
                
                if not manager:
                    logger.error("❌ manager is None")
                    return
                    
                logger.info("🔍 Step 3: Checking connection — is_connected=%s", 
                        getattr(manager, 'is_connected', 'ATTR_NOT_FOUND'))
                
                if not manager.is_connected:
                    logger.error("❌ manager not connected")
                    return
                    
            except Exception as e:
                logger.error("❌ Step 1-3 failed — %s", e, exc_info=True)
                return
            
            # 1. conversationsシートへの保存
            logger.info("🔍 Step 4: Starting conversations sheet save...")
            try:
                # user_infoから必要な情報を取得
                if user_info:
                    username = user_info.get("username", "unknown")
                    design_mode = user_info.get("design_mode", "unknown")
                    session_id = user_info.get("session_id", "unknown")
                    gpt_model = user_info.get("gpt_model", "unknown")
                    temperature = user_info.get("temperature", 1.0)
                    max_tokens = user_info.get("max_tokens")
                    use_rag = user_info.get("use_rag", False)
                    chat_title = user_info.get("chat_title", "未設定")
                else:
                    # フォールバック値
                    username = design_mode = session_id = gpt_model = "unknown"
                    temperature = 1.0
                    max_tokens = None
                    use_rag = False
                    chat_title = "未設定"
                
                # log_to_sheetsに全ての情報を渡す
                success = log_to_sheets(
                    input_text=input_text,
                    output_text=output_text,
                    prompt=prompt,
                    chat_title=chat_title,
                    user_id=username,
                    session_id=session_id,
                    mode=design_mode,
                    model=gpt_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_rag=use_rag
                )
                
                logger.info("🔍 Step 5: log_to_sheets result — success=%s", success)
                
                if success:
                    logger.info("✅ conversations sheet success — user=%s mode=%s", 
                            username, design_mode)
                else:
                    logger.warning("⚠️ conversations sheet failed — log_to_sheets returned False")
                    
            except Exception as e:
                logger.error("❌ Step 4-5 failed — %s", e, exc_info=True)
            
            # 2. model比較シートへの保存（オプション）
            if send_to_model_comparison and chat_messages is not None:
                logger.info("🔍 Step 6: Starting model comparison sheet save...")
                try:
                    # 事前に取得したメッセージを使用
                    msgs = chat_messages
                    
                    # 完全なプロンプトを構築（実際のAPI呼び出しと同じ形式）
                    full_prompt_parts = []
                    
                    # システムプロンプト
                    if prompt:
                        full_prompt_parts.append(f"System: {prompt}")
                    
                    # 会話履歴（最後のメッセージ以外）
                    for msg in msgs[:-1]:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user":
                            full_prompt_parts.append(f"Human: {content}")
                        elif role == "assistant":
                            full_prompt_parts.append(f"Assistant: {content}")
                    
                    # 現在のユーザー入力
                    full_prompt_parts.append(f"Human: {input_text}")
                    
                    # 完全なプロンプトを作成
                    comparison_prompt = "\n\n".join(full_prompt_parts)
                    
                    logger.info("🔍 Step 7: Sending to model comparison sheet...")
                    
                    # model比較シートに送信（プロンプトのみ）
                    model_success = send_prompt_to_model_comparison(
                        prompt_text=comparison_prompt,
                        user_note=None  # 使用しない
                    )
                    
                    logger.info("🔍 Step 8: model comparison result — success=%s", model_success)
                    
                    if model_success:
                        logger.info("✅ model comparison sheet success")
                    else:
                        logger.warning("⚠️ model comparison sheet failed")
                        
                except Exception as e:
                    logger.error("❌ Step 6-8 failed — %s", e, exc_info=True)
            elif send_to_model_comparison:
                logger.warning("⚠️ model comparison requested but chat_messages is None")
            
            logger.info("🔍 post_log completed successfully")
                
        except Exception as e:
            logger.error("❌ post_log outer error — %s", e, exc_info=True)

class StreamlitAsyncLogger:
    """Streamlit向け非同期ログ処理クラス"""
    
    def __init__(self):
        self.log_queue = queue.Queue(maxsize=100)  # キューサイズ制限
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.stats = {
            "processed": 0,
            "errors": 0,
            "last_process_time": None,
            "last_error_time": None,
            "last_error_msg": None
        }
        self._lock = threading.Lock()
        self.start_worker()
    
    def start_worker(self):
        """ワーカースレッドを開始"""
        with self._lock:
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.shutdown_event.clear()
                self.worker_thread = threading.Thread(
                    target=self._worker_loop,
                    daemon=True,  # Streamlitではdaemon=Trueが適切
                    name="StreamlitAsyncLogger"
                )
                self.worker_thread.start()
                logger.info("🚀 StreamlitAsyncLogger worker started")
    
    def _worker_loop(self):
        """ワーカースレッドのメインループ"""
        while not self.shutdown_event.is_set():
            try:
                # タイムアウト付きでキューから取得
                log_data = self.log_queue.get(timeout=2.0)
                
                if log_data is None:  # shutdown シグナル
                    break
                
                # 実際のログ処理を実行
                self._process_log_safe(log_data)
                self.log_queue.task_done()
                
            except queue.Empty:
                continue  # タイムアウト時は継続
            except Exception as e:
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error_time"] = time.time()
                    self.stats["last_error_msg"] = str(e)
                logger.error("❌ AsyncLogger worker error — %s", e, exc_info=True)

    def _process_log_safe(self, log_data: dict):
        """安全なログ処理（例外処理付き）"""
        try:
            start_time = time.perf_counter()
            
            # 元のpost_log関数を呼び出し（事前取得した情報を渡す）
            post_log(
                input_text=log_data["input_text"],
                output_text=log_data["output_text"], 
                prompt=log_data["prompt"],
                send_to_model_comparison=log_data.get("send_to_model_comparison", False),
                user_info=log_data.get("user_info"),  # 新しく追加
                chat_messages=log_data.get("chat_messages")  # 新しく追加
            )
            
            elapsed = time.perf_counter() - start_time
            
            # 統計情報を更新
            with self._lock:
                self.stats["processed"] += 1
                self.stats["last_process_time"] = time.time()
            
            logger.info("✅ Async log completed — elapsed=%.2fs processed=%d", 
                       elapsed, self.stats["processed"])
            
        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error_time"] = time.time()
                self.stats["last_error_msg"] = str(e)
            logger.error("❌ Async log processing failed — %s", e, exc_info=True)
    
    def post_log_async(self, input_text: str, output_text: str, prompt: str, 
                       send_to_model_comparison: bool = False,
                       user_info: dict = None, chat_messages: list = None):
        """非同期ログ投稿"""
        # ワーカーが生きているか確認し、必要に応じて再起動
        if not self.worker_thread or not self.worker_thread.is_alive():
            logger.warning("⚠️ Worker thread not alive, restarting...")
            self.start_worker()
        
        log_data = {
            "input_text": input_text,
            "output_text": output_text,
            "prompt": prompt,
            "send_to_model_comparison": send_to_model_comparison,
            "timestamp": time.time(),
            "session_id": user_info.get("session_id", "unknown") if user_info else "unknown",
            "user": user_info.get("username", "unknown") if user_info else "unknown",
            "user_info": user_info,  # 新しく追加
            "chat_messages": chat_messages  # 新しく追加
        }
        
        try:
            # ノンブロッキングでキューに追加
            self.log_queue.put_nowait(log_data)
            logger.info("📝 Log queued — queue_size=%d", self.log_queue.qsize())
            
        except queue.Full:
            logger.error("❌ Log queue is full — dropping log entry")
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error_msg"] = "Queue full - log dropped"
    
    def get_status(self) -> Dict[str, Any]:
        """現在のステータスを取得"""
        with self._lock:
            return {
                "queue_size": self.log_queue.qsize(),
                "worker_alive": self.worker_thread.is_alive() if self.worker_thread else False,
                "shutdown_requested": self.shutdown_event.is_set(),
                "stats": self.stats.copy()
            }
    
    def force_shutdown(self, timeout: float = 5.0):
        """強制シャットダウン（主にデバッグ用）"""
        logger.info("🛑 Force shutting down AsyncLogger...")
        self.shutdown_event.set()
        
        # 可能な限りキューを空にする
        try:
            while not self.log_queue.empty():
                self.log_queue.get_nowait()
                self.log_queue.task_done()
        except queue.Empty:
            pass
        
        self.log_queue.put_nowait(None)  # worker終了シグナル
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
        
        logger.info("✅ AsyncLogger force shutdown completed")

# セッション状態でインスタンスを管理
def get_async_logger() -> StreamlitAsyncLogger:
    """StreamlitAsyncLoggerのインスタンスを取得（セッション管理）"""
    if "async_logger" not in st.session_state:
        st.session_state.async_logger = StreamlitAsyncLogger()
    
    # インスタンスが無効になっている場合は再作成
    async_logger = st.session_state.async_logger
    if not async_logger.worker_thread or not async_logger.worker_thread.is_alive():
        logger.warning("⚠️ AsyncLogger instance invalid, creating new one")
        st.session_state.async_logger = StreamlitAsyncLogger()
        async_logger = st.session_state.async_logger
    
    return async_logger

def post_log_async(input_text: str, output_text: str, prompt: str, 
                   send_to_model_comparison: bool = False):
    """非同期ログ投稿の便利関数（セッション状態対応）"""
    try:
        # デバッグ: セッション状態の内容を確認
        logger.info("🔍 Collecting session state info...")
        
        # セッション状態から必要な情報をすべて取得
        username = st.session_state.get("username") or st.session_state.get("name")
        design_mode = st.session_state.get("design_mode")
        session_id = st.session_state.get("sid")
        gpt_model = st.session_state.get("gpt_model")
        temperature = st.session_state.get("temperature", 1.0)
        max_tokens = st.session_state.get("max_tokens")
        use_rag = st.session_state.get("use_rag", False)
        chat_title = st.session_state.get("current_chat", "未設定")
        
        # デバッグログ
        logger.info("🔍 Session state values — username=%s design_mode=%s gpt_model=%s", 
                   username, design_mode, gpt_model)
        
        user_info = {
            "username": username or "unknown",
            "design_mode": design_mode or "unknown",
            "session_id": session_id or "unknown",
            "gpt_model": gpt_model or "unknown",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_rag": use_rag,
            "chat_title": chat_title
        }
        
        logger.info("🔍 Final user_info — %s", user_info)
        
        # チャットメッセージも事前に取得（deep copyで安全に）
        chat_messages = None
        if send_to_model_comparison:
            try:
                current_chat = st.session_state.get("current_chat", "New Chat")
                chats_dict = st.session_state.get("chats", {})
                msgs = chats_dict.get(current_chat, [])
                
                logger.info("🔍 Chat info — current_chat=%s msgs_count=%d", 
                           current_chat, len(msgs))
                
                # 深いコピーを作成（参照ではなく値をコピー）
                import copy
                chat_messages = copy.deepcopy(msgs)
                
            except Exception as e:
                logger.error("❌ Failed to get chat messages — %s", e)
                chat_messages = []
        
        logger_instance = get_async_logger()
        logger_instance.post_log_async(
            input_text, output_text, prompt, send_to_model_comparison,
            user_info=user_info, chat_messages=chat_messages
        )
        
    except Exception as e:
        logger.error("❌ post_log_async failed — %s", e)
        # フォールバック: 同期処理で確実にログを保存
        try:
            logger.warning("⚠️ Falling back to synchronous logging")
            post_log(input_text, output_text, prompt, send_to_model_comparison)
        except Exception as fallback_error:
            logger.error("❌ Fallback logging also failed — %s", fallback_error)

# =====  基本設定（Azure OpenAI対応）  ============================================================
client = setup_azure_openai()

# =====  ログインUIの表示  ============================================================
authenticator.login()

if st.session_state["authentication_status"]:
    name = st.session_state["name"]
    username = st.session_state["username"]

    logger.info("🔐 login success — user=%s  username=%s", name, username)

    # Chromaコレクションを input_data から自動初期化（persist_directory=None → インメモリ）
    if st.session_state.get("rag_collection") is None:
        try:
            res = initialize_chroma_from_input(
                input_dir="rag_data",
                persist_dir=None,  # 永続化しない
                collection_name="session_docs"
            )
            st.session_state.rag_collection = res["collection"]
            st.session_state.rag_files = res["rag_files"]

            logger.info("📂 Chroma init — chunks=%d  files=%d",
                    res["collection"].count(), len(res["rag_files"]))
            
        except Exception as e:
            logger.exception("❌ Chroma init failed — %s", e)
            st.warning(f"RAG初期化中にエラーが発生しました: {e}")

    # --------------------------------------------------------------------------- #
    #                         ★ 各モード専用プロンプト ★                           #
    # --------------------------------------------------------------------------- #
    DEFAULT_PROMPTS: Dict[str, str] = {
        "暗黙知法令チャットモード": """
    あなたは建築電気設備設計のエキスパートエンジニアです。
    今回の対象は **複合用途ビルのオフィス入居工事（B工事）** に限定されます。
    以下の知識と技術をもとに、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。
    専門用語は必要に応じて解説を加え、判断の背景にある理由を丁寧に説明します。
    **重要：以下の各事項は「代表的なビル（丸の内ビルディング）」を想定して記載しています。他のビルでは仕様や基準が異なる可能性があることを、回答時には必ず言及してください。**
    **注意：過度に込み入った条件の詳細説明をユーザーに求めることは避け、一般的な設計基準に基づく実務的な回答を心がけてください。**
    ────────────────────────────────
    ## 【工事区分について】
    - **B工事**：本システムが対象とする工事。入居者負担でビル側が施工する工事
    - **C工事**：入居者が独自に施工する工事（電話・LAN・防犯設備など）
    - 本システムでは、C工事設備については配管類の数量算出のみを行います
    ────────────────────────────────
    ## 【ビル共通要件】
    1. **防火対象建物区分**
    - 消防法施行令 区分「（16）複合用途防火対象物　イ」
    - 当該階床面積 1000 m² 以上
    ────────────────────────────────
    ## 【コンセント設備（床）】
    ### ■ 図面の指示と基本的な割り振り
    - 図面や要望書の指示を優先（単独回路や専用ELB等）
    - 機器やデスクの配置をもとにグループ化
    - 一般的なオフィス机は複数の座席をまとめて1回路
    - 機器の消費電力が高い場合や同時使用想定で回路分割
    ### ■ 机・椅子（デスク周り）の標準設計とOAタップ仕様
    - **個人用デスク**： 1席ごとにOAタップ1個、6席ごとに1回路（300VA/席）
    - 使用機器：分岐 4口タップ（300VA）
    - **フリーアドレスデスク**：1席ごとにOAタップ1個、8席ごとに1回路（150VA/席）
    - 使用機器：分岐 4口タップ（150VA）
    - **昇降デスク**：1席ごとにOAタップ1個、2席ごとに1回路（600VA/席）
    - 使用機器：分岐 4口タップ（600VA）
    - **会議室テーブル**：4席ごとにOAタップ1個、12席ごとに1回路（150VA/席）
    - 使用機器：分岐 4口タップ（600VA）
    ### ■ 設備機器の設計とOAタップ仕様
    - **単独回路が必要な機器**
    - 通常機器用：単独 2口タップ（単独回路）
    - 水気のある機器用：単独 ELB 2口タップ（単独回路、ELB付き）
    - 対象機器：複合機（コピー機）、プリンター、シュレッダー、テレブース、自動販売機、冷蔵庫、ウォーターサーバー、電子レンジ、食器洗い乾燥機、コーヒーメーカー、ポット、造作家具（什器用コンセント）、インターホン親機、セキュリティシステム、等
    - **サーバーラック**：什器1つにつき2回路必要（単独 2口タップを2個設置）
    - **分岐回路でもよい機器**
    - 使用機器：分岐 4口タップ（150VA/300VA/600VA/1200VA）（容量に応じて選択）
    - 対象機器：ディスプレイ（会議室、応接室、役員室）、テレビ（共用）、スタンド照明、ロッカー（電源供給機能付）、等
    - 300〜1200VA程度の機器は近い位置で1回路にまとめ可能（1500VA上限）
    ### ■ 特殊エリアの電源
    - パントリー：最低OAタップ5個と5回路
    - エントランス：最低OAタップ1個と1回路
    - プリンター台数：20人に1台が目安、40人に1台が確保できてなければ電源の追加を提案
    ────────────────────────────────
    ## 【コンセント設備（壁）※清掃用電源】
    ### ■ 用途と設置考え方
    - 清掃時に掃除機を接続するための電源（入居企業は使用不可）
    - 見積図面では提案するが、入居企業の要望により省略も可能
    - 設置位置は主に扉横
    ### ■ 配置判断ルール
    - 清掃時の動線（≒避難経路）を考慮して配置
    - 扉を挟んだどちら側に設置するかの精査が必要
    - 各部屋の入口付近に最低1箇所
    ────────────────────────────────
    ## 【コンセント設備（壁）※客先指示】
    ### ■ 設置基準
    - 見積依頼図に指示された場所、指示された仕様で設置
    - 客先からの特殊指示（単独回路、専用ELB等）を最優先
    - 図面上の明示がなくても打合せ記録等で指示があれば対応
    ### ■ 追加提案判断
    - 見積図に指示がなくても、使用目的が明確な場合は追加提案
    - 特殊機器（給湯器、加湿器等）の近くには設置を提案
    ────────────────────────────────
    ## 【コンセント設備（天井）】
    ### ■ 設置基準
    - 見積依頼図に指示があった場所に設置
    - 電源が必要な天井付近の機器がある場合に1個設置
    ### ■ 対象機器
    - プロジェクター
    - 電動スクリーン
    - 電動ブラインド
    - 壁面発光サイン
    - その他天井付近に設置される電気機器
    ────────────────────────────────
    ## 【自動火災報知設備】
    ### ■ 感知器の種類・仕様
    - **丸ビル標準**
    - 廊下：煙感知器スポット型2種（全ビル共通）
    - 居室：煙感知器スポット型2種（丸ビル標準）
    - 厨房：定温式スポット型（1種）
    - **他ビルでの例**
    - 居室で熱感知器（差動式スポット型1種等）を使用するビルもある
    - 天井面中央付近、または障害を避けて煙が集まりやすい位置に設置
    ### ■ 設置基準
    - 廊下：端点から15m以内、感知器間30m以内
    - 居室：面積150m²ごとに1個（切り上げ）
    - 煙を遮る障害物がある場合は個数増
    - 天井高2.3m未満または40m²未満の居室は入口付近
    - 吸気口付近に設置、排気口付近は避ける
    - 防火シャッター近くは専用感知器（煙感知器スポット型3種等）
    ────────────────────────────────
    ## 【非常放送設備】
    ### ■ スピーカ設置基準
    - 到達距離10m以内（各居室・廊下を半径10mの円でカバー）
    - パーテーションや什器による遮音は考慮しない（半径10mの円は不変）
    ### ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷200㎡）の切り上げ
    - 200㎡は「L級スピーカー」の有効範囲円（半径10m）に内接する正方形の面積
    ### ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 省略可能条件（居室・廊下は6m²以下、その他区域は30m²以下、かつ隣接スピーカから8m以内なら省略可能）は適用しない（丸ビル方針）
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置
    ────────────────────────────────
    ## 【誘導灯設備】
    ### ■ 種類・採用機種
    - 避難口誘導灯・通路誘導灯のみ使用
    - 両者ともB級BH型（20A形）のみ使用（丸ビル標準）
    ### ■ 設置箇所・有効距離
    - 避難口誘導灯：最終避難口、または最終避難口に通じる避難経路上の扉
    有効距離30m（シンボル無）／20m（矢印付き）
    - 通路誘導灯：廊下の曲がり角や分岐点、または避難口誘導灯の有効距離補完
    有効距離15m
    ### ■ 配置判断
    - 扉開閉・パーテーション・背の高い棚などで視認阻害→位置変更または追加
    ────────────────────────────────
    ## 【非常照明設備】
    ### ■ 照度条件
    - 常温下の床面で1lx以上を確保（建築基準法施行令第126条の5）
    - 照度計算は逐点法を用いる（カタログの1lx到達範囲表使用）
    ### ■ 器具仕様・種別
    - バッテリー別置型：ビル基本設備分（入居前既設分）
    - バッテリー内蔵型：B工事追加分（間仕切り変更などで追加した分）
    ### ■ 設置判断ルール
    - 天井高別の1lx到達範囲表を用い、器具間隔を決定
    - パーテーション・什器で遮光の恐れがあれば器具を追加
    - 2018年改正の個室緩和（30m²以下は不要）は適用しない（丸ビル方針）
    ### ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷50㎡）の切り上げ
    - 50㎡は新丸ビルにおける非常照明設備の有効範囲円（半径5.0m）に内接する正方形の面積
    ### ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置
    ────────────────────────────────
    ## 【照明制御設備（照度センサ）】
    ### ■ 設置判断ルール
    - 天井高別の有効範囲表を用い、器具間隔を決定
    - パーテーション・什器で遮られる恐れがあれば器具を追加
    ### ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷28㎡）の切り上げ
    - 28㎡は新丸ビルにおける照度センサの有効範囲円（半径3.75m）に内接する正方形の面積
    ### ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置
    ────────────────────────────────
    ## 【照明制御設備（スイッチ）】
    ### ■ 設置判断ルール
    - 新規に間仕切りされた領域に対してそれぞれ設置
    - 設置するスイッチ数は領域の大きさや扉の配置、制御の分け方による
    ### ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷20㎡）の切り上げ
    - 算出個数に基づき、「最終避難口」以外の「扉」の横に配置（最終避難口の横にはビル基本のスイッチがあるため追加設置不要）
    ### ■ 配置ルール
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 配置数2個以上かつ扉数2個以上の場合は、領域内の「扉」の横に均等に配置
    - 扉数＞個数の場合は最終避難口への距離が短い扉から優先的に配置
    - 本来は入退室ルート（≒避難経路）に基づく動線計画に従い、設置位置を精査
    ────────────────────────────────
    ## 【テレビ共聴設備】
    ### ■ 設置基準
    - 見積依頼図に指示があった場所に設置
    - テレビ共聴設備が必要な什器がある場所に1個設置
    ### ■ 設置が必要な部屋・什器
    - 会議室：最低1個は設置
    - 応接室：最低1個は設置
    - 役員室：最低1個は設置
    - ディスプレイ（会議室、応接室、役員室にあるもの）
    - テレビ（共用のもの）
    ────────────────────────────────
    ## 【電話・LAN設備（配管）】【防犯設備（配管）】（C工事設備）
    ### ■ 業務基本原則
    - 本設備はC工事のため、B工事では配管の設置のみを行う
    - 基本的には客先から図面を受領して見積りを作成
    - C工事会社から配管の設置のみ依頼される場合が多い
    ### ■ 概算見積りの考え方
    - 概算段階では配管図を作成せず、細部計算を省略することが一般的
    - 「設備数×○m」という形式で概算を算出
    - 各ビル・各設備ごとの「黄金数字（○m）」を考慮した設計が必要
    ────────────────────────────────
    ## 【動力設備（配管、配線）】
    ### ■ 適用場面と業務原則
    - 基本的には客先から図面を受領して見積りを作成
    - 店舗（特に飲食店）では必要性が高い
    - オフィスでも稀に必要となるケースがある
    ### ■ 概算見積りの特徴
    - 概算段階では配置平面図よりも、必要な動力設備の種類と数をまとめた表から算出
    - 表を読み解いて必要数を算出し見積りに反映
    ### ■ オフィスでの対応
    - オフィスで必要な場合：動力用の分電盤、配線・配管を設置
    - 詳細な設計検討が必要（概算見積対応はできない）
    ────────────────────────────────
    ## 【重要な注意事項】
    1. **ビル仕様の違い**：上記の内容は丸の内ビルディングを基準としています。他のビルでは異なる仕様・基準が適用される可能性があります。
    2. **過度な詳細要求の回避**：ユーザーに対して、込み入った条件の詳細説明を過度に求めることは避けてください。
    3. **工事区分の明確化**：B工事とC工事の区分を常に意識し、C工事設備については配管類のみを扱うことを明確にしてください。
    4. **法令準拠**：検索結果の言い回しをそのまま複製することを避け、直接引用以外のすべてを自分の言葉で表現します。
    """,

        "質疑応答書添削モード": """
    あなたは建築電気設備分野における質疑応答書作成の専門家です。
    ユーザーが入力した文章を、見積根拠図や見積書と一緒に提出する質疑応答書として最適な文章に添削してください。

    【重要】添削文のみを出力し、添削内容の説明は一切不要です。

    【添削・整形の仕様】
    1. **誤字脱字の修正**
        - 一般的な誤記、表記揺れを修正し、読みやすく整えます。

    2. **表現の統一・調整**
        - 質疑応答書として適切かつ丁寧な表現に統一・調整します
        - 文体は敬体（です・ます調）に統一します
        - 過度な敬語や冗長な表現は避け、簡潔で分かりやすい表現に修正します
        - 専門用語は業界標準に則って表記統一します

    3. **見積・提案の文脈に合わせた表現**
        - 指示がなくても合理的に見積もれる内容であれば、**確認文を使わずに断定的に表現**してください。
        例：「○○については□□として見込んでおります。」
        - 情報が明らかに不足しており、仕様決定の判断ができない場合のみ、
        **前提を提示したうえで控えめに確認を促す表現**としてください。
        例：「図面記載がないため、○○として想定しておりますが、仕様のご確認をお願いいたします。」

    4. **クローズドクエスチョンへの変換**
        - 「〜でよろしいでしょうか？」「〜でしょうか？」といった**クローズドクエスチョン表現は使用しないでください。**
        - 「〜と見込んでおります」や「〜とさせていただきたいと考えております」といった**先方のリアクションがなくてもそのまま見積を行えるような文章**が理想です。

    【変換例】
    変換前：
    家具コンセント・テレキューブが設置される場所に関してはOA内にOAタップを設置する認識でよろしいでしょうか。
    変換後：
    家具コンセント・テレキューブが設置される場所については、OA内にOAタップを設置する前提としております。

    変換前：
    NW工事（光ケーブル、電話含め）、AV工事は全てC工事という認識でよろしいですね。
    変換後：
    NW工事（光ケーブル、電話含む）およびAV工事は、全てC工事区分として想定しております。

    変換前：
    ＴＶ共聴信号については、壁埋め込みとしコンセントと２連での設置でよろしいでしょうか。また、口数はいくつ必要でしょうか。
    変換後：
    TV共聴信号については、コンセントと2連の壁埋め込み型で設置する想定です。必要な口数は未記載のため、ご指示をお願いいたします。

    【出力】
    添削内容を1つだけ出力してください。説明や理由などの付加情報は一切不要です。
    出力は添削した質疑応答書の文章のみとしてください。

    【注意点】
    検索結果の言い回しをそのまま複製することを避け、直接引用以外のすべてを自分の言葉で表現します。
    """
    }

    # =====  セッション変数  =======================================================
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "chat_sids"   not in st.session_state:
        st.session_state.chat_sids = {"New Chat": str(uuid.uuid4())}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"
    if "sid"         not in st.session_state:
        st.session_state.sid = st.session_state.chat_sids["New Chat"]
    if "edit_target" not in st.session_state:
        st.session_state.edit_target = None
    if "rag_files" not in st.session_state:
        st.session_state.rag_files: List[Dict[str, Any]] = []
    if "rag_collection" not in st.session_state:
        st.session_state.rag_collection = None
    if "design_mode" not in st.session_state:
        st.session_state.design_mode = list(DEFAULT_PROMPTS.keys())[0]
    if "prompts" not in st.session_state:
        st.session_state.prompts = DEFAULT_PROMPTS.copy()
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-4.1"
    if "use_rag" not in st.session_state:
        st.session_state["use_rag"] = False  # ← デフォルトでRAGを使わない
    if "last_rag_sources" not in st.session_state:
        st.session_state.last_rag_sources = []
    if "last_rag_images" not in st.session_state:
        st.session_state.last_rag_images = []


    # =====  ヘルパー  ============================================================
    def get_messages() -> List[Dict[str, str]]:
        title = st.session_state.current_chat
        return st.session_state.chats.setdefault(title, [])
    
    def new_chat():
        title = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[title] = []
        st.session_state.chat_sids[title] = str(uuid.uuid4())
        st.session_state.current_chat = title
        st.session_state.sid = st.session_state.chat_sids[title]

        logger.info("➕ new_chat — sid=%s  title='%s'", st.session_state.sid, title)
        st.rerun()

    def switch_chat(title: str):
        if title not in st.session_state.chat_sids:
            st.session_state.chat_sids[title] = str(uuid.uuid4())
        st.session_state.current_chat = title
        st.session_state.sid = st.session_state.chat_sids[title]

        logger.info("🔀 switch_chat — sid=%s  title='%s'", st.session_state.sid, title)
        st.rerun()

    def rebuild_rag_collection():
        """
        アップロードされたファイルを前処理 → Chroma 登録し、セッションに保持
        """
        if not st.session_state.rag_files:
            st.warning("まず PDF / TXT をアップロードしてください")
            logger.warning("📚 RAG rebuild aborted — no files")
            return

        total_files = len(st.session_state.rag_files)
        logger.info("📚 RAG rebuild start — files=%d", total_files)

        t0 = time.perf_counter()

        try:
            with st.spinner("📚 ファイルを解析し、ベクトル DB に登録中..."):
                docs = preprocess_files(st.session_state.rag_files)
                col = save_docs_to_chroma(
                    docs=docs,
                    collection_name="session_docs",
                    persist_directory=None,
                )
                st.session_state.rag_collection = col

            chunk_count = col.count()
            elapsed = time.perf_counter() - t0
            logger.info("✅ RAG rebuild done — chunks=%d  files=%d  elapsed=%.2fs",
                        chunk_count, total_files, elapsed)

            st.success("🔍 検索インデックスを更新しました！")

        except Exception as e:
            logger.exception("❌ RAG rebuild failed — %s", e)
            st.error(f"RAG 初期化中にエラーが発生しました: {e}")

    def generate_chat_title(messages):
        if len(messages) >= 2:
            prompt = f"以下の会話の内容を25文字以内の簡潔なタイトルにしてください:\n{messages[0]['content'][:200]}"
            try:
                resp = client.chat.completions.create(
                    model=get_azure_model_name("gpt-4.1-nano"),  # Azure用に変換
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                )
                return resp.choices[0].message.content.strip('"').strip()
            except Exception as e:
                logger.error(f"Chat title generation failed: {e}")
                return f"Chat {len(st.session_state.chats) + 1}"
        return f"Chat {len(st.session_state.chats) + 1}"

    # =====  編集機能用のヘルパー関数  ==============================================
    def handle_save_prompt(mode_name, edited_text):
        st.session_state.prompts[mode_name] = edited_text
        st.session_state.edit_target = None

        logger.info("✏️ prompt_saved — mode=%s  len=%d", mode_name, len(edited_text))
        
        st.success(f"「{mode_name}」のプロンプトを更新しました")
        time.sleep(1)
        st.rerun()

    def handle_reset_prompt(mode_name):
        if mode_name in DEFAULT_PROMPTS:
            st.session_state.prompts[mode_name] = DEFAULT_PROMPTS[mode_name]

            logger.info("🔄 prompt_reset — mode=%s", mode_name)

            st.success(f"「{mode_name}」のプロンプトをデフォルトに戻しました")
            time.sleep(1)
            st.rerun()
        else:
            st.error("このモードにはデフォルト設定がありません")

    def handle_cancel_edit():
        st.session_state.edit_target = None
        st.rerun()

    # =====  CSS  ================================================================
    st.markdown(
        """
        <style>
        :root{ --sidebar-w:260px; --pad:1rem; }
        aside[data-testid="stSidebar"]{width:var(--sidebar-w)!important;}
        .chat-body{max-height:70vh;overflow-y:auto;}
        .stButton button {font-size: 16px; padding: 8px 16px;}

        /* モバイル対応 */
        @media (max-width: 768px) {
            :root{ --sidebar-w:100%; --pad:0.5rem; }
            .chat-body {max-height: 60vh;}
            .stButton button {font-size: 14px; padding: 6px 12px;}
        }

        /* ダークモード対応メッセージスタイル */
        .user-message, .assistant-message {
            border-radius: 10px;
            padding: 8px 12px;
            margin: 5px 0;
            border-left: 3px solid;
        }
        .user-message {
            border-left-color: #4c8bf5;
        }
        .assistant-message {
            border-left-color: #ff7043;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =====  サイドバー  ==========================================================
    with st.sidebar:
        st.markdown(f"👤 ログインユーザー: `{name}`")
        authenticator.logout('ログアウト', 'sidebar')

        st.divider()

        # ------- チャット履歴 -------
        st.header("💬 チャット履歴")
        for title in list(st.session_state.chats.keys()):
            if st.button(title, key=f"hist_{title}"):
                switch_chat(title)

        if st.button("➕ 新しいチャット"):
            new_chat()
        
        st.divider()

        # ------- モデル選択 -------
        st.markdown("### 🤖 GPTモデル選択")
        model_options = {
            "gpt-4.1": "GPT-4.1 (標準・最新世代)",
            "gpt-4.1-mini": "GPT-4.1-mini (小・最新世代)",
            "gpt-4.1-nano": "GPT-4.1-nano (超小型・高速)",
            "gpt-4o": "GPT-4o (標準・高性能)",
            "gpt-4o-mini": "GPT-4o-mini (小・軽量)"
        }
        st.session_state.gpt_model = st.selectbox(
            "使用するモデルを選択",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.gpt_model) if st.session_state.gpt_model in model_options else 0,
        )
        st.markdown(f"**🛈 現在のモデル:** `{model_options[st.session_state.gpt_model]}`")

        # ------- モデル詳細設定 -------
        with st.expander("🔧 詳細設定"):
            st.slider("応答の多様性",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.1,
                    key="temperature",
                    help="値が高いほど創造的、低いほど一貫した回答になります（OpenAIデフォルト: 1.0）")

            max_tokens_options = {
                "未設定（モデル上限）": None,
                "500": 500,
                "1000": 1000,
                "2000": 2000,
                "4000": 4000,
                "8000": 8000
            }
            selected_max_tokens = st.selectbox(
                "最大応答長",
                options=list(max_tokens_options.keys()),
                index=0,
                key="max_tokens_select",
                help="生成される回答の最大トークン数（OpenAIデフォルト: モデル上限）"
            )
            st.session_state["max_tokens"] = max_tokens_options[selected_max_tokens]

        st.divider()

        # ------- モード選択 -------
        st.markdown("### ⚙️ 設計対象モード")
        st.session_state.design_mode = st.radio(
            "対象設備を選択",
            options=list(st.session_state.prompts.keys()),
            index=0,
            key="design_mode_radio",
        )
        st.markdown(f"**🛈 現在のモード:** `{st.session_state.design_mode}`")

        # ------- プロンプト編集ボタン -------
        if st.button("✏️ 現在のプロンプトを編集"):
            st.session_state.edit_target = st.session_state.design_mode

        st.divider()

        # ===== RAG 検索の使用設定 =====
        st.markdown("### 🧠 RAG 検索の使用設定")

        st.session_state["use_rag"] = st.checkbox(
            "検索資料（ベクトルDB）を活用する",
            value=st.session_state["use_rag"],
            help="OFFにすると、プロンプトと履歴のみで応答を生成します"
        )

        if st.session_state["use_rag"]:
            st.success("現在のモード: RAG使用中")
        else:
            st.info("現在のモード: GPTのみ（検索なし）")

        # ベクトルDBステータス
        st.markdown("### 🗂 ベクトルDBステータス")

        if st.session_state.get("rag_collection"):
            st.success("✔️ ベクトルDBは初期化済みです")
            try:
                count = st.session_state.rag_collection.count()
                st.markdown(f"📄 登録チャンク数: `{count}`")
            except Exception as e:
                st.warning(f"⚠️ 件数取得失敗: {e}")
        else:
            st.error("❌ ベクトルDBがまだ初期化されていません")

        # ------- RAG アップロード -------
        st.markdown("### 📂 追加RAG 資料アップロード")
        uploads = st.file_uploader(
            "PDF / TXT を選択…",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )
        if uploads:
            st.session_state.rag_files = [
                {"name": f.name, "type": f.type, "size": f.size, "data": f.getvalue()} for f in uploads
            ]

            logger.info("📥 file_uploaded — files=%d  total_bytes=%d",
                len(uploads), sum(f.size for f in uploads))
            
        if st.button("🔄 インデックス再構築", disabled=not st.session_state.rag_files):
            rebuild_rag_collection()
        
        if st.button("🔧 接続診断実行"):
            from src.sheets_manager import debug_connection_streamlit
            debug_connection_streamlit()
        
        st.divider()
        st.markdown("### 🔧 ログ処理状況")

        try:
            async_logger = get_async_logger()
            status = async_logger.get_status()
            stats = status["stats"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("キュー", status["queue_size"])
                st.metric("処理済み", stats["processed"])
            
            with col2:
                worker_status = "🟢 動作中" if status["worker_alive"] else "🔴 停止"
                st.markdown(f"**ワーカー**: {worker_status}")
                st.metric("エラー", stats["errors"])
            
            if stats["last_error_msg"]:
                st.error(f"最新エラー: {stats['last_error_msg']}")
            
            if st.button("🔄 ログステータス更新"):
                st.rerun()
                
            # デバッグ用の強制再起動ボタン
            if st.button("🛑 ログワーカー再起動", type="secondary"):
                async_logger.force_shutdown()
                if "async_logger" in st.session_state:
                    del st.session_state.async_logger
                st.success("ログワーカーを再起動しました")
                st.rerun()
                
        except Exception as e:
            st.error(f"ログステータス取得失敗: {e}")

    # =====  プロンプト編集画面  =================================================
    if st.session_state.edit_target:
        mode_name = st.session_state.edit_target

        st.title(f"✏️ プロンプト編集: {mode_name}")

        with st.form(key=f"prompt_edit_form_{mode_name}"):
            prompt_text = st.text_area(
                "プロンプトを編集してください",
                value=st.session_state.prompts[mode_name],
                height=400
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                save_button = st.form_submit_button(label="✅ 保存")
            with col2:
                reset_button = st.form_submit_button(label="🔄 デフォルトに戻す")
            with col3:
                cancel_button = st.form_submit_button(label="❌ キャンセル")

        if save_button:
            handle_save_prompt(mode_name, prompt_text)
        elif reset_button:
            handle_reset_prompt(mode_name)
        elif cancel_button:
            handle_cancel_edit()

    # =====  メイン画面表示  ==========================================================
    if not st.session_state.edit_target:
        st.title("💬 GPT + RAG チャットボット")
        st.subheader(f"🗣️ {st.session_state.current_chat}")
        st.markdown(f"**モデル:** {st.session_state.gpt_model} | **モード:** {st.session_state.design_mode}")

        # -- メッセージ表示 --
        st.markdown('<div class="chat-body">', unsafe_allow_html=True)
        for m in get_messages():
            message_class = "user-message" if m["role"] == "user" else "assistant-message"
            with st.chat_message(m["role"]):
                st.markdown(f'<div class="{message_class}">{m["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # -- 入力欄 --
        user_prompt = st.chat_input("メッセージを入力…")

        # ===== RAG検索結果の表示（シンプル版） =====
        if st.session_state.get("last_rag_sources"):
            
            with st.expander("🔎 RAG検索結果を表示"):
                sources = st.session_state.last_rag_sources
                st.markdown(f"**検索チャンク数:** {len(sources)} 件")
                
                # セレクトボックスでチャンクを選択
                if sources:
                    chunk_options = []
                    text_sources = []
                    
                    for idx, source in enumerate(sources):
                        meta = source.get("metadata", {})
                        kind = meta.get("kind", "text")
                        
                        if kind in ("text", "table"):
                            source_name = meta.get("source", "N/A")
                            page_num = meta.get("page", "N/A")
                            distance = source.get("distance", 0)
                            similarity = 1 - distance
                            
                            chunk_options.append(f"チャンク {len(text_sources)+1}: {source_name} (p.{page_num}) | 類似度: {similarity:.3f}")
                            text_sources.append(source)
                    
                    if chunk_options:
                        selected_chunk = st.selectbox(
                            "表示するチャンクを選択:",
                            options=range(len(chunk_options)),
                            format_func=lambda x: chunk_options[x],
                            key="chunk_selector"
                        )
                        
                        # 選択されたチャンクの詳細表示
                        if selected_chunk is not None and selected_chunk < len(text_sources):
                            source = text_sources[selected_chunk]
                            meta = source.get("metadata", {})
                            content = source.get("content", "")
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown("**内容:**")
                                st.text_area(
                                    label="",
                                    value=content,
                                    height=200,
                                    disabled=True,
                                    key=f"selected_content_{selected_chunk}"
                                )
                            
                            with col2:
                                st.markdown("**詳細情報:**")
                                st.markdown(f"**種類:** {meta.get('kind', 'N/A')}")
                                st.markdown(f"**ソース:** {meta.get('source', 'N/A')}")
                                st.markdown(f"**ページ:** {meta.get('page', 'N/A')}")
                                st.markdown(f"**距離:** {source.get('distance', 0):.4f}")
                    else:
                        st.info("📄 テキスト・表データはありませんでした")
                
                # 画像情報（あれば簡単に表示）
                images = st.session_state.get("last_rag_images", [])
                if images:
                    st.markdown("---")
                    st.markdown(f"**関連画像:** {len(images)} 件")
                    
                    # 画像選択
                    if len(images) > 0:
                        image_options = [f"{img['name']} (ページ {img.get('page', 'N/A')})" for img in images]
                        selected_image = st.selectbox(
                            "表示する画像を選択:",
                            options=range(len(image_options)),
                            format_func=lambda x: image_options[x],
                            key="image_selector"
                        )
                        
                        if selected_image is not None and selected_image < len(images):
                            img_info = images[selected_image]
                            st.image(img_info['data'], caption=img_info['name'], width=400)
                
                # クリアボタン
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ RAG結果をクリア"):
                        st.session_state.last_rag_sources = []
                        st.session_state.last_rag_images = []
                        st.rerun()
                with col2:
                    st.markdown(f"*RAG使用: {'✅' if st.session_state.get('use_rag', False) else '❌'}*")

    else:
        # プロンプト編集モード時は入力欄を無効化
        user_prompt = None

    # =====  応答生成  ============================================================
    if user_prompt and not st.session_state.edit_target:  # 編集モード時は応答生成をスキップ
        # メッセージリストに現在の質問を追加
        msgs = get_messages()
        msgs.append({"role": "user", "content": user_prompt})

        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{user_prompt}</div>', unsafe_allow_html=True)

        # シンプルなステータス表示 - 折りたたみなし
        with st.status(f"🤖 {st.session_state.gpt_model} で回答を生成中...", expanded=True) as status:
            # プロンプト取得
            prompt = st.session_state.prompts[st.session_state.design_mode]

            logger.info("💬 gen_start — mode=%s model=%s use_rag=%s sid=%s",
                st.session_state.design_mode,
                st.session_state.gpt_model,
                st.session_state.get("use_rag", True),
                st.session_state.sid)

            try:
                # ---------- RAG あり ----------
                if st.session_state.get("use_rag", True):
                    st.session_state["last_answer_mode"] = "RAG"

                    t_api = time.perf_counter()
                    
                    # API呼び出しパラメータを準備
                    rag_params = {
                        "prompt": prompt,
                        "question": user_prompt,
                        "collection": st.session_state.rag_collection,
                        "rag_files": st.session_state.rag_files,
                        "top_k": 4,
                        "model": st.session_state.gpt_model,
                        "chat_history": msgs,
                    }
                    
                    # カスタム設定があれば追加
                    if st.session_state.get("temperature") != 0.0:
                        rag_params["temperature"] = st.session_state.temperature
                    if st.session_state.get("max_tokens") is not None:
                        rag_params["max_tokens"] = st.session_state.max_tokens
                    
                    # generate_answerを呼び出し
                    rag_res = generate_answer(**rag_params)
                    
                    api_elapsed = time.perf_counter() - t_api
                    assistant_reply = rag_res["answer"]
                    sources = rag_res["sources"]
                    
                    # 🔥 RAG結果をセッション状態に保存（rerunで消えないように）
                    st.session_state.last_rag_sources = sources
                    # 画像情報も保存（generate_answer関数から返されるようにする必要がある場合）
                    st.session_state.last_rag_images = rag_res.get("images", [])

                    logger.info("💬 GPT done — tokens≈%d  api_elapsed=%.2fs  sources=%d",
                                len(assistant_reply.split()), api_elapsed, len(sources))

                # ---------- GPT-only ----------
                else:
                    st.session_state["last_answer_mode"] = "GPT-only"
                    # API呼び出し部分（Azure用に修正）
                    params = {
                        "model": get_azure_model_name(st.session_state.gpt_model),  # Azure用に変換
                        "messages": [
                            {"role": "system", "content": prompt},
                            *msgs[:-1],
                            {"role": "user", "content": user_prompt},
                        ]
                    }

                    # カスタム設定があれば追加
                    if st.session_state.get("temperature") != 1.0:
                        params["temperature"] = st.session_state.temperature
                    if st.session_state.get("max_tokens") is not None:
                        params["max_tokens"] = st.session_state.max_tokens

                    import time
                    t_api = time.perf_counter()

                    # APIを呼び出し
                    resp = client.chat.completions.create(**params)

                    api_elapsed = time.perf_counter() - t_api

                    assistant_reply = resp.choices[0].message.content
                    sources = []

                    logger.info("💬 GPT done — tokens≈%d  api_elapsed=%.2fs",
                                    len(assistant_reply.split()), api_elapsed)
                    
            except Exception as e:
                logger.exception("❌ answer_gen failed — %s", e)
                st.error("回答生成時にエラーが発生しました")

            # ---------- 画面反映 ----------
            with st.chat_message("assistant"):
                # モデル情報を応答に追加
                model_info = f"\n\n---\n*このレスポンスは `{st.session_state.gpt_model}` で生成されました*"
                full_reply = assistant_reply + model_info
                st.markdown(full_reply)

            # 保存するのは元の応答（モデル情報なし）
            msgs.append({"role": "assistant", "content": assistant_reply})
            # ★ 重要：ログ保存を先に実行
            logger.info("📝 Executing post_log before any other operations")
            post_log_async(user_prompt, assistant_reply, prompt, send_to_model_comparison=True)

            # ★ チャットタイトル生成は後回し（ログ保存完了後）
            try:
                new_title = generate_chat_title(msgs)
                if new_title and new_title != st.session_state.current_chat:
                    old_title = st.session_state.current_chat
                    st.session_state.chats[new_title] = st.session_state.chats[old_title]
                    del st.session_state.chats[old_title]
                    st.session_state.current_chat = new_title
                    logger.info("📝 Chat title updated: %s -> %s", old_title, new_title)
            except Exception as e:
                logger.warning("⚠️ Chat title generation failed (non-critical): %s", e)

            st.rerun()

elif st.session_state["authentication_status"] is False:
    st.error('ユーザー名またはパスワードが間違っています。')
elif st.session_state["authentication_status"] is None:
    st.warning("ユーザー名とパスワードを入力してください。")
    st.stop()