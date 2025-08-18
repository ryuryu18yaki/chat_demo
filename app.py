
import streamlit as st
from typing import List, Dict, Any
import time

from src.startup_loader import initialize_equipment_data, get_available_buildings, get_building_info_for_prompt
from src.logging_utils import init_logger
from src.sheets_manager import log_to_sheets, get_sheets_manager, send_prompt_to_model_comparison
from src.langchain_chains import generate_smart_answer_with_langchain, generate_chat_title_with_llm
from src.building_manager import get_building_manager
from src.firestore_manager import log_to_firestore, send_prompt_to_firestore_comparison

import yaml
import streamlit_authenticator as stauth
import uuid

# === Chat Store (SID 主キー) 基盤 ===
import unicodedata as _ud
import re

def _sanitize_title(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    t = _ud.normalize("NFC", s).strip()
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    # 両端のカギや引用符を剥がす（LLMの癖対策）
    if (t.startswith("「") and t.endswith("」")) or (t.startswith("『") and t.endswith("』")):
        t = t[1:-1].strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return t[:60] or "Chat"

# === 🔥 改良版：統合されたタイトル更新システム ===

# 🔥 修正版: 統合されたタイトル更新システム

def update_chat_title_safely(new_title: str, force_rerun: bool = True) -> bool:
    """
    タイトル更新を安全に実行し、すべての関連状態を同期する統合関数
    
    Args:
        new_title: 新しいタイトル
        force_rerun: 更新後に強制rerunするか
        
    Returns:
        bool: 更新が成功したかどうか
    """
    # 🚨 デバッグ用ログを追加
    logger.info(f"🚨 update_chat_title_safely CALLED - new_title='{new_title}', force_rerun={force_rerun}")
    
    try:
        # 1. タイトル正規化
        sanitized_title = _sanitize_title(new_title)
        logger.info(f"🔧 Title sanitized: '{new_title}' -> '{sanitized_title}'")
        
        if not sanitized_title or len(sanitized_title.strip()) == 0:
            logger.warning("⚠️ Invalid title after sanitization")
            return False
            
        # 2. 現在の状態取得
        s = st.session_state.chat_store
        sid = s["current_sid"]
        old_title = s["by_id"][sid]["title"]
        
        logger.info(f"📊 Current state - sid={sid}, old_title='{old_title}'")
        
        if sanitized_title == old_title:
            logger.info("🔍 Title unchanged, skipping update")
            return False
            
        # 3. 重複回避処理
        existing_titles = {row["title"] for row in s["by_id"].values() if row != s["by_id"][sid]}
        final_title = sanitized_title
        counter = 2
        
        while final_title in existing_titles:
            final_title = f"{sanitized_title} ({counter})"
            counter += 1
            
        logger.info(f"🎯 Title update: '{old_title}' -> '{final_title}'")
        
        # 4. chat_store の更新
        logger.info("📄 Updating chat_store...")
        s["by_id"][sid]["title"] = final_title
        
        # 🔥 デバッグ: 更新確認
        logger.info(f"📄 chat_store updated - by_id[{sid}]['title'] = '{s['by_id'][sid]['title']}'")
        
        # 5. 🔥 即座にミラー状態を更新（ensure_chat_store を呼ばずに直接更新）
        logger.info("📄 Updating mirror states...")
        by_id, order, current_sid = s["by_id"], s["order"], s["current_sid"]
        
        # 🔥 デバッグ: 更新前の状態
        logger.info(f"📄 Before mirror update - current_chat='{st.session_state.get('current_chat', 'NONE')}'")
        
        # chat_sids と chats を直接再構築
        new_chat_sids = {by_id[_sid]["title"]: _sid for _sid in order}
        new_chats = {by_id[_sid]["title"]: by_id[_sid]["messages"] for _sid in order}
        new_current_title = by_id[current_sid]["title"]
        
        logger.info(f"📄 Mirror update - new_chat_sids_keys={list(new_chat_sids.keys())}")
        logger.info(f"📄 Mirror update - new_current_title='{new_current_title}'")
        
        # session_state を原子的に更新
        st.session_state.chat_sids = new_chat_sids
        st.session_state.chats = new_chats
        st.session_state.current_chat = new_current_title
        st.session_state.sid = current_sid
        
        # 🔥 デバッグ: 更新後の状態
        logger.info(f"📄 After mirror update - current_chat='{st.session_state.current_chat}'")
        logger.info(f"📄 After mirror update - chat_sids keys={list(st.session_state.chat_sids.keys())}")
        
        logger.info("✅ Title update completed - new_title=%r, chat_sids_keys=%s", 
                   final_title, list(new_chat_sids.keys()))
        
        # 6. 🔥 タイトル更新フラグを設定（rerun後にリセットされるように）
        st.session_state["_title_update_pending"] = True
        logger.info("🔥 Set _title_update_pending = True")
        
        # 7. 強制rerun（必要な場合）
        if force_rerun:
            logger.info("🚀 Preparing to call st.rerun()...")
            try:
                # 🔥 rerun前の最終確認
                logger.info(f"🚀 Final state before rerun - current_chat='{st.session_state.current_chat}'")
                logger.info(f"🚀 Final state before rerun - chat_store title='{s['by_id'][sid]['title']}'")
                
                st.rerun()
                
                # この行は実行されないはず（rerunで処理が中断するため）
                logger.error("❌ This should not be logged - st.rerun() failed to stop execution")
                
            except Exception as rerun_error:
                logger.error(f"💥 st.rerun() failed: {rerun_error}", exc_info=True)
                return False
            
        return True
        
    except Exception as e:
        logger.error(f"💥 Title update failed: {e}", exc_info=True)
        return False


def ensure_chat_store():
    """
    chat_store を初期化またはミラー状態を同期
    🔥 改良版：タイトル更新処理との競合を回避
    """
    ss = st.session_state
    
    # タイトル更新中の場合は、ミラー同期をスキップ
    if ss.get("_title_update_pending"):
        logger.info("📄 Skipping chat_store sync during title update")
        # 🔥 フラグをリセットしない（rerun後の最初の呼び出しでリセット）
        return
    
    # 🔥 rerun後の最初の呼び出しでフラグをリセット
    if "_title_update_pending" in ss:
        logger.info("🔥 Resetting _title_update_pending flag after rerun")
        del ss["_title_update_pending"]
    
    if "chat_store" not in ss:
        # 初期化処理（既存と同じ）
        logger.info("🔥 Initializing new chat_store")
        by_id, order, current_sid = {}, [], None

        if "chat_sids" in ss and "chats" in ss and ss["chat_sids"]:
            # 旧構造からマイグレーション
            for title, sid in ss["chat_sids"].items():
                by_id[sid] = {"title": _sanitize_title(title),
                              "messages": ss.get("chats", {}).get(title, [])}
                order.append(sid)
            
            cur_title = ss.get("current_chat") or "Chat 1"
            current_sid = next((sid for t, sid in ss["chat_sids"].items()
                                if _sanitize_title(t) == _sanitize_title(cur_title)),
                               (order[0] if order else None))
        else:
            # 新規作成
            import uuid
            sid = str(uuid.uuid4())
            by_id[sid] = {"title": "Chat 1", "messages": []}
            order = [sid]
            current_sid = sid

        ss.chat_store = {"by_id": by_id, "order": order, "current_sid": current_sid}

    # 🔥 ミラー同期（改良版）
    s = ss.chat_store
    by_id, order, current_sid = s["by_id"], s["order"], s["current_sid"]

    # より安全なミラー再生成
    try:
        # 🔥 デバッグ: 同期前の状態
        logger.info(f"🔄 Before sync - current_chat='{ss.get('current_chat', 'NONE')}'")
        
        chat_sids = {by_id[sid]["title"]: sid for sid in order if sid in by_id}
        chats = {by_id[sid]["title"]: by_id[sid]["messages"] for sid in order if sid in by_id}
        current_title = by_id[current_sid]["title"] if current_sid in by_id else "Chat 1"

        ss.chat_sids = chat_sids
        ss.chats = chats
        ss.current_chat = current_title
        ss.sid = current_sid

        # 🔥 デバッグ: 同期後の状態
        logger.info(f"🔄 After sync - current_chat='{current_title}'")
        logger.info("🧱 chat_store synced - current_sid=%s title=%r titles=%s",
                    current_sid, current_title, list(chat_sids.keys()))
                    
    except KeyError as e:
        logger.error(f"❌ chat_store sync failed: {e}", exc_info=True)
        # フォールバック：chat_store を削除して次回に再初期化
        if "chat_store" in ss:
            del ss["chat_store"]

import threading
import queue

st.set_page_config(page_title="Claude + RAG Chatbot", page_icon="💬", layout="wide")

logger = init_logger()

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

# ===== post_log関数（変更なし） =====
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
                    claude_model = user_info.get("claude_model", "unknown")
                    temperature = user_info.get("temperature", 0.0)
                    max_tokens = user_info.get("max_tokens")
                    use_rag = user_info.get("use_rag", False)
                    chat_title = user_info.get("chat_title", "未設定")
                else:
                    # フォールバック値
                    username = design_mode = session_id = claude_model = "unknown"
                    temperature = 0.0
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
                    model=claude_model,
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

# 🔥 新しいFirestore用の非同期ログ関数を追加（既存のpost_log_asyncは変更しない）
def post_log_firestore_async(input_text: str, output_text: str, prompt: str, 
                             send_to_model_comparison: bool = False):
    """Firestore専用の非同期ログ投稿関数"""
    try:
        logger.info("🔥 Firestore logging start...")
        
        # セッション状態から必要な情報を取得
        username = st.session_state.get("username") or st.session_state.get("name")
        design_mode = st.session_state.get("design_mode")
        session_id = st.session_state.get("sid")
        claude_model = st.session_state.get("claude_model")
        temperature = st.session_state.get("temperature", 0.0)
        max_tokens = st.session_state.get("max_tokens")
        use_rag = st.session_state.get("use_rag", False)
        chat_title = st.session_state.get("current_chat", "未設定")
        
        logger.info(f"🔥 Session data - user: {username}, mode: {design_mode}, model: {claude_model}")
        
        # メタデータ準備
        metadata = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_rag": use_rag,
            "app_version": "2.0",
            "log_source": "firestore",
            "timestamp": time.time()
        }
        
        # 🔥 Firestoreに会話ログを保存
        firestore_success = log_to_firestore(
            input_text=input_text,
            output_text=output_text,
            prompt=prompt,
            chat_title=chat_title,
            user_id=username or "unknown",
            session_id=session_id or "unknown",
            mode=design_mode or "unknown",
            model=claude_model or "unknown",
            temperature=temperature,
            max_tokens=max_tokens,
            use_rag=use_rag
        )
        
        if firestore_success:
            logger.info("✅ Firestore conversation log saved")
        else:
            logger.warning("⚠️ Firestore conversation log failed")
        
        # 🔥 モデル比較への送信（オプション）
        if send_to_model_comparison:
            try:
                # チャットメッセージを取得
                current_chat = st.session_state.get("current_chat", "New Chat")
                chats_dict = st.session_state.get("chats", {})
                msgs = chats_dict.get(current_chat, [])
                
                if msgs:
                    # 完全なプロンプトを構築
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
                    
                    # モデル比較に送信
                    model_success = send_prompt_to_firestore_comparison(
                        prompt_text=comparison_prompt,
                        user_note=f"User: {username}, Mode: {design_mode}, Model: {claude_model}"
                    )
                    
                    if model_success:
                        logger.info("✅ Firestore model comparison saved")
                    else:
                        logger.warning("⚠️ Firestore model comparison failed")
                        
            except Exception as comparison_error:
                logger.error(f"❌ Firestore model comparison save failed: {comparison_error}")
        
        logger.info("🔥 Firestore logging completed")
        return firestore_success
        
    except Exception as e:
        logger.error(f"❌ Firestore logging failed: {e}")
        return False

# ===== StreamlitAsyncLogger（変更なし） =====
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
        claude_model = st.session_state.get("claude_model")
        temperature = st.session_state.get("temperature", 0.0)
        max_tokens = st.session_state.get("max_tokens")
        use_rag = st.session_state.get("use_rag", False)
        chat_title = st.session_state.get("current_chat", "未設定")
        
        # デバッグログ
        logger.info("🔍 Session state values — username=%s design_mode=%s claude_model=%s", 
                    username, design_mode, claude_model)
        
        user_info = {
            "username": username or "unknown",
            "design_mode": design_mode or "unknown",
            "session_id": session_id or "unknown",
            "claude_model": claude_model or "unknown",
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

# =====  ログインUIの表示  ============================================================
# 🔥 ログイン状態をチェックしてから login() を呼ぶ
if st.session_state.get("authentication_status") is None:
    authenticator.login()
elif st.session_state.get("authentication_status") is False:
    authenticator.login()
else:
    # 既にログイン済みの場合はlogin()を呼ばない
    pass

if st.session_state["authentication_status"]:
    name = st.session_state["name"]
    username = st.session_state["username"]

    logger.info("🔐 login success — user=%s  username=%s", name, username)
    logger.info("🧭 STATE@ENTRY — current=%r keys=%s", st.session_state.get("current_chat"), list(st.session_state.get("chat_sids", {}).keys()))

    # 設備データを input_data から自動初期化
    # 設備データ初期化
    if st.session_state.get("equipment_data") is None:
        logger.info("🔍🔍🔍 設備データ初期化開始")
        
        try:
            logger.info("🔍🔍🔍 try ブロック開始")
            
            # Google DriveフォルダIDが設定されているかチェック
            drive_folder_id = None
            try:
                logger.info("🔍🔍🔍 secrets取得試行")
                drive_folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
                logger.info("🔍🔍🔍 取得結果: '%s'", drive_folder_id)
                if drive_folder_id:
                    drive_folder_id = drive_folder_id.strip()  # 前後の空白を除去
                    logger.info("🔍🔍🔍 strip後: '%s'", drive_folder_id)
            except Exception as secrets_error:
                logger.error("🔍🔍🔍 secrets取得エラー: %s", secrets_error)
            
            # 初期化実行
            if drive_folder_id:
                logger.info("🔍🔍🔍 Google Driveモード選択")
                # Google Driveから読み込み
                st.info("📁 Google Driveからファイルを読み込み中...")
                
                param = f"gdrive:{drive_folder_id}"
                logger.info("🔍🔍🔍 呼び出しパラメータ: '%s'", param)
                logger.info("🔍🔍🔍 initialize_equipment_data 呼び出し直前")
                
                res = initialize_equipment_data(param)
                
                logger.info("🔍🔍🔍 initialize_equipment_data 呼び出し完了")
                logger.info("📂 Google Driveから設備データ初期化完了")
            else:
                logger.info("🔍🔍🔍 ローカルモード選択")
                # ローカルから読み込み（既存処理）
                st.info("📂 ローカル rag_data フォルダからファイルを読み込み中...")
                logger.info("🔍🔍🔍 initialize_equipment_data 呼び出し直前（ローカル）")
                
                res = initialize_equipment_data("rag_data")
                
                logger.info("🔍🔍🔍 initialize_equipment_data 呼び出し完了（ローカル）")
                logger.info("📂 ローカルディレクトリから設備データ初期化完了")
            
            logger.info("🔍🔍🔍 結果処理開始")
            st.session_state.equipment_data = res["equipment_data"]
            st.session_state.equipment_list = res["equipment_list"]
            st.session_state.category_list = res["category_list"]
            st.session_state.rag_files = res["file_list"]
            logger.info("🔍🔍🔍 セッション状態更新完了")

            logger.info("📂 設備データ初期化完了 — 設備数=%d  ファイル数=%d",
                    len(res["equipment_list"]), len(res["file_list"]))
            
        except Exception as e:
            logger.error("🔍🔍🔍 メイン例外キャッチ: %s", e, exc_info=True)
            logger.exception("❌ 設備データ初期化失敗 — %s", e)
            st.error(f"設備データ初期化中にエラーが発生しました: {e}")
    else:
        logger.info("🔍🔍🔍 設備データは既に初期化済み")

    # --------------------------------------------------------------------------- #
    #                         ★ 各モード専用プロンプト ★                           #
    # --------------------------------------------------------------------------- #
    DEFAULT_PROMPTS: Dict[str, str] = {
        "暗黙知法令チャットモード": """
    あなたは建築電気設備設計のエキスパートエンジニアです。
    今回の対象は **複合用途ビルのオフィス入居工事（B工事）** に限定されます。
    以下の知識と技術をもとに、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。
    専門用語は必要に応じて解説を加え、判断の背景にある理由を丁寧に説明します。
    ────────────────────────────────
    ## 【回答方針】
    **重要：以下の各事項は「代表的なビル（丸の内ビルディング）」を想定して記載しています。他のビルでは仕様や基準が異なる可能性があることを、回答時には必ず言及してください。**
    **注意：過度に込み入った条件の詳細説明をユーザーに求めることは避け、一般的な設計基準に基づく実務的な回答を心がけてください。**
    ### ■ 暗黙知情報不足時の対応プロセス
    現在保有している暗黙知情報では適切な回答ができない場合は、以下の手順で対応してください：
    1. **現状把握の明示**
    - 「現在の暗黙知情報では、○○ビルの△△設備について十分な情報がございません」と明確に伝える
    - 一般的な設計基準に基づく暫定的な回答がある場合は、その旨を明記して提供する
    2. **逆質問の実行**
    - ユーザーの実務経験や現場知識を活用するため、具体的な逆質問を行う
    - 質問例：「○○ビルでは△△設備についてどのような仕様・基準をお使いでしょうか？」
    - 「過去の類似案件では、どのような対応をされましたか？」
    3. **暗黙知情報の記録**
    - ユーザーから有効な回答が得られた場合、以下のフォーマットで情報を記録する：
    ```
    【暗黙知情報：ビル名、設備名】
    内容：（ユーザーから得られた情報の要約）
    適用条件：（どのような条件下で適用されるか）
    ```
    4. **情報活用とフィードバック**
    - 得られた暗黙知情報を元に、改めて適切な回答を提供する
    - 「この情報は今後の設計業務改善に活用させていただきます」と感謝の意を示す
    ────────────────────────────────
    ## 【工事区分について】
    - **B工事**：本システムが対象とする工事。入居者負担でビル側が施工する工事
    - **C工事**：入居者が独自に施工する工事（電話・LAN・防犯設備など）
    - 本システムでは、C工事設備については配管類の数量算出のみを行います
    ────────────────────────────────
    ## 【消防署事前相談の指針】
    ### ■ 事前相談が必要な状況
    法令のルールが競合する場合や細かな仕様で判断が分かれる場合は、**必ず消防署への事前相談を行う**ことを推奨してください。
    ### ■ 事前相談のタイミング
    - **着工届出書提出時**：通常の手続きの中で相談
    - **軽微な工事で着工届が不要な場合**：別途消防署に出向いて相談
    ### ■ 法令競合の典型例
    1. **自火報（煙感知器）関連**
    - 狭い部屋内で「吹き出しから離して設置」「吸込口付近に設置」「入口付近に設置」を同時に満たす場所がない場合
    ### ■ 細かな仕様判断の典型例
    1. **自火報（煙感知器）関連**
    - 欄間オープン内に侵入防止バーがあって面積が阻害されている場合
    - 阻害された面積分を補うように欄間オープンの面積を広げている場合の扱い
    2. **避難口誘導灯関連**
    - 扉の直上扱いとして矢印シンボルなしを設置してよい範囲（扉周辺3m程度が目安だが、最終的には担当者判断）
    - パーテーション等による視認阻害の程度と補完誘導灯設置の要否
    ────────────────────────────────
    ## 【重要な注意事項】
    1. **ビル仕様の違い**：上記の内容は丸の内ビルディングを基準としています。他のビルでは異なる仕様・基準が適用される可能性があります。
    2. **過度な詳細要求の回避**：ユーザーに対して、込み入った条件の詳細説明を過度に求めることは避けてください。
    3. **工事区分の明確化**：B工事とC工事の区分を常に意識し、C工事設備については配管類のみを扱うことを明確にしてください。
    4. **法令準拠**：検索結果の言い回しをそのまま複製することを避け、直接引用以外のすべてを自分の言葉で表現します。
    5. **判断困難時の対応**：法令競合や細かな仕様判断で迷いが生じた場合は、必ず消防署への事前相談を推奨し、一般的な傾向は示しつつも最終判断は消防署見解に委ねることを明記してください。
    6. **暗黙知収集**：現在の知識で対応できない質問については、積極的にユーザーの実務経験を活用し、将来の暗黙知データベース拡充に貢献してください。
    7. **資料からの原文抜粋の禁止**：ユーザーから提供された資料や図面からの原文抜粋は行わず、必ず自分の言葉で説明してください。
    ────────────────────────────────
    ## 【文字数制限と回答作成プロセス】
    ### 回答は1500文字以内で作成してください
    
    以下のPythonコードを参考に文字数を意識して回答を作成してください：
    
    ```python
    def validate_answer_length(answer, max_chars=1500):
        char_count = len(answer)
        
        print(f"文字数チェック結果:")
        print(f"- 現在の文字数: {{{{char_count}}}}")
        print(f"- 制限文字数: {{{{max_chars}}}}")
        
        if char_count > max_chars:
            excess = char_count - max_chars
            print(f"- 超過文字数: {{{{excess}}}}")
            print("⚠️ 文字数制限を超過しています")
            print("→ 以下の方針で要約してください：")
            print("  1. 重要でない詳細を削除")
            print("  2. 冗長な表現を簡潔に")
            print("  3. 例示を減らす")
            return False
        else:
            print("✅ 文字数制限内です")
            return True
    ```
    
    **重要な指示:**
    1. 回答作成時に文字数を意識する
    2. 冗長な表現を避ける  
    3. 要点を簡潔にまとめる
    4. 回答末尾に「（回答文字数：XXX文字）」を必ず記載
    
    最大1500文字以内で、簡潔かつ的確な回答を心がけてください。
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
    """,

    "ビルマスタ質問モード": """
    あなたは建築電気設備設計のエキスパートエンジニアです。
    今回の対象は **複合用途ビルのオフィス入居工事（B工事）** に限定されます。
    提供されたビルマスターデータを参照して、ユーザーの質問に正確に回答してください。

    【回答方針】
    1. **正確性を最優先**: ビルマスターデータに記載されている情報のみを使用してください  
    2. **複数ビルの比較**: 複数のビルについて質問された場合は、各ビルの情報を比較して回答してください  
    3. **情報の出典明示**: 回答する際は、どのビルの情報を参照しているかを明確にしてください  
    4. **データ不足時の対応**: 要求された情報がデータにない場合は、「情報が記載されていません」と明記し、必要情報を逆質問してください  

    【似ているビルの判定方法】  
    ビル同士を参考にする必要がある場合は、以下の順で近いものを選んでください。  
    1. 用途区分（消防）が同じ  
    2. オーナー  
        -  三菱であるか
        -  それ以外か(それ以外の場合でも三菱系のビルを参考とする)
    3. 竣工年月が近い  
    4. 延床面積が近い  
    5. 所在地が近い  
    該当するビルが複数存在する場合は、回答ビル全てを参考にしてください。

    【注意事項】
    - ビルマスターデータにない情報は推測しない  
    - 設備仕様や設計基準はデータに記載されている範囲のみを使用  

    【回答形式】
    - 簡潔でわかりやすい日本語  
    - 必要に応じて箇条書きや表形式  
    - ビル名は正式名称で記載
    """
    }

    # =====  セッション変数  =======================================================
    ensure_chat_store()
    if "edit_target" not in st.session_state:
        st.session_state.edit_target = None
    if "rag_files" not in st.session_state:
        st.session_state.rag_files: List[Dict[str, Any]] = []
    if "design_mode" not in st.session_state:
        st.session_state.design_mode = list(DEFAULT_PROMPTS.keys())[0]
    if "prompts" not in st.session_state:
        st.session_state.prompts = DEFAULT_PROMPTS.copy()
    if "claude_model" not in st.session_state:
        st.session_state.claude_model = "claude-4-sonnet"
    if "selected_equipment" not in st.session_state:
        st.session_state.selected_equipment = None
    if "selection_mode" not in st.session_state:
        st.session_state.selection_mode = "manual"
    
    user_prompt: str | None = None

    # =====  ヘルパー  ============================================================
    def get_messages() -> List[Dict[str, str]]:
        s = st.session_state.chat_store
        return s["by_id"][s["current_sid"]]["messages"]

    def new_chat():
        import uuid
        s = st.session_state.chat_store
        sid = str(uuid.uuid4())
        idx = len(s["by_id"]) + 1
        s["by_id"][sid] = {"title": f"Chat {idx}", "messages": []}
        s["order"].insert(0, sid)     # 新しいものを先頭に（任意）
        s["current_sid"] = sid
        ensure_chat_store()           # ミラー再生成
        logger.info("➕ new_chat — sid=%s  title='%s'", sid, st.session_state.current_chat)
        st.rerun()

    def switch_chat(title: str):
        """タイトルからSIDを引いて切替（互換用）"""
        sid = st.session_state.chat_sids.get(title)
        if not sid:
            # 念のためタイトル探索
            for _sid, row in st.session_state.chat_store["by_id"].items():
                if row["title"] == title:
                    sid = _sid; break
        if not sid:
            logger.warning("⚠️ switch_chat: title %r not found", title)
            return
        st.session_state.chat_store["current_sid"] = sid
        ensure_chat_store()
        logger.info("🔀 switch_chat — sid=%s  title='%s'", sid, st.session_state.current_chat)
        st.rerun()
    
    # =====  データ準備関数（新規追加）  ===============================================
    def prepare_prompt_data():
        """セッション状態から選択されたデータを取得してLangChain用に準備"""
        current_mode = st.session_state.design_mode
        
        equipment_content = None
        building_content = None
        target_building_content = None  # 🔥 新規追加
        other_buildings_content = None  # 🔥 新規追加
        
        # 設備資料の取得（暗黙知モードのみ）
        if current_mode == "暗黙知法令チャットモード":
            selected_equipment = st.session_state.get("selected_equipment")
            if selected_equipment:
                selected_files_key = f"selected_files_{selected_equipment}"
                selected_files = st.session_state.get(selected_files_key, [])
                
                if selected_files:
                    equipment_texts = []
                    equipment_data = st.session_state.equipment_data
                    
                    for file_name in selected_files:
                        if file_name in equipment_data[selected_equipment]["files"]:
                            file_text = equipment_data[selected_equipment]["files"][file_name]
                            equipment_texts.append(file_text)
                    
                    if equipment_texts:
                        equipment_content = "\n\n".join(equipment_texts)
        
        # 🔥 修正: ビル情報の取得（新しいbuilding_mode対応）
        if current_mode in ["暗黙知法令チャットモード", "ビルマスタ質問モード"]:
            include_building = st.session_state.get("include_building_info", False)
            
            # ビルマスタモードは常にビル情報を使用、暗黙知モードはチェックボックス次第
            if (current_mode == "ビルマスタ質問モード") or \
            (current_mode == "暗黙知法令チャットモード" and include_building):
                
                building_mode = st.session_state.get("building_mode", "none")
                selected_building = st.session_state.get("selected_building")
                
                try:
                    building_manager = get_building_manager()
                    if building_manager and building_manager.available:
                        
                        if building_mode == "specific_only" and selected_building:
                            # 特定ビルのみ（従来の動作）
                            building_content = building_manager.format_building_info_for_prompt(selected_building)
                            target_building_content = building_content
                            other_buildings_content = None
                            
                        elif building_mode == "specific_with_others" and selected_building:
                            # 🔥 新機能: 特定ビル + 他のビル
                            target_building_content = building_manager.format_building_info_for_prompt(selected_building)
                            
                            # 他のビル情報を取得（選択したビル以外）
                            all_buildings = building_manager.get_building_list()
                            other_buildings = [b for b in all_buildings if b != selected_building]
                            
                            if other_buildings:
                                other_building_parts = []
                                for other_building in other_buildings:
                                    other_info = building_manager.format_building_info_for_prompt(other_building)
                                    other_building_parts.append(other_info)
                                other_buildings_content = "\n\n".join(other_building_parts)
                            else:
                                other_buildings_content = "他のビル情報はありません。"
                            
                            # 従来のbuilding_contentも設定（後方互換性のため）
                            building_content = target_building_content + "\n\n" + other_buildings_content
                            
                        elif building_mode == "all":
                            # 全ビル情報（従来の動作）
                            building_content = building_manager.format_building_info_for_prompt()
                            target_building_content = None
                            other_buildings_content = building_content
                            
                        elif building_mode in ["specific", "specific_only"]:
                            # 🔥 後方互換性: 既存のspecificモードを specific_only として処理
                            if selected_building:
                                building_content = building_manager.format_building_info_for_prompt(selected_building)
                                target_building_content = building_content
                                other_buildings_content = None
                            
                except Exception as e:
                    logger.warning(f"⚠️ ビル情報取得失敗: {e}")
        
        return {
            "mode": current_mode,
            "equipment_content": equipment_content,
            "building_content": building_content,  # 従来の統合版（後方互換性）
            "target_building_content": target_building_content,  # 🔥 新規: 対象ビル
            "other_buildings_content": other_buildings_content,   # 🔥 新規: その他ビル
        }
        
    # =====  編集機能用のヘルパー関数（変更なし）  ==============================================
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

    # =====  サイドバー用ヘルパー関数  ==================================================
    
    def render_equipment_selection():
        """設備選択UIを描画（共通関数）"""
        st.markdown("### 🔧 対象設備選択")
        
        available_equipment = st.session_state.get("equipment_list", [])
        available_categories = st.session_state.get("category_list", [])

        if not available_equipment:
            st.error("❌ 設備データが読み込まれていません")
            st.session_state["selected_equipment"] = None
            return

        st.info(f"📊 利用可能設備数: {len(available_equipment)}")
        
        # 設備選択方式
        selection_mode = st.radio(
            "選択方式",
            ["設備名で選択", "カテゴリから選択"],
            index=0,
            help="質問に使用する設備の選択方法"
        )
        
        if selection_mode == "設備名で選択":
            selected_equipment = st.selectbox(
                "設備を選択してください",
                options=[""] + available_equipment,
                index=0,
                help="この設備の資料のみを使用して回答を生成します"
            )
            st.session_state["selected_equipment"] = selected_equipment if selected_equipment else None
            st.session_state["selection_mode"] = "manual"
            
        elif selection_mode == "カテゴリから選択":
            selected_category = st.selectbox(
                "カテゴリを選択してください",
                options=[""] + available_categories,
                index=0
            )
            
            if selected_category:
                category_equipment = [
                    eq for eq in available_equipment 
                    if st.session_state.equipment_data[eq]["equipment_category"] == selected_category
                ]
                
                selected_equipment = st.selectbox(
                    f"「{selected_category}」内の設備を選択",
                    options=[""] + category_equipment,
                    index=0
                )
                st.session_state["selected_equipment"] = selected_equipment if selected_equipment else None
            else:
                st.session_state["selected_equipment"] = None
            st.session_state["selection_mode"] = "category"

    def render_file_selection(current_equipment):
        """ファイル選択UIを描画（共通関数）"""
        if not current_equipment:
            return
            
        eq_info = st.session_state.equipment_data[current_equipment]
        st.success(f"✅ 選択中: **{current_equipment}**")
        
        st.markdown("#### 📄 使用ファイル選択")
        available_files = eq_info['sources']
        
        # セッション状態でファイル選択を管理
        selected_files_key = f"selected_files_{current_equipment}"
        if selected_files_key not in st.session_state:
            st.session_state[selected_files_key] = available_files.copy()
        
        # ファイル選択UI
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 全選択", key=f"select_all_{current_equipment}"):
                st.session_state[selected_files_key] = available_files.copy()
                st.rerun()
        with col2:
            if st.button("❌ 全解除", key=f"deselect_all_{current_equipment}"):
                st.session_state[selected_files_key] = []
                st.rerun()
        
        # 各ファイルのチェックボックス
        for file in available_files:
            checked = st.checkbox(
                file,
                value=file in st.session_state[selected_files_key],
                key=f"file_{current_equipment}_{file}"
            )
            
            if checked and file not in st.session_state[selected_files_key]:
                st.session_state[selected_files_key].append(file)
            elif not checked and file in st.session_state[selected_files_key]:
                st.session_state[selected_files_key].remove(file)
        
        # 選択状況の表示
        selected_count = len(st.session_state[selected_files_key])
        total_count = len(available_files)
        
        if selected_count == 0:
            st.error("⚠️ ファイルが選択されていません")
        elif selected_count == total_count:
            st.info(f"📊 全ファイル使用: {selected_count}/{total_count}")
        else:
            st.info(f"📊 選択ファイル: {selected_count}/{total_count}")
        
        # 設備詳細（折りたたみ）
        with st.expander("📋 設備詳細", expanded=False):
            st.markdown(f"- **カテゴリ**: {eq_info['equipment_category']}")
            st.markdown(f"- **総ファイル数**: {eq_info['total_files']}")
            st.markdown(f"- **総ページ数**: {eq_info['total_pages']}")
            st.markdown(f"- **総文字数**: {eq_info['total_chars']:,}")
            
            if selected_count > 0:
                st.markdown("- **選択中のファイル**:")
                for file in st.session_state[selected_files_key]:
                    file_chars = len(eq_info['files'].get(file, ''))
                    st.markdown(f"  - ✅ {file} ({file_chars:,}文字)")
                
                if selected_count < total_count:
                    selected_chars = sum(len(eq_info['files'].get(f, '')) for f in st.session_state[selected_files_key])
                    char_ratio = 100 * selected_chars / eq_info['total_chars'] if eq_info['total_chars'] > 0 else 0
                    st.markdown(f"- **選択ファイル統計**:")
                    st.markdown(f"  - ファイル数: {selected_count}/{total_count} ({100*selected_count/total_count:.1f}%)")
                    st.markdown(f"  - 文字数: {selected_chars:,}/{eq_info['total_chars']:,} ({char_ratio:.1f}%)")

    def render_building_selection(expanded=False):
        """ビル選択UIを描画（共通関数）"""
        with st.expander("🏢 対象ビル選択", expanded=expanded):
            available_buildings = get_available_buildings()

            if not available_buildings:
                st.error("⚠️ ビル情報が読み込まれていません")
                st.session_state["selected_building"] = None
                st.session_state["include_building_info"] = False
                return

            st.info(f"📊 利用可能ビル数: {len(available_buildings)}")
            
            include_building = st.checkbox(
                "ビル情報をプロンプトに含める",
                value=st.session_state.get("include_building_info", False),
                help="チェックを入れると、選択されたビルの詳細情報が回答生成時に使用されます"
            )
            st.session_state["include_building_info"] = include_building
            
            if include_building:
                building_selection_mode = st.radio(
                    "ビル選択方式",
                    ["特定ビルを選択", "全ビル情報を使用"],
                    index=st.session_state.get("building_selection_mode_index", 0),
                    help="質問に使用するビル情報の選択方法"
                )

                mode_options = ["特定ビルを選択", "全ビル情報を使用"]
                st.session_state["building_selection_mode_index"] = mode_options.index(building_selection_mode)
                
                if building_selection_mode == "特定ビルを選択":
                    search_query = st.text_input(
                        "🔍 ビル名で検索",
                        placeholder="ビル名の一部を入力...",
                        help="入力した文字でビル一覧をフィルタリングできます"
                    )
                    
                    if search_query:
                        filtered_buildings = [
                            building for building in available_buildings 
                            if search_query.lower() in building.lower()
                        ]
                        st.info(f"🔍 検索結果: {len(filtered_buildings)}件")
                    else:
                        filtered_buildings = available_buildings
                    
                    if filtered_buildings:
                        selected_building = st.selectbox(
                            "ビルを選択してください",
                            options=[""] + filtered_buildings,
                            index=0,
                            help="上の検索ボックスで絞り込むか、直接選択してください"
                        )
                    else:
                        st.warning("⚠️ 検索条件に一致するビルが見つかりません")
                        selected_building = None
                    
                    # 🔥 新規追加: 他のビルも参考にするオプション
                    if selected_building:
                        include_other_buildings = st.checkbox(
                            "他のビルも参考にする",
                            value=st.session_state.get("include_other_buildings", False),
                            help="選択したビル以外の情報も比較・参考のために使用します"
                        )
                        st.session_state["include_other_buildings"] = include_other_buildings
                        
                        # building_mode の設定
                        if include_other_buildings:
                            st.session_state["building_mode"] = "specific_with_others"
                        else:
                            st.session_state["building_mode"] = "specific_only"
                    else:
                        st.session_state["include_other_buildings"] = False
                        st.session_state["building_mode"] = "specific_only"
                    
                    st.session_state["selected_building"] = selected_building if selected_building else None
                    
                elif building_selection_mode == "全ビル情報を使用":
                    st.info("🏢 全ビルの情報を使用して回答します")
                    st.session_state["selected_building"] = None
                    st.session_state["building_mode"] = "all"
                    st.session_state["include_other_buildings"] = False  # 全ビル使用時は無効
            
            else:
                st.session_state["selected_building"] = None
                st.session_state["building_mode"] = "none"
                st.session_state["include_other_buildings"] = False
            
            # 現在の選択状況を表示
            if include_building:
                current_building = st.session_state.get("selected_building")
                building_mode = st.session_state.get("building_mode", "none")
                include_others = st.session_state.get("include_other_buildings", False)
                
                if building_mode == "specific_only" and current_building:
                    st.success(f"✅ 選択中: **{current_building}** (単独)")
                    
                elif building_mode == "specific_with_others" and current_building:
                    other_count = len(available_buildings) - 1
                    st.success(f"✅ 基準ビル: **{current_building}**")
                    st.info(f"ℹ️ 他のビルも参考: {other_count}件のビル情報も使用")
                    
                elif building_mode == "all":
                    st.success("✅ 全ビル情報を使用")
                    
                # ビル詳細プレビュー
                if current_building:
                    with st.expander("🏢 ビル詳細情報", expanded=False):
                        building_info_text = get_building_info_for_prompt(current_building)
                        st.text_area(
                            "ビル情報プレビュー",
                            value=building_info_text,
                            height=300,
                            key=f"building_preview_{current_building}"
                        )
                elif building_mode == "all":
                    with st.expander("🏢 全ビル情報プレビュー", expanded=False):
                        all_building_info = get_building_info_for_prompt()
                        st.text_area(
                            "全ビル情報プレビュー",
                            value=all_building_info,
                            height=400,
                            key="all_buildings_preview"
                        )
            else:
                st.info("ℹ️ ビル情報は使用しません")

    def render_data_viewer():
        """資料内容確認UIを描画（共通関数）"""
        st.markdown("### 📚 資料内容確認")
        
        if not st.session_state.get("equipment_data"):
            st.error("❌ 設備データが読み込まれていません")
            return

        equipment_data = st.session_state.equipment_data
        
        # 統計情報の表示
        total_equipments = len(equipment_data)
        total_files = sum(data['total_files'] for data in equipment_data.values())
        total_chars = sum(data['total_chars'] for data in equipment_data.values())
        
        st.info(f"📊 **総統計**\n"
               f"- 設備数: {total_equipments}\n"
               f"- ファイル数: {total_files}\n"
               f"- 総文字数: {total_chars:,}")
        
        # 設備選択
        selected_equipment_for_view = st.selectbox(
            "📋 資料を確認する設備を選択",
            options=[""] + sorted(equipment_data.keys()),
            key="equipment_viewer_select"
        )
        
        if not selected_equipment_for_view:
            return
            
        equipment_info = equipment_data[selected_equipment_for_view]
        
        # 設備情報の表示
        st.markdown(f"#### 🔧 {selected_equipment_for_view}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ファイル数", equipment_info['total_files'])
            st.metric("ページ数", equipment_info['total_pages'])
        with col2:
            st.metric("文字数", f"{equipment_info['total_chars']:,}")
            st.markdown(f"**カテゴリ**: {equipment_info['equipment_category']}")
        
        # ファイル一覧と詳細表示
        st.markdown("##### 📄 ファイル一覧")
        
        for file_name in equipment_info['sources']:
            if "暗黙知メモ" in file_name:
                continue
                
            file_text = equipment_info['files'][file_name]
            file_chars = len(file_text)
            
            with st.expander(f"📄 {file_name} ({file_chars:,}文字)", expanded=False):
                st.markdown(f"**文字数**: {file_chars:,}")
                
                view_option = st.radio(
                    "表示方法",
                    ["プレビュー（最初の500文字）", "全文表示", "構造化表示"],
                    key=f"view_option_{selected_equipment_for_view}_{file_name}"
                )
                
                if view_option == "プレビュー（最初の500文字）":
                    preview_text = file_text[:500]
                    if len(file_text) > 500:
                        preview_text += "\n\n... （以下省略）"
                    st.text_area(
                        "プレビュー",
                        value=preview_text,
                        height=200,
                        key=f"preview_{selected_equipment_for_view}_{file_name}"
                    )
                    
                elif view_option == "全文表示":
                    st.text_area(
                        "全文",
                        value=file_text,
                        height=400,
                        key=f"fulltext_{selected_equipment_for_view}_{file_name}"
                    )
                    
                elif view_option == "構造化表示":
                    sections = file_text.split("--- ページ ")
                    
                    st.markdown("**ファイルヘッダー**:")
                    st.code(sections[0] if sections else "ヘッダーなし")
                    
                    if len(sections) > 1:
                        st.markdown("**ページ別内容**:")
                        for i, section in enumerate(sections[1:], 1):
                            page_lines = section.split("\n", 1)
                            if len(page_lines) >= 2:
                                page_num = page_lines[0].split(" ---")[0]
                                page_content = page_lines[1]
                                
                                with st.expander(f"ページ {page_num} ({len(page_content)}文字)", expanded=False):
                                    st.text_area(
                                        f"ページ {page_num} 内容",
                                        value=page_content,
                                        height=200,
                                        key=f"page_{selected_equipment_for_view}_{file_name}_{i}"
                                    )
                
                st.download_button(
                    label="📥 テキストをダウンロード",
                    data=file_text,
                    file_name=f"{selected_equipment_for_view}_{file_name}.txt",
                    mime="text/plain",
                    key=f"download_{selected_equipment_for_view}_{file_name}"
                )

    # =====  サイドバー  ==========================================================
    with st.sidebar:
        st.markdown(f"👤 ログインユーザー: `{name}`")
        authenticator.logout('ログアウト', 'sidebar')

        st.divider()

        st.header("💬 チャット履歴")
        
        # 🔥 デバッグ情報をサイドバーにも表示（開発時のみ）
        if st.checkbox("🔍 デバッグ表示", value=False):
            st.json({
                "current_chat": st.session_state.current_chat,
                "chat_sids_count": len(st.session_state.chat_sids),
                "chat_sids_keys": list(st.session_state.chat_sids.keys())
            })
        
        # 🔥 チャット履歴ボタンの改良（キーにタイトルも含める）
        for title, sid in st.session_state.chat_sids.items():
            # より一意なキーを生成（タイトル変更時の問題を回避）
            button_key = f"hist_{sid}_{hash(title) % 10000}"
            
            if st.button(title, key=button_key):
                st.session_state.chats.setdefault(title, [])
                switch_chat(title)
                logger.info("🔥 SIDEBAR CLICK - title=%r sid=%s", title, sid)

        if st.button("➕ 新しいチャット"):
            new_chat()
        
        st.divider()

        # ------- モデル選択 -------
        st.markdown("### 🤖 モデル選択")
        model_options = {
            "claude-4-sonnet": "Claude 4 Sonnet (最高性能・推奨)",
            "claude-3.7": "Claude 3.7 Sonnet (高性能)",
            "gpt-4.1": "GPT-4.1 (最新・高性能)",
            "gpt-4o": "GPT-4o(高性能)"
        }
        st.session_state.claude_model = st.selectbox(
            "使用するモデルを選択",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.claude_model) if st.session_state.claude_model in model_options else 0,
        )
        st.markdown(f"**🛈 現在のモデル:** `{model_options[st.session_state.claude_model]}`")

        # ------- モデル詳細設定 -------
        with st.expander("🔧 詳細設定"):
            st.slider("応答の多様性",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    key="temperature",
                    help="値が高いほど創造的、低いほど一貫した回答になります（Claudeデフォルト: 0.0）")

            # max_tokensのキーボード自由入力欄
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if "max_tokens" not in st.session_state or st.session_state.get("max_tokens") is None:
                    st.session_state["max_tokens"] = 4096
                
                max_tokens_text = st.text_input(
                    "最大応答長（トークン数）",
                    value=str(st.session_state.get("max_tokens", 4096)),
                    placeholder="例: 4096, 8000, 16000 （空欄=モデル上限使用）",
                    key="max_tokens_text",
                    help="数値を入力してください。空欄にするとモデルの上限値を使用します。"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                apply_button = st.button("✅ 適用", key="apply_max_tokens")
            
            current_max_tokens = st.session_state.get("max_tokens")
            if current_max_tokens is None:
                st.info("💡 現在の設定: モデル上限値を使用")
            else:
                st.info(f"💡 現在の設定: {current_max_tokens:,} トークン")
            
            if apply_button:
                if max_tokens_text.strip() == "":
                    st.session_state["max_tokens"] = None
                    st.success("✅ モデル上限値に設定しました")
                    st.rerun()
                else:
                    try:
                        max_tokens_value = int(max_tokens_text.strip())
                        
                        if max_tokens_value <= 0:
                            st.error("❌ 1以上の数値を入力してください")
                        elif max_tokens_value > 200000:
                            st.warning("⚠️ 200,000を超える値ですが設定しました")
                            st.session_state["max_tokens"] = max_tokens_value
                            st.success(f"✅ 最大トークン数を {max_tokens_value:,} に設定しました")
                            st.rerun()
                        else:
                            st.session_state["max_tokens"] = max_tokens_value
                            st.success(f"✅ 最大トークン数を {max_tokens_value:,} に設定しました")
                            st.rerun()
                            
                    except ValueError:
                        st.error("❌ 有効な数値を入力してください（例: 4096）")

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

        # ========== モード別のサイドバー表示（リファクタリング済み） ==========
        current_mode = st.session_state.design_mode
        
        if current_mode == "暗黙知法令チャットモード":
            # 設備選択
            render_equipment_selection()
            
            # ファイル選択（設備が選択されている場合のみ）
            current_equipment = st.session_state.get("selected_equipment")
            if current_equipment:
                render_file_selection(current_equipment)

            # ビル情報選択（閉じられた状態）
            render_building_selection(expanded=False)

            st.divider()

            # 資料内容確認
            render_data_viewer()

        elif current_mode == "質疑応答書添削モード":
            st.info("📝 質疑応答書添削モード用のサイドバーは後で実装予定")
            
        elif current_mode == "ビルマスタ質問モード":
            # ビル情報選択（そのまま表示）
            render_building_selection(expanded=True)
        
        else:
            st.warning(f"⚠️ 未対応のモード: {current_mode}")
        
        st.divider()

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
    else:
        st.title("💬 Claude + 設備資料チャットボット")

        st.subheader(f"🗣️ {st.session_state.current_chat}")
        st.markdown(f"**モデル:** {st.session_state.claude_model} | **モード:** {st.session_state.design_mode}")

        # -- メッセージ表示 --
        st.markdown('<div class="chat-body">', unsafe_allow_html=True)
        for idx, m in enumerate(get_messages()):
            message_class = "user-message" if m["role"] == "user" else "assistant-message"

            with st.chat_message(m["role"]):
                st.markdown(
                    f'<div class="{message_class}">{m["content"]}</div>',
                    unsafe_allow_html=True
                )

            # 使用設備・ファイルを表示（アシスタントメッセージの場合）
            if m["role"] == "assistant" and "used_equipment" in m:
                equipment_name = m['used_equipment']
                used_files = m.get('used_files', [])
                
                if used_files:
                    file_count_info = f"（{len(used_files)}ファイル使用）"
                    with st.expander(f"🔧 使用設備: {equipment_name} {file_count_info}", expanded=False):
                        for file in used_files:
                            st.markdown(f"- 📄 {file}")
                else:
                    st.info(f"🔧 使用設備: {equipment_name}")

        st.markdown('</div>', unsafe_allow_html=True)

        # -- 入力欄 --
        user_prompt = st.chat_input("メッセージを入力…")

    # =====  🔥 LangChain統合による応答生成  ============================================================
    if user_prompt and not st.session_state.edit_target:
        # メッセージリストに現在の質問を追加
        msgs = get_messages()
        msgs.append({"role": "user", "content": user_prompt})

        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{user_prompt}</div>', unsafe_allow_html=True)

        # シンプルなステータス表示
        with st.status(f"🤖 {st.session_state.claude_model} で回答を生成中...", expanded=True) as status:
            # プロンプト取得
            base_prompt = st.session_state.prompts[st.session_state.design_mode]
            # 🔥 LangChainでプロンプト処理も自動化されるため簡素化
            prompt = base_prompt

            logger.info("💬 gen_start — mode=%s model=%s sid=%s",
                st.session_state.design_mode,
                st.session_state.claude_model,
                st.session_state.sid)

            try:
                # データ準備
                prompt_data = prepare_prompt_data()
                
                # 使用データの表示
                if prompt_data["equipment_content"]:
                    selected_equipment = st.session_state.get("selected_equipment")
                    selected_files_key = f"selected_files_{selected_equipment}"
                    selected_files = st.session_state.get(selected_files_key, [])
                    st.info(f"📄 設備資料使用: {selected_equipment} ({len(selected_files)}ファイル)")
                
                if prompt_data["building_content"]:
                    building_mode = st.session_state.get("building_mode", "none")
                    if building_mode == "specific":
                        selected_building = st.session_state.get("selected_building")
                        st.info(f"🏢 ビル情報使用: {selected_building}")
                    elif building_mode == "all":
                        st.info("🏢 全ビル情報使用")
                
                if not prompt_data["equipment_content"] and not prompt_data["building_content"]:
                    st.info("💭 一般知識による回答")
                
                # 🔥 LangChainによる統一回答生成
                st.info("🚀 LangChainで最適化された回答を生成中...")
                
                import time
                t_api = time.perf_counter()
                
                result = generate_smart_answer_with_langchain(
                    prompt=prompt,
                    question=user_prompt,
                    model=st.session_state.claude_model,
                    mode=prompt_data["mode"],
                    equipment_content=prompt_data["equipment_content"],
                    building_content=prompt_data["building_content"],
                    target_building_content=prompt_data.get("target_building_content"),  # 🔥 新規追加
                    other_buildings_content=prompt_data.get("other_buildings_content"),   # 🔥 新規追加
                    chat_history=msgs,
                    temperature=st.session_state.get("temperature", 0.0),
                    max_tokens=st.session_state.get("max_tokens")
                )
                
                api_elapsed = time.perf_counter() - t_api
                
                assistant_reply = result["answer"]
                complete_prompt = result.get("complete_prompt", prompt)
                
                # 使用した設備・ファイル情報の記録
                used_equipment = "なし（一般知識による回答）"
                used_files = []
                
                if prompt_data["equipment_content"]:
                    selected_equipment = st.session_state.get("selected_equipment")
                    if selected_equipment:
                        used_equipment = selected_equipment
                        selected_files_key = f"selected_files_{selected_equipment}"
                        used_files = st.session_state.get(selected_files_key, [])
                
                processing_mode = "equipment_with_files" if used_files else "no_equipment"
                
                # ステータス表示
                if processing_mode == "equipment_with_files":
                    st.success(f"✅ 設備資料を使用した回答: {used_equipment} ({len(used_files)}ファイル)")
                elif processing_mode == "equipment_no_files":
                    st.info(f"📋 設備選択済み（ファイル未選択）: {used_equipment}")
                elif processing_mode == "no_equipment":
                    st.info(f"💭 {used_equipment}")
                else:
                    st.info(f"🔧 処理モード: {processing_mode}")
                
                logger.info("💬 LangChain回答完了 — mode=%s equipment=%s files=%d api_elapsed=%.2fs 回答文字数=%d",
                        processing_mode, used_equipment, len(used_files), api_elapsed, len(assistant_reply))

            except Exception as e:
                logger.exception("❌ LangChain answer_gen failed — %s", e)
                st.error(f"回答生成時にエラーが発生しました: {e}")
                st.stop()

            # 画面反映 
            with st.chat_message("assistant"):
                # モデル情報と使用設備・ファイルを応答に追加 
                if used_files:
                    file_info = f"（{len(used_files)}ファイル使用）"
                    model_info = f"\n\n---\n*このレスポンスは `{st.session_state.claude_model}` と設備「{used_equipment}」{file_info}で生成されました*"
                else:
                    model_info = f"\n\n---\n*このレスポンスは `{st.session_state.claude_model}` で生成されました（設備資料なし）*"
                
                full_reply = assistant_reply + model_info
                st.markdown(full_reply)

            # 保存するのは元の応答（付加情報なし）
            msg_to_save = {
                "role": "assistant",
                "content": assistant_reply,
            }
            
            # 設備・ファイル情報がある場合のみ追加 
            if used_equipment and used_equipment != "なし（一般知識による回答）":
                msg_to_save["used_equipment"] = used_equipment
                msg_to_save["used_files"] = used_files

            msgs.append(msg_to_save)

            logger.info("📝 === TITLE GENERATION IMPROVED START ===")
            try:
                logger.info(f"📊 Current state: msgs_count={len(msgs)}, current_chat='{st.session_state.current_chat}'")
                
                is_first_message = len(msgs) == 2
                is_default_title = (
                    st.session_state.current_chat.startswith("Chat ") or 
                    st.session_state.current_chat == "New Chat"
                )
                
                logger.info(f"✅ is_first_message: {is_first_message}")
                logger.info(f"✅ is_default_title: {is_default_title}")
                
                if is_first_message and is_default_title:
                    logger.info("🎯 TITLE GENERATION CONDITIONS MET!")
                    
                    user_content = msgs[0]['content'][:200]
                    logger.info(f"📝 Generating title for: '{user_content}'")
                    
                    # タイトル生成
                    new_title = generate_chat_title_with_llm(
                        user_message=user_content,
                        model=st.session_state.claude_model,
                        temperature=0.0,
                        max_tokens=30
                    )
                    
                    logger.info(f"🏷️ Generated title: '{new_title}'")
                    
                    # 🚨 関数が定義されているかチェック
                    if 'update_chat_title_safely' in globals():
                        logger.info("✅ update_chat_title_safely function found in globals")
                    else:
                        logger.error("❌ update_chat_title_safely function NOT found in globals")
                    
                    # 🔥 改良された統合タイトル更新関数を使用
                    logger.info("🚀 About to call update_chat_title_safely...")
                    try:
                        update_result = update_chat_title_safely(new_title, force_rerun=True)
                        logger.info(f"🎯 update_chat_title_safely returned: {update_result}")
                        
                        if update_result:
                            # 更新成功時は、rerun により処理が停止するため、以降のコードは実行されない
                            logger.info("✅ Title update initiated with rerun - PROCESSING SHOULD STOP HERE")
                            # return は使えないので、代わりに st.stop() を使用
                            st.stop()
                        else:
                            logger.warning("⚠️ Title update failed or skipped")
                            
                    except Exception as title_update_error:
                        logger.error(f"💥 update_chat_title_safely failed: {title_update_error}", exc_info=True)
                        
                        # 🔥 フォールバック：従来の方法でタイトル更新
                        logger.info("🔄 Falling back to manual title update...")
                        try:
                            # 手動でタイトル更新
                            s = st.session_state.chat_store
                            sid = s["current_sid"]
                            sanitized_title = _sanitize_title(new_title)
                            
                            # 重複回避
                            existing_titles = {row["title"] for row in s["by_id"].values() if row != s["by_id"][sid]}
                            final_title = sanitized_title
                            counter = 2
                            while final_title in existing_titles:
                                final_title = f"{sanitized_title} ({counter})"
                                counter += 1
                            
                            # 更新実行
                            s["by_id"][sid]["title"] = final_title
                            
                            # ミラー同期を強制実行
                            ensure_chat_store()
                            
                            logger.info(f"🔄 Manual title update completed: '{final_title}'")
                            st.rerun()
                            
                        except Exception as fallback_error:
                            logger.error(f"💥 Manual title update also failed: {fallback_error}", exc_info=True)
                    
                else:
                    logger.info(f"❌ Title generation skipped - first_msg:{is_first_message}, default_title:{is_default_title}")
                    
            except Exception as e:
                logger.error(f"💥 Title generation error: {e}", exc_info=True)

            logger.info("📝 === TITLE GENERATION IMPROVED END ===")

            # 🔥 二重rerun防止システムの簡素化
            # タイトル更新時は即座にrerunされるため、以下の複雑な制御は不要

            # ログ保存
            logger.info("📝 Executing post_log operations")
            post_log_async(user_prompt, assistant_reply, complete_prompt, send_to_model_comparison=True) 
            post_log_firestore_async(user_prompt, assistant_reply, complete_prompt, send_to_model_comparison=True)

            # 通常のrerun（タイトル更新時以外）
            logger.info("⏳ Final rerun check")
            time.sleep(3)
            st.rerun()

elif st.session_state["authentication_status"] is False:
    st.error('ユーザー名またはパスワードが間違っています。')
elif st.session_state["authentication_status"] is None:
    st.warning("ユーザー名とパスワードを入力してください。")
    st.stop()