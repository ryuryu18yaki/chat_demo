# src/firestore_manager.py

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
import streamlit as st
import json
from typing import Dict, Any, List, Optional
import uuid

from src.logging_utils import init_logger
logger = init_logger()

class FirestoreManager:
    """Firestore用のログ管理クラス"""
    
    def __init__(self):
        self.db = None
        self.is_connected = False
        self._initialize()
    
    def _initialize(self):
        """Firestoreの初期化"""
        try:
            # 既存のアプリがあるかチェック
            if not firebase_admin._apps:
                # Streamlit secretsからFirebase認証情報を取得
                firebase_credentials = st.secrets["firebase_credentials"]
                
                # 🔥 修正: 辞書を直接渡すのではなく、一時ファイルを作成
                import tempfile
                import json
                import os
                
                # 一時ファイルに認証情報を書き込み
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    json.dump(dict(firebase_credentials), temp_file, indent=2)
                    temp_file_path = temp_file.name
                
                try:
                    # 一時ファイルのパスを使って認証
                    cred = credentials.Certificate(temp_file_path)
                    firebase_admin.initialize_app(cred)
                    
                finally:
                    # 一時ファイルを削除
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            
            # Firestoreクライアントを取得
            self.db = firestore.client()
            self.is_connected = True
            
            logger.info("✅ Firestore接続成功")
            
        except Exception as e:
            logger.error(f"❌ Firestore接続失敗: {e}")
            self.is_connected = False
    
    def log_conversation(self, 
                        user_id: str,
                        session_id: str,
                        mode: str,
                        model: str,
                        input_text: str, 
                        output_text: str,
                        prompt_used: str,
                        chat_title: str = None,
                        metadata: Dict[str, Any] = None) -> bool:
        """会話ログをFirestoreに保存"""
        
        if not self.is_connected:
            logger.error("❌ Firestore未接続")
            return False
        
        try:
            # 🔥 文字数制限を大幅に緩和（Firestoreの最大ドキュメントサイズは1MB）
            # ドキュメントデータの準備
            doc_data = {
                "timestamp": datetime.now(timezone.utc),
                "user_id": user_id,
                "session_id": session_id,
                "chat_title": chat_title or "未設定",
                "mode": mode,
                "model": model,
                "input_text": self._truncate_text(input_text, 100000),    # 10万文字に拡大
                "output_text": self._truncate_text(output_text, 200000),   # 20万文字に拡大
                "prompt_used": self._truncate_text(prompt_used, 500000),    # 50万文字に拡大（完全プロンプト用）
                "metadata": metadata or {},
                "id": str(uuid.uuid4())
            }
            
            # conversationsコレクションに追加
            doc_ref = self.db.collection('conversations').add(doc_data)
            
            logger.info(f"✅ 会話ログ保存成功: {doc_ref[1].id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 会話ログ保存失敗: {e}")
            return False
    
    def send_to_model_comparison(self, prompt_text: str, user_note: str = None) -> bool:
        """モデル比較用プロンプトをFirestoreに保存"""
        
        if not self.is_connected:
            logger.error("❌ Firestore未接続")
            return False
        
        try:
            doc_data = {
                "timestamp": datetime.now(timezone.utc),
                "prompt_text": prompt_text,
                "user_note": user_note,
                "processed": False,  # 処理済みフラグ
                "id": str(uuid.uuid4())
            }
            
            # model_comparisonsコレクションに追加
            doc_ref = self.db.collection('model_comparisons').add(doc_data)
            
            logger.info(f"✅ モデル比較プロンプト保存成功: {doc_ref[1].id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ モデル比較プロンプト保存失敗: {e}")
            return False
    
    def get_recent_conversations(self, 
                               user_id: str = None, 
                               session_id: str = None,
                               days: int = 7,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """最近の会話データを取得"""
        
        if not self.is_connected:
            return []
        
        try:
            # クエリを構築
            query = self.db.collection('conversations')
            
            # フィルター条件を追加
            if user_id:
                query = query.where('user_id', '==', user_id)
            if session_id:
                query = query.where('session_id', '==', session_id)
            
            # 日付フィルター
            cutoff_date = datetime.now(timezone.utc) - datetime.timedelta(days=days)
            query = query.where('timestamp', '>=', cutoff_date)
            
            # 並び順と件数制限
            query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)
            query = query.limit(limit)
            
            # クエリ実行
            docs = query.stream()
            
            conversations = []
            for doc in docs:
                data = doc.to_dict()
                data['firestore_id'] = doc.id
                conversations.append(data)
            
            logger.info(f"✅ 会話データ取得成功: {len(conversations)}件")
            return conversations
            
        except Exception as e:
            logger.error(f"❌ 会話データ取得失敗: {e}")
            return []
    
    def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """統計情報を取得"""
        
        if not self.is_connected:
            return {"total": 0, "by_mode": {}, "by_model": {}}
        
        try:
            # 最近のデータを取得
            conversations = self.get_recent_conversations(days=days, limit=1000)
            
            if not conversations:
                return {"total": 0, "by_mode": {}, "by_model": {}}
            
            # 統計計算
            by_mode = {}
            by_model = {}
            users = set()
            sessions = set()
            
            for conv in conversations:
                mode = conv.get('mode', 'unknown')
                model = conv.get('model', 'unknown')
                user_id = conv.get('user_id')
                session_id = conv.get('session_id')
                
                by_mode[mode] = by_mode.get(mode, 0) + 1
                by_model[model] = by_model.get(model, 0) + 1
                
                if user_id:
                    users.add(user_id)
                if session_id:
                    sessions.add(session_id)
            
            return {
                "total": len(conversations),
                "by_mode": by_mode,
                "by_model": by_model,
                "users": len(users),
                "sessions": len(sessions)
            }
            
        except Exception as e:
            logger.error(f"❌ 統計情報取得失敗: {e}")
            return {"total": 0, "by_mode": {}, "by_model": {}}
    
    def search_conversations(self, 
                           query_text: str,
                           user_id: str = None,
                           days: int = 30,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """会話の検索（テキスト部分一致）"""
        
        conversations = self.get_recent_conversations(
            user_id=user_id, 
            days=days, 
            limit=limit*2  # 検索前に多めに取得
        )
        
        # クライアントサイドでテキスト検索
        query_lower = query_text.lower()
        filtered = []
        
        for conv in conversations:
            input_text = conv.get('input_text', '').lower()
            output_text = conv.get('output_text', '').lower()
            
            if query_lower in input_text or query_lower in output_text:
                filtered.append(conv)
                
                if len(filtered) >= limit:
                    break
        
        return filtered
    
    def delete_old_conversations(self, days: int = 90) -> int:
        """古い会話データを削除"""
        
        if not self.is_connected:
            return 0
        
        try:
            cutoff_date = datetime.now(timezone.utc) - datetime.timedelta(days=days)
            
            # 削除対象のドキュメントを取得
            query = self.db.collection('conversations').where('timestamp', '<', cutoff_date)
            docs = query.stream()
            
            deleted_count = 0
            batch = self.db.batch()
            
            for doc in docs:
                batch.delete(doc.reference)
                deleted_count += 1
                
                # バッチサイズ制限（Firestoreは500件まで）
                if deleted_count % 500 == 0:
                    batch.commit()
                    batch = self.db.batch()
            
            # 残りのバッチをコミット
            if deleted_count % 500 != 0:
                batch.commit()
            
            logger.info(f"✅ 古い会話データ削除完了: {deleted_count}件")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ 古い会話データ削除失敗: {e}")
            return 0
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """テキストを指定長さで切り詰め"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."

# Streamlit統合用ヘルパー
@st.cache_resource
def get_firestore_manager():
    """キャッシュされたFirestoreManagerインスタンス"""
    return FirestoreManager()

def log_to_firestore(
    input_text: str, 
    output_text: str, 
    prompt: str, 
    chat_title: str = None,
    user_id: str = None,
    session_id: str = None,
    mode: str = None,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    use_rag: bool = None
) -> bool:
    """Streamlitアプリからの便利なログ関数"""
    
    manager = get_firestore_manager()
    
    if not manager.is_connected:
        return False
    
    # session_stateから値を取得（非同期対応）
    try:
        title_to_use = chat_title or st.session_state.get("current_chat", "未設定")
        final_user_id = user_id or st.session_state.get("username", "unknown")
        final_session_id = session_id or st.session_state.get("sid", "unknown")
        final_mode = mode or st.session_state.get("design_mode", "")
        final_model = model or st.session_state.get("claude_model", "")
        final_temperature = temperature if temperature is not None else st.session_state.get("temperature", 0.0)
        final_max_tokens = max_tokens if max_tokens is not None else st.session_state.get("max_tokens")
        final_use_rag = use_rag if use_rag is not None else st.session_state.get("use_rag", False)
    except Exception:
        # session_stateにアクセスできない場合
        title_to_use = chat_title or "未設定"
        final_user_id = user_id or "unknown"
        final_session_id = session_id or "unknown"
        final_mode = mode or ""
        final_model = model or ""
        final_temperature = temperature if temperature is not None else 0.0
        final_max_tokens = max_tokens
        final_use_rag = use_rag if use_rag is not None else False
    
    # メタデータ準備
    metadata = {
        "temperature": final_temperature,
        "max_tokens": final_max_tokens,
        "use_rag": final_use_rag,
        "app_version": "1.0"
    }
    
    # 保存実行
    return manager.log_conversation(
        user_id=final_user_id,
        session_id=final_session_id,
        mode=final_mode,
        model=final_model,
        input_text=input_text,
        output_text=output_text,
        prompt_used=prompt,
        chat_title=title_to_use,
        metadata=metadata
    )

def send_prompt_to_firestore_comparison(prompt_text: str, user_note: str = None) -> bool:
    """モデル比較用プロンプトをFirestoreに送信"""
    
    manager = get_firestore_manager()
    return manager.send_to_model_comparison(prompt_text, user_note)

# 接続テスト用
def test_firestore_connection():
    """Firestore接続テスト"""
    print("🔧 Firestore接続テスト開始...")
    
    try:
        manager = FirestoreManager()
        
        if manager.is_connected:
            print("✅ 接続成功")
            
            # テストデータの書き込み
            success = manager.log_conversation(
                user_id="test_user",
                session_id="test_session",
                mode="テストモード",
                model="test-model",
                input_text="テスト質問です",
                output_text="テスト回答です",
                prompt_used="テストプロンプト",
                chat_title="テストチャット",
                metadata={"test": True}
            )
            
            if success:
                print("✅ データ書き込みテスト成功")
            else:
                print("❌ データ書き込みテスト失敗")
            
            # 統計情報の取得テスト
            stats = manager.get_stats()
            print(f"✅ 統計情報取得成功: {stats}")
            
        else:
            print("❌ 接続失敗")
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")

if __name__ == "__main__":
    test_firestore_connection()