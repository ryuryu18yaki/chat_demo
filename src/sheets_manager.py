import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
import streamlit as st
import json
from typing import List, Dict, Any, Optional

class SheetsManager:
    def __init__(self):
        """gspread専用のGoogle Sheetsマネージャー"""
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        self.client = None
        self.spreadsheet = None
        self.is_connected = False
        self._initialize()
    
    def _initialize(self):
        """初期化処理"""
        try:
            # 認証情報の取得
            credentials_info = st.secrets["gcp_service_account"]
            spreadsheet_id = st.secrets["SPREADSHEET_ID"]
            
            # 認証
            creds = Credentials.from_service_account_info(
                credentials_info, scopes=self.scope
            )
            self.client = gspread.authorize(creds)
            
            # スプレッドシートを開く
            self.spreadsheet = self.client.open_by_key(spreadsheet_id)
            self.is_connected = True
            
            print(f"✅ Sheets接続成功: {self.spreadsheet.title}")
            
        except Exception as e:
            print(f"❌ Sheets接続失敗: {e}")
            self.is_connected = False
    
    def _ensure_worksheet(self, name: str, headers: List[str]) -> Optional[gspread.Worksheet]:
        """ワークシートの存在確認・作成"""
        if not self.is_connected:
            return None
        
        try:
            worksheet = self.spreadsheet.worksheet(name)
            # 既存ワークシートの場合、ヘッダーをチェック
            existing_headers = worksheet.row_values(1)
            if not existing_headers or existing_headers != headers:
                # ヘッダーが異なる場合は更新
                worksheet.clear()
                worksheet.append_row(headers)
                print(f"📋 ワークシートヘッダー更新: {name}")
            
        except gspread.WorksheetNotFound:
            # 新規作成
            worksheet = self.spreadsheet.add_worksheet(
                title=name, 
                rows=1000, 
                cols=len(headers)
            )
            worksheet.append_row(headers)
            print(f"📋 新規ワークシート作成: {name}")
        
        return worksheet
    
    def log_conversation(self, 
                        user_id: str,
                        session_id: str,
                        mode: str,
                        model: str,
                        input_text: str, 
                        output_text: str,
                        prompt_used: str,
                        metadata: Dict[str, Any] = None) -> bool:
        """会話ログをSheetsに保存"""
        
        if not self.is_connected:
            return False
        
        # ヘッダー定義
        headers = [
            "timestamp", "user_id", "session_id", "mode", "model", 
            "input_text", "output_text", "prompt_used", "metadata"
        ]
        
        # ワークシート準備
        worksheet = self._ensure_worksheet("conversations", headers)
        if not worksheet:
            return False
        
        # データ準備（文字数制限）
        row_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_id,
            session_id,
            mode,
            model,
            self._truncate_text(input_text, 1000),
            self._truncate_text(output_text, 2000),
            self._truncate_text(prompt_used, 500),
            json.dumps(metadata or {}, ensure_ascii=False)
        ]
        
        try:
            worksheet.append_row(row_data)
            return True
        except Exception as e:
            print(f"❌ ログ保存失敗: {e}")
            return False
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """テキストを指定長さで切り詰め"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def get_recent_conversations(self, days: int = 7) -> pd.DataFrame:
        """最近の会話データを取得"""
        if not self.is_connected:
            return pd.DataFrame()
        
        try:
            worksheet = self.spreadsheet.worksheet("conversations")
            data = worksheet.get_all_records()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 指定日数でフィルター
            cutoff = datetime.now() - pd.Timedelta(days=days)
            return df[df['timestamp'] >= cutoff].sort_values('timestamp', ascending=False)
            
        except Exception as e:
            print(f"❌ データ取得失敗: {e}")
            return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        df = self.get_recent_conversations(days=30)
        
        if df.empty:
            return {"total": 0, "by_mode": {}, "by_model": {}}
        
        return {
            "total": len(df),
            "by_mode": df['mode'].value_counts().to_dict(),
            "by_model": df['model'].value_counts().to_dict(),
            "users": df['user_id'].nunique(),
            "sessions": df['session_id'].nunique()
        }

# Streamlit統合用ヘルパー
@st.cache_resource
def get_sheets_manager():
    """キャッシュされたSheetsManagerインスタンス"""
    return SheetsManager()

def log_to_sheets(input_text: str, output_text: str, prompt: str):
    """Streamlitアプリから呼び出すログ関数"""
    
    manager = get_sheets_manager()
    
    if not manager.is_connected:
        st.warning("⚠️ Google Sheets接続エラー")
        return False
    
    # メタデータ準備
    metadata = {
        "temperature": st.session_state.get("temperature", 1.0),
        "max_tokens": st.session_state.get("max_tokens"),
        "use_rag": st.session_state.get("use_rag", False),
        "timestamp": datetime.now().isoformat()
    }
    
    # 保存実行
    success = manager.log_conversation(
        user_id=st.session_state.get("username", "unknown"),
        session_id=st.session_state.get("sid", "unknown"),
        mode=st.session_state.get("design_mode", ""),
        model=st.session_state.get("gpt_model", ""),
        input_text=input_text,
        output_text=output_text,
        prompt_used=prompt,
        metadata=metadata
    )
    
    return success

# 接続テスト用
def test_connection():
    """接続テスト"""
    print("🔧 Google Sheets接続テスト開始...")
    
    try:
        manager = SheetsManager()
        
        if manager.is_connected:
            print("✅ 接続成功")
            
            # テストデータ
            success = manager.log_conversation(
                user_id="test_user",
                session_id="test_session",
                mode="テストモード",
                model="gpt-4o",
                input_text="テスト質問です",
                output_text="テスト回答です",
                prompt_used="テストプロンプト",
                metadata={"test": True}
            )
            
            if success:
                print("✅ データ書き込み成功")
            else:
                print("❌ データ書き込み失敗")
                
        else:
            print("❌ 接続失敗")
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")

if __name__ == "__main__":
    test_connection()