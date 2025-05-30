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

# デバッグ用関数を追加
def debug_connection():
    """詳細な接続診断"""
    print("🔧 Google Sheets接続診断開始...")
    
    # Step 1: Streamlit secrets確認
    print("\n--- Step 1: Streamlit secrets確認 ---")
    try:
        secrets_keys = list(st.secrets.keys())
        print(f"✅ secrets利用可能 - キー: {secrets_keys}")
        
        if "gcp_service_account" in st.secrets:
            gcp_keys = list(st.secrets["gcp_service_account"].keys())
            print(f"✅ gcp_service_account セクション存在 - キー: {gcp_keys}")
        else:
            print("❌ gcp_service_account セクションが見つかりません")
            return
            
        if "SPREADSHEET_ID" in st.secrets:
            print(f"✅ SPREADSHEET_ID存在: {st.secrets['SPREADSHEET_ID'][:10]}...")
        else:
            print("❌ SPREADSHEET_IDが見つかりません")
            return
            
    except Exception as e:
        print(f"❌ secrets確認失敗: {e}")
        return
    
    # Step 2: 認証テスト
    print("\n--- Step 2: Google認証テスト ---")
    try:
        credentials_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            credentials_info, 
            scopes=[
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
        )
        print("✅ 認証情報作成成功")
        
        client = gspread.authorize(creds)
        print("✅ gspreadクライアント作成成功")
        
    except Exception as e:
        print(f"❌ 認証失敗: {e}")
        return
    
    # Step 3: スプレッドシートアクセステスト
    print("\n--- Step 3: スプレッドシートアクセステスト ---")
    try:
        spreadsheet_id = st.secrets["SPREADSHEET_ID"]
        spreadsheet = client.open_by_key(spreadsheet_id)
        print(f"✅ スプレッドシート接続成功: {spreadsheet.title}")
        
        # ワークシート一覧表示
        worksheets = spreadsheet.worksheets()
        print(f"✅ ワークシート一覧: {[ws.title for ws in worksheets]}")
        
    except Exception as e:
        print(f"❌ スプレッドシートアクセス失敗: {e}")
        print("考えられる原因:")
        print("  - スプレッドシートIDが間違っている")
        print("  - サービスアカウントに共有権限がない")
        print("  - Google Sheets APIが有効化されていない")
        return
    
    # Step 4: 書き込みテスト
    print("\n--- Step 4: 書き込みテスト ---")
    try:
        manager = SheetsManager()
        if manager.is_connected:
            success = manager.log_conversation(
                user_id="debug_user",
                session_id="debug_session",
                mode="デバッグテスト",
                model="debug",
                input_text="デバッグ質問",
                output_text="デバッグ回答",
                prompt_used="デバッグプロンプト",
                metadata={"debug": True}
            )
            
            if success:
                print("✅ 書き込みテスト成功")
            else:
                print("❌ 書き込みテスト失敗")
        else:
            print("❌ マネージャー接続失敗")
            
    except Exception as e:
        print(f"❌ 書き込みテスト例外: {e}")
    
    print("\n🔧 診断完了")

def debug_connection_streamlit():
    """Streamlit用の詳細な接続診断"""
    st.write("🔧 Google Sheets接続診断開始...")
    
    # Step 1: Streamlit secrets確認
    st.write("### Step 1: Streamlit secrets確認")
    try:
        secrets_keys = list(st.secrets.keys())
        st.success(f"✅ secrets利用可能 - キー: {secrets_keys}")
        
        if "gcp_service_account" in st.secrets:
            gcp_keys = list(st.secrets["gcp_service_account"].keys())
            st.success(f"✅ gcp_service_account セクション存在 - キー: {gcp_keys}")
        else:
            st.error("❌ gcp_service_account セクションが見つかりません")
            return
            
        if "SPREADSHEET_ID" in st.secrets:
            st.success(f"✅ SPREADSHEET_ID存在: {st.secrets['SPREADSHEET_ID'][:10]}...")
        else:
            st.error("❌ SPREADSHEET_IDが見つかりません")
            return
            
    except Exception as e:
        st.error(f"❌ secrets確認失敗: {e}")
        return
    
    # Step 2: 認証テスト
    st.write("### Step 2: Google認証テスト")
    try:
        credentials_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            credentials_info, 
            scopes=[
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
        )
        st.success("✅ 認証情報作成成功")
        
        client = gspread.authorize(creds)
        st.success("✅ gspreadクライアント作成成功")
        
    except Exception as e:
        st.error(f"❌ 認証失敗: {e}")
        st.write("**考えられる原因:**")
        st.write("- private_keyの改行が正しくない")
        st.write("- サービスアカウントの設定が間違っている")
        return
    
    # Step 3: スプレッドシートアクセステスト
    st.write("### Step 3: スプレッドシートアクセステスト")
    try:
        spreadsheet_id = st.secrets["SPREADSHEET_ID"]
        spreadsheet = client.open_by_key(spreadsheet_id)
        st.success(f"✅ スプレッドシート接続成功: {spreadsheet.title}")
        
        # ワークシート一覧表示
        worksheets = spreadsheet.worksheets()
        st.success(f"✅ ワークシート一覧: {[ws.title for ws in worksheets]}")
        
    except Exception as e:
        st.error(f"❌ スプレッドシートアクセス失敗: {e}")
        st.write("**考えられる原因:**")
        st.write("- スプレッドシートIDが間違っている")
        st.write("- サービスアカウントに共有権限がない ← **最も可能性が高い**")
        st.write("- Google Sheets APIが有効化されていない")
        
        st.write("**解決方法:**")
        st.write("1. Google Sheetsを開く")
        st.write("2. 右上の「共有」ボタンをクリック")
        st.write("3. 以下のメールアドレスを追加:")
        st.code("sheets-service-account@streamlit-spread-integration.iam.gserviceaccount.com")
        st.write("4. 権限を「編集者」に設定")
        return
    
    # Step 4: 書き込みテスト
    st.write("### Step 4: 書き込みテスト")
    try:
        manager = SheetsManager()
        if manager.is_connected:
            success = manager.log_conversation(
                user_id="debug_user",
                session_id="debug_session",
                mode="デバッグテスト",
                model="debug",
                input_text="デバッグ質問",
                output_text="デバッグ回答",
                prompt_used="デバッグプロンプト",
                metadata={"debug": True}
            )
            
            if success:
                st.success("✅ 書き込みテスト成功")
                st.balloons()  # 成功時のアニメーション
            else:
                st.error("❌ 書き込みテスト失敗")
        else:
            st.error("❌ マネージャー接続失敗")
            
    except Exception as e:
        st.error(f"❌ 書き込みテスト例外: {e}")
    
    st.write("### 🔧 診断完了")

# 追加: 簡単な接続状態確認関数
def check_connection_status():
    """簡単な接続状態チェック"""
    try:
        manager = get_sheets_manager()
        return manager.is_connected
    except:
        return False

# 接続テスト用
def test_connection():
    """簡単な接続テスト"""
    debug_connection()

if __name__ == "__main__":
    test_connection()