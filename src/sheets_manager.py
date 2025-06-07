# src/sheets_manager.py

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
                        chat_title: str = None,
                        metadata: Dict[str, Any] = None) -> bool:
        """会話ログをSheetsに保存"""
        
        if not self.is_connected:
            return False
        
        # ヘッダー定義（chat_titleを追加）
        headers = [
            "timestamp", "user_id", "session_id", "chat_title", "mode", "model", 
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
            chat_title or "未設定",
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
    
    def send_to_model_comparison(self, prompt_text: str, user_note: str = None) -> bool:
        """model比較シートのB列にプロンプトを送信"""
        
        if not self.is_connected:
            print("❌ Sheets未接続")
            return False
        
        try:
            # model比較シートを取得
            worksheet = self.spreadsheet.worksheet("model比較")
            
            # 次の空いている行を見つける
            # B列の値を取得して、最初の空のセルを探す
            b_column_values = worksheet.col_values(2)  # B列 = index 2
            
            # 空の行を見つける（ヘッダー行は除く）
            next_row = len(b_column_values) + 1
            if next_row <= 1:  # ヘッダー行がない場合
                next_row = 2
            
            # B列にプロンプトのみを書き込み
            worksheet.update_cell(next_row, 2, prompt_text)  # 2 = B列
            
            print(f"✅ model比較シート書き込み成功: 行{next_row}")
            return True
            
        except gspread.WorksheetNotFound:
            print("❌ 'model比較'シートが見つかりません")
            return False
        except Exception as e:
            print(f"❌ model比較シート書き込み失敗: {e}")
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

def log_to_sheets(
    input_text: str, 
    output_text: str, 
    prompt: str, 
    chat_title: str = None,
    # 非同期対応のための新しい引数
    user_id: str = None,
    session_id: str = None,
    mode: str = None,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    use_rag: bool = None
) -> bool:
    """Streamlitアプリから呼び出すログ関数（非同期対応版）"""
    
    manager = get_sheets_manager()
    
    if not manager.is_connected:
        return False
    
    # 引数が提供されていない場合はsession_stateから取得（メインスレッドの場合）
    try:
        title_to_use = chat_title or st.session_state.get("current_chat", "未設定")
        final_user_id = user_id or st.session_state.get("username", "unknown")
        final_session_id = session_id or st.session_state.get("sid", "unknown")
        final_mode = mode or st.session_state.get("design_mode", "")
        final_model = model or st.session_state.get("gpt_model", "")
        final_temperature = temperature if temperature is not None else st.session_state.get("temperature", 1.0)
        final_max_tokens = max_tokens if max_tokens is not None else st.session_state.get("max_tokens")
        final_use_rag = use_rag if use_rag is not None else st.session_state.get("use_rag", False)
    except Exception:
        # session_stateにアクセスできない場合（非同期スレッド）は引数の値をそのまま使用
        title_to_use = chat_title or "未設定"
        final_user_id = user_id or "unknown"
        final_session_id = session_id or "unknown"
        final_mode = mode or ""
        final_model = model or ""
        final_temperature = temperature if temperature is not None else 1.0
        final_max_tokens = max_tokens
        final_use_rag = use_rag if use_rag is not None else False
    
    # メタデータ準備
    metadata = {
        "temperature": final_temperature,
        "max_tokens": final_max_tokens,
        "use_rag": final_use_rag,
        "timestamp": datetime.now().isoformat()
    }
    
    # 保存実行
    success = manager.log_conversation(
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
    
    return success

def send_prompt_to_model_comparison(prompt_text: str, user_note: str = None) -> bool:
    """model比較シートにプロンプトを送信（Streamlit用）"""
    
    manager = get_sheets_manager()
    
    if not manager.is_connected:
        return False
    
    return manager.send_to_model_comparison(prompt_text, user_note)

# 接続テスト用
def test_connection():
    """接続テスト"""
    print("🔧 Google Sheets接続テスト開始...")
    
    try:
        manager = SheetsManager()
        
        if manager.is_connected:
            print("✅ 接続成功")
            
            # conversationsシートテスト
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
                print("✅ conversationsシート書き込み成功")
            else:
                print("❌ conversationsシート書き込み失敗")
            
            # model比較シートテスト
            model_success = manager.send_to_model_comparison(
                "これはテストプロンプトです。複数のLLMモデルで比較してください。",
                "テスト実行"
            )
            
            if model_success:
                print("✅ model比較シート書き込み成功")
            else:
                print("❌ model比較シート書き込み失敗")
                
        else:
            print("❌ 接続失敗")
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")

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
    
    # Step 4: conversationsシート書き込みテスト
    st.write("### Step 4: conversationsシート書き込みテスト")
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
                chat_title="デバッグチャット",
                metadata={"debug": True}
            )
            
            if success:
                st.success("✅ conversationsシート書き込みテスト成功")
            else:
                st.error("❌ conversationsシート書き込みテスト失敗")
        else:
            st.error("❌ マネージャー接続失敗")
            
    except Exception as e:
        st.error(f"❌ conversationsシート書き込みテスト例外: {e}")
    
    # Step 5: model比較シート書き込みテスト
    st.write("### Step 5: model比較シート書き込みテスト")
    try:
        manager = get_sheets_manager()
        if manager.is_connected:
            test_prompt = f"デバッグテストプロンプト - {datetime.now().strftime('%H:%M:%S')}"
            success = manager.send_to_model_comparison(test_prompt, "Streamlit デバッグテスト")
            
            if success:
                st.success("✅ model比較シート書き込みテスト成功")
                st.info("Google Sheetsの「model比較」シートを確認してください")
            else:
                st.error("❌ model比較シート書き込みテスト失敗")
        else:
            st.error("❌ マネージャー接続失敗")
            
    except Exception as e:
        st.error(f"❌ model比較シート書き込みテスト例外: {e}")
    
    st.write("### 🔧 診断完了")

# 追加: 簡単な接続状態確認関数
def check_connection_status():
    """簡単な接続状態チェック"""
    try:
        manager = get_sheets_manager()
        return manager.is_connected
    except:
        return False

if __name__ == "__main__":
    test_connection()