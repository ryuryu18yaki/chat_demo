import os
import io
from typing import List, Dict, Any, Optional
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def download_files_from_drive(folder_id: str) -> List[Dict[str, Any]]:
    try:
        print(f"🔍 開始: フォルダID = {folder_id}")
        
        # 認証
        credentials_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            credentials_info, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        print("🔍 認証成功")
        
        # フォルダ情報取得
        try:
            folder_info = service.files().get(fileId=folder_id).execute()
            print(f"🔍 フォルダ名: {folder_info.get('name')}")
        except Exception as e:
            print(f"❌ フォルダアクセスエラー: {e}")
            return []
        
        # フォルダ内のファイル一覧取得
        query = f"'{folder_id}' in parents and trashed=false"
        print(f"🔍 検索クエリ: {query}")
        
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        print(f"🔍 発見したアイテム数: {len(files)}")
        
        # 全アイテムの詳細表示
        for i, file_info in enumerate(files):
            print(f"🔍 [{i+1}] 名前: {file_info['name']}")
            print(f"🔍 [{i+1}] MIME: {file_info['mimeType']}")
            print(f"🔍 [{i+1}] サイズ: {file_info.get('size', 'N/A')}")
            print("---")
        
        file_dicts = []
        
        for file_info in files:
            file_name = file_info['name']
            file_id = file_info['id']
            mime_type = file_info['mimeType']
            
            print(f"🔍 処理中: {file_name}")
            
            # PDFとテキストファイルのみ処理
            if not (file_name.lower().endswith('.pdf') or file_name.lower().endswith('.txt')):
                print(f"⏭️ スキップ: {file_name} (拡張子: {file_name.split('.')[-1] if '.' in file_name else 'なし'})")
                continue
            
            print(f"📄 ダウンロード開始: {file_name}")
            
            # ファイルダウンロード
            request = service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            file_data = file_io.getvalue()
            print(f"📄 ダウンロード完了: {file_name} ({len(file_data)} bytes)")
            
            # 以下、既存のコードと同じ...
        
        print(f"📊 Google Drive読み込み完了: {len(file_dicts)}ファイル")
        return file_dicts
        
    except Exception as e:
        print(f"❌ Google Drive読み込みエラー: {e}")
        return []