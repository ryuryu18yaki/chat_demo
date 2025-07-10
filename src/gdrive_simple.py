import os
import io
from typing import List, Dict, Any, Optional
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.logging_utils import init_logger
logger = init_logger()

def download_files_from_drive(folder_id: str) -> List[Dict[str, Any]]:
    logger.info("🔍 Google Drive開始: フォルダID = %s", folder_id)
    
    try:
        # 認証
        logger.info("🔍 認証開始")
        credentials_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            credentials_info, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        logger.info("🔍 認証成功")
        
        # フォルダ情報取得
        try:
            folder_info = service.files().get(fileId=folder_id).execute()
            logger.info("🔍 フォルダ名: %s", folder_info.get('name'))
        except Exception as e:
            logger.error("❌ フォルダアクセスエラー: %s", e)
            return []
        
        # ファイル一覧取得
        query = f"'{folder_id}' in parents and trashed=false"
        logger.info("🔍 検索クエリ: %s", query)
        
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        logger.info("🔍 発見したアイテム数: %d", len(files))
        
        # 全アイテムの詳細表示
        for i, file_info in enumerate(files):
            logger.info("🔍 [%d] 名前: %s", i+1, file_info['name'])
            logger.info("🔍 [%d] MIME: %s", i+1, file_info['mimeType'])
        
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