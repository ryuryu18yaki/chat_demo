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
            scopes = [
                    'https://www.googleapis.com/auth/drive',  # フルアクセス
                    'https://www.googleapis.com/auth/drive.file',  # ファイルアクセス
                    'https://www.googleapis.com/auth/drive.readonly'  # 読み取り専用（念のため）
                ]
        )
        service = build('drive', 'v3', credentials=creds)
        logger.info("🔍 認証成功")
        
        # フォルダ情報取得をスキップして直接ファイル検索
        logger.info("🔍 フォルダ内ファイル検索開始")
        
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
            
            logger.info("🔍 処理中: %s", file_name)
            
            # PDFとテキストファイルのみ処理
            if not (file_name.lower().endswith('.pdf') or file_name.lower().endswith('.txt')):
                logger.info("⏭️ スキップ: %s (拡張子: %s)", file_name, 
                           file_name.split('.')[-1] if '.' in file_name else 'なし')
                continue
            
            logger.info("📄 ダウンロード開始: %s", file_name)
            
            # ファイルダウンロード
            request = service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            file_data = file_io.getvalue()
            logger.info("📄 ダウンロード完了: %s (%d bytes)", file_name, len(file_data))
            
            # 設備名を推定
            from src.equipment_classifier import extract_equipment_from_filename, get_equipment_category
            equipment_name = extract_equipment_from_filename(file_name)
            equipment_category = get_equipment_category(equipment_name)
            
            # 既存形式に合わせる
            file_dict = {
                "name": file_name,
                "type": mime_type,
                "size": len(file_data),
                "data": file_data,
                "equipment_name": equipment_name,
                "equipment_category": equipment_category
            }
            
            file_dicts.append(file_dict)
            logger.info("✅ 完了: %s → 設備: %s", file_name, equipment_name)
        
        logger.info("📊 Google Drive読み込み完了: %dファイル", len(file_dicts))
        return file_dicts
        
    except Exception as e:
        logger.error("❌ Google Drive読み込みエラー: %s", e, exc_info=True)
        return []