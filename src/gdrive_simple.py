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
        
        # 🔥 より広いスコープを試す
        scopes = [
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = Credentials.from_service_account_info(
            credentials_info, 
            scopes=scopes  # より広いスコープ
        )
        service = build('drive', 'v3', credentials=creds)
        logger.info("🔍 認証成功")
        
        # 🔥 まず自分がアクセスできるファイル一覧を取得（デバッグ用）
        logger.info("🔍 アクセス可能なファイル一覧を取得中...")
        try:
            test_results = service.files().list(
                pageSize=10,
                fields="files(id, name, parents, owners)",
                supportsAllDrives=True
            ).execute()
            
            test_files = test_results.get('files', [])
            logger.info("🔍 アクセス可能なファイル総数: %d", len(test_files))
            
            for i, f in enumerate(test_files[:5]):  # 最初の5個だけ表示
                logger.info("🔍 アクセス可能ファイル[%d]: %s (ID: %s)", i+1, f.get('name'), f.get('id'))
                if 'parents' in f:
                    logger.info("🔍   親フォルダ: %s", f.get('parents'))
        except Exception as e:
            logger.error("❌ アクセス可能ファイル一覧取得エラー: %s", e)
        
        # 🔥 対象フォルダ情報を詳しく取得
        logger.info("🔍 対象フォルダの詳細情報取得中...")
        try:
            folder_info = service.files().get(
                fileId=folder_id,
                fields="id, name, mimeType, permissions, owners",
                supportsAllDrives=True,       # ★必須②-1
            ).execute()
            logger.info("🔍 フォルダ名: %s", folder_info.get('name'))
            logger.info("🔍 フォルダMIME: %s", folder_info.get('mimeType'))
            logger.info("🔍 フォルダ所有者: %s", folder_info.get('owners'))
        except Exception as e:
            logger.error("❌ フォルダ詳細取得エラー: %s", e)
        
        # 🔥 異なる検索クエリを試す
        logger.info("🔍 複数の検索クエリを試行中...")
        
        # クエリ1: 基本クエリ
        query1 = f"'{folder_id}' in parents and trashed=false"
        logger.info("🔍 クエリ1: %s", query1)
        
        results1 = service.files().list(
            q=query1,
            fields="files(id, name, mimeType, size, parents, owners)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files1 = results1.get('files', [])
        logger.info("🔍 クエリ1結果: %d個", len(files1))
        
        # クエリ2: trashedも含める
        query2 = f"'{folder_id}' in parents"
        logger.info("🔍 クエリ2: %s", query2)
        
        results2 = service.files().list(
            q=query2,
            fields="files(id, name, mimeType, size, trashed)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files2 = results2.get('files', [])
        logger.info("🔍 クエリ2結果: %d個", len(files2))
        
        # 結果の詳細表示
        if files1:
            logger.info("🔍 クエリ1の詳細結果:")
            for i, f in enumerate(files1):
                logger.info("🔍   [%d] %s (MIME: %s, Size: %s)", 
                           i+1, f.get('name'), f.get('mimeType'), f.get('size'))
        
        if files2:
            logger.info("🔍 クエリ2の詳細結果:")
            for i, f in enumerate(files2):
                logger.info("🔍   [%d] %s (MIME: %s, Trashed: %s)", 
                           i+1, f.get('name'), f.get('mimeType'), f.get('trashed'))
        
        # 元の処理を継続
        files = files1  # 基本クエリの結果を使用
        
        # 以下は既存のコードと同じ...
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
            
            # 以下既存のダウンロード処理...
        
        logger.info("📊 Google Drive読み込み完了: %dファイル", len(file_dicts))
        return file_dicts
        
    except Exception as e:
        logger.error("❌ Google Drive読み込みエラー: %s", e, exc_info=True)
        return []