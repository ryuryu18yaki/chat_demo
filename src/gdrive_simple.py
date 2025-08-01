import os
import io, mimetypes
from typing import List, Dict, Any, Optional
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.equipment_classifier import extract_equipment_from_filename, get_equipment_category
from src.logging_utils import init_logger
logger = init_logger()

def download_files_from_drive(folder_id: str) -> List[Dict[str, Any]]:
    logger.info("🔍 Google Drive開始: フォルダID = %s", folder_id)
    
    try:
        # 認証
        logger.info("🔍 認証開始")
        credentials_info = st.secrets["gcp_service_account"]
        
        scopes = [
            'https://www.googleapis.com/auth/drive.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = Credentials.from_service_account_info(
            credentials_info, 
            scopes=scopes
        )
        service = build('drive', 'v3', credentials=creds)
        logger.info("🔍 認証成功")
        
        # フォルダ内のファイル一覧を取得
        logger.info("🔍 フォルダ内ファイル検索中...")
        query = f"'{folder_id}' in parents and trashed=false"
        
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, size)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        files = results.get('files', [])
        logger.info("🔍 検索結果: %d個のファイル", len(files))
        
        # ファイルダウンロードと分類処理
        file_dicts = []

        for file_info in files:
            file_name = file_info["name"]
            file_id   = file_info["id"]
            mime_type = file_info["mimeType"]
            file_size = file_info.get("size", 0)

            # PDF / TXT / JSON だけ対象
            if not file_name.lower().endswith((".pdf", ".txt", ".json")):
                continue

            logger.info("⬇️ ダウンロード開始: %s", file_name)

            fh = io.BytesIO()
            request = service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.debug("   %.1f%%", status.progress() * 100)

            fh.seek(0)
            file_data = fh.read()
            
            # ファイル名から設備名を抽出
            equipment_name = extract_equipment_from_filename(file_name)
            equipment_category = get_equipment_category(equipment_name)
            
            # 三菱地所が含まれる場合は名前を変更
            display_name = file_name
            if "170301" in file_name:
                display_name = "新暗黙知メモ.pdf"
            elif "001-取扱い注意_改修工事図面作成要領(案)H11年3月三菱地所㈱リニューアル建築部_OCR済み" in file_name:
                display_name = "旧暗黙知メモ.pdf"
            
            file_dicts.append({
                "name": display_name,
                "type": mime_type,
                "size": file_size,
                "data": file_data,
                "equipment_name": equipment_name,
                "equipment_category": equipment_category
            })
            logger.info("✅ 取得完了: %s (%d bytes) → 設備: %s (カテゴリ: %s)", 
                       file_name, len(file_data), equipment_name, equipment_category)
        
        logger.info("📊 Google Drive読み込み完了: %dファイル", len(file_dicts))
        return file_dicts
        
    except Exception as e:
        logger.error("❌ Google Drive読み込みエラー: %s", e, exc_info=True)
        return []

def download_fix_files_from_drive(fixes_folder_id: str) -> Dict[str, bytes]:
    """
    Google Drive内の fixes フォルダから補正用ファイル（json/yaml/txtなど）を取得する。

    Args:
        fixes_folder_id (str): Google Drive の fixes フォルダID

    Returns:
        Dict[str, bytes]: ファイル名をキーとし、内容をバイナリで保持する辞書
    """
    logger.info("🔧 Fixesフォルダ読み込み開始: フォルダID = %s", fixes_folder_id)

    try:
        # --- 認証 ---
        credentials_info = st.secrets["gcp_service_account"]
        scopes = ['https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        service = build('drive', 'v3', credentials=creds)
        logger.info("🔧 認証成功")

        # --- Fixes フォルダ内のファイル一覧を取得 ---
        query = f"'{fixes_folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()

        files = results.get("files", [])
        logger.info("📦 Fixes内ファイル数: %d", len(files))

        fix_files: Dict[str, bytes] = {}
        for file_info in files:
            file_name = file_info["name"]
            file_id = file_info["id"]
            
            if not file_name.lower().endswith((".json", ".txt", ".yaml", ".yml", ".png")):
                logger.debug("📁 スキップ（対象外）: %s", file_name)
                continue

            logger.info("⬇️ 補正ファイルダウンロード: %s", file_name)
            try:
                fh = io.BytesIO()
                request = service.files().get_media(fileId=file_id)
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logger.debug("   %.1f%%", status.progress() * 100)

                fh.seek(0)
                fix_files[file_name] = fh.read()
                logger.info("✅ ダウンロード成功: %s (%d bytes)", file_name, len(fix_files[file_name]))
            except Exception as e:
                logger.warning("⚠️ ダウンロード失敗: %s - %s", file_name, e)

        logger.info("🎯 Fixesフォルダ読み込み完了: 有効ファイル数 = %d", len(fix_files))
        return fix_files

    except Exception as e:
        logger.error("❌ Fixesフォルダ処理エラー: %s", e, exc_info=True)
        return {}