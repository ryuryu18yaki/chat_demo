import os
import io
from typing import List, Dict, Any, Optional
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def download_files_from_drive(folder_id: str) -> List[Dict[str, Any]]:
    """
    Google Driveフォルダからファイルをダウンロードして、
    既存のpreprocess_files形式のリストを返す
    """
    try:
        # 認証
        credentials_info = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            credentials_info, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        
        # フォルダ内のファイル一覧取得
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        print(f"📁 Google Drive: {len(files)}個のファイルを発見")
        
        file_dicts = []
        
        for file_info in files:
            file_name = file_info['name']
            file_id = file_info['id']
            mime_type = file_info['mimeType']
            
            # PDFとテキストファイルのみ処理
            if not (file_name.lower().endswith('.pdf') or file_name.lower().endswith('.txt')):
                continue
            
            print(f"📄 ダウンロード中: {file_name}")
            
            # ファイルダウンロード
            request = service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            file_data = file_io.getvalue()
            
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
            print(f"✅ 完了: {file_name} → 設備: {equipment_name}")
        
        print(f"📊 Google Drive読み込み完了: {len(file_dicts)}ファイル")
        return file_dicts
        
    except Exception as e:
        print(f"❌ Google Drive読み込みエラー: {e}")
        return []