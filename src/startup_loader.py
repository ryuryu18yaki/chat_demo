import streamlit as st
from pathlib import Path
from src.rag_preprocess import preprocess_files
from src.rag_vector import save_docs_to_chroma
import hashlib

def initialize_chroma_from_input_debug(input_dir: str, persist_dir: str | None, collection_name: str = "default") -> dict:
    """
    ベクトルDBを構築し、collection と PDFファイル群を返す（Streamlit表示版）
    """
    debug_messages = []  # デバッグメッセージを蓄積
    
    debug_messages.append(f"🔍 RAG初期化開始 - input_dir={input_dir}")
    
    input_path = Path(input_dir)
    files = list(input_path.glob("**/*.*"))
    
    debug_messages.append(f"📁 発見ファイル数: {len(files)}")
    for f in files:
        debug_messages.append(f"  - {f.name} (サイズ: {f.stat().st_size} bytes)")

    # PDFを読み込んで doc 化
    file_dicts = []
    for f in files:
        file_dict = {
            "name": f.name,
            "type": "application/pdf",
            "size": f.stat().st_size,
            "data": f.read_bytes()
        }
        file_dicts.append(file_dict)
        
        # ファイルのハッシュを計算して重複チェック
        file_hash = hashlib.md5(file_dict["data"]).hexdigest()
        debug_messages.append(f"📄 ファイル: {f.name}, ハッシュ: {file_hash[:12]}")

    debug_messages.append("📚 preprocess_files開始...")
    docs = preprocess_files(file_dicts)
    debug_messages.append(f"📚 preprocess_files完了 - 生成ドキュメント数: {len(docs)}")
    
    # 生成されたドキュメントの分析
    doc_analysis = {}
    content_hashes = {}
    duplicate_contents = []
    
    for i, doc in enumerate(docs):
        # メタデータ分析
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        source = metadata.get('source', 'unknown')
        page = metadata.get('page', 'unknown')
        kind = metadata.get('kind', 'unknown')
        
        key = f"{source}_p{page}_{kind}"
        doc_analysis[key] = doc_analysis.get(key, 0) + 1
        
        # コンテンツ重複チェック
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in content_hashes:
            duplicate_contents.append({
                'hash': content_hash,
                'original_index': content_hashes[content_hash],
                'duplicate_index': i,
                'content_preview': content[:100]
            })
            debug_messages.append(f"⚠️ 重複コンテンツ検出 - ハッシュ: {content_hash[:12]}, インデックス: {content_hashes[content_hash]} -> {i}")
        else:
            content_hashes[content_hash] = i
    
    # 分析結果を蓄積
    debug_messages.append("📊 ドキュメント分析結果:")
    for key, count in doc_analysis.items():
        if count > 1:
            debug_messages.append(f"  ⚠️ {key}: {count}個 (重複の可能性)")
        else:
            debug_messages.append(f"  ✅ {key}: {count}個")
    
    if duplicate_contents:
        debug_messages.append(f"❌ コンテンツ重複を {len(duplicate_contents)} 件検出!")
        for dup in duplicate_contents[:3]:  # 最初の3件
            debug_messages.append(f"  - ハッシュ {dup['hash'][:12]}: {dup['original_index']} -> {dup['duplicate_index']}")
    else:
        debug_messages.append("✅ コンテンツ重複は検出されませんでした")

    debug_messages.append("💾 save_docs_to_chroma開始...")
    collection = save_docs_to_chroma(
        docs=docs, 
        collection_name=collection_name, 
        persist_directory=persist_dir
    )
    debug_messages.append("💾 save_docs_to_chroma完了")
    
    # 最終的なコレクション統計
    final_count = collection.count()
    debug_messages.append(f"🎯 最終結果 - 入力docs: {len(docs)}, DBチャンク数: {final_count}")
    
    if len(docs) != final_count:
        debug_messages.append("⚠️ 入力ドキュメント数とDB保存数が一致しません!")

    return {
        "collection": collection,
        "rag_files": file_dicts,
        "debug_info": {
            "input_docs": len(docs),
            "final_chunks": final_count,
            "duplicate_contents": len(duplicate_contents),
            "doc_analysis": doc_analysis,
            "debug_messages": debug_messages  # デバッグメッセージも保存
        }
    }

# Streamlitサイドバー用のシンプルなデバッグパネル
def render_debug_panel():
    """Streamlitサイドバーにデバッグパネルをレンダリング"""
    
    st.markdown("### 🔬 RAGデバッグツール")
    
    # デバッグ情報表示
    if st.button("📋 初期化ログを表示"):
        if st.session_state.get("debug_info") and st.session_state.debug_info.get("debug_messages"):
            messages = st.session_state.debug_info["debug_messages"]
            
            # メッセージをStreamlitで表示
            st.markdown("#### 🔍 初期化ログ")
            
            log_container = st.container()
            with log_container:
                for msg in messages:
                    if "⚠️" in msg or "❌" in msg:
                        st.warning(msg)
                    elif "✅" in msg:
                        st.success(msg)
                    else:
                        st.info(msg)
                        
            # 統計表示
            debug_info = st.session_state.debug_info
            st.markdown("#### 📊 統計サマリー")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("入力ドキュメント", debug_info.get("input_docs", "N/A"))
            with col2:
                st.metric("最終チャンク数", debug_info.get("final_chunks", "N/A"))
            with col3:
                st.metric("重複コンテンツ", debug_info.get("duplicate_contents", "N/A"))
                
        else:
            st.warning("デバッグ情報がありません")
    
    # 簡単な診断
    if st.button("🔍 簡単診断"):
        if st.session_state.get("rag_collection"):
            collection = st.session_state.rag_collection
            
            try:
                # 基本統計
                total_count = collection.count()
                st.info(f"📊 総チャンク数: {total_count}")
                
                # サンプル検索で重複チェック
                if total_count > 0:
                    all_data = collection.get(limit=10)  # 最初の10件だけ取得
                    
                    documents = all_data.get('documents', [])
                    metadatas = all_data.get('metadatas', [])
                    ids = all_data.get('ids', [])
                    
                    if documents:
                        st.success(f"✅ サンプル取得成功: {len(documents)}件")
                        
                        # メタデータ分析
                        source_page_analysis = {}
                        for meta in metadatas:
                            if meta:
                                source = meta.get('source', 'unknown')
                                page = meta.get('page', 'unknown')
                                kind = meta.get('kind', 'unknown')
                                
                                key = f"{source}_p{page}_{kind}"
                                source_page_analysis[key] = source_page_analysis.get(key, 0) + 1
                        
                        st.markdown("**サンプル分析:**")
                        for key, count in source_page_analysis.items():
                            if count > 1:
                                st.warning(f"⚠️ {key}: {count}件")
                            else:
                                st.success(f"✅ {key}: {count}件")
                    
                    # ID重複チェック
                    if ids:
                        unique_ids = len(set(ids))
                        total_ids = len(ids)
                        
                        if unique_ids != total_ids:
                            st.error(f"❌ ID重複検出: 総数{total_ids}, ユニーク{unique_ids}")
                        else:
                            st.success(f"✅ ID重複なし: {unique_ids}件")
                            
            except Exception as e:
                st.error(f"診断中にエラー: {e}")
                
        else:
            st.warning("RAGコレクションが初期化されていません")
    
    # 再初期化（デバッグ版）
    if st.button("🔄 デバッグ版で再初期化"):
        with st.spinner("デバッグ版で再初期化中..."):
            try:
                # セッション状態をクリア
                st.session_state.rag_collection = None
                st.session_state.rag_files = []
                st.session_state.last_rag_sources = []
                st.session_state.last_rag_images = []
                
                # デバッグ版で初期化
                res = initialize_chroma_from_input_debug(
                    input_dir="rag_data",
                    persist_dir=None,
                    collection_name="session_docs"
                )
                
                st.session_state.rag_collection = res["collection"]
                st.session_state.rag_files = res["rag_files"] 
                st.session_state.debug_info = res.get("debug_info", {})
                
                st.success("✅ デバッグ版で再初期化完了！")
                st.info("「📋 初期化ログを表示」で詳細を確認してください")
                
            except Exception as e:
                st.error(f"再初期化エラー: {e}")
                import traceback
                st.text(traceback.format_exc())

# メインアプリの修正
# startup_loader.pyの関数を置き換える場合：
def initialize_chroma_from_input(input_dir: str, persist_dir: str | None, collection_name: str = "default") -> dict:
    """
    元の関数をデバッグ版に置き換え
    """
    return initialize_chroma_from_input_debug(input_dir, persist_dir, collection_name)