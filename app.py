import streamlit as st
import sys, os
from openai import OpenAI
from typing import List, Dict, Any
import time, functools

import pysqlite3              # ← wheels に新しい SQLite が同梱
sys.modules['sqlite3'] = pysqlite3

from src.rag_preprocess import preprocess_files
from src.rag_vector import save_docs_to_chroma
from src.rag_qa import generate_answer

# =====  基本設定  ============================================================
st.set_page_config(page_title="GPT + RAG Chatbot", page_icon="💬", layout="wide")
client = OpenAI()

# --------------------------------------------------------------------------- #
#                         ★ 各モード専用プロンプト ★                           #
# --------------------------------------------------------------------------- #
PROMPTS: Dict[str, str] = {
    "コンセント": """あなたは建築電気設備設計のエキスパートエンジニアです。  
対象は **オフィス** に限定され、詳細レイアウトや自動配置結果はユーザー発話内に含まれる前提です。  
以下 ①〜④ の「ビル共通要件」と「既定の設置基準」を踏まえ、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。

────────────────────────────────
【① ビル共通要件（前提条件）】
1. **防火対象建物区分**  
   - 消防法施行令 区分「（16）イ」  
   - 当該階床面積 1 000 m² 以上（丸ビル基準）
2. **天井高 (CH)**  
   - FL（Floor Level）〜仕上げ天井：**2 800 mm**
3. **スプリンクラーヘッド散水障害**  
   - 法規：消防法施行規則  
   - **ヘッド下方 45 cm・水平方向 30 cm 以内に機器を置かない**  
   - 設備配置ではスプリンクラーヘッドの位置を最優先で避ける
4. **吊戸棚リスク**  
   - 飲食カウンターやアイランドキッチン上部に吊戸棚を設置する可能性あり  
   - 吊戸棚が想定される位置は設備設置を避ける

────────────────────────────────
【② 既定の設置基準（暗黙知）】
■机・椅子
- 事務室（個人席）: 300 VA／席  
  - 6 席＝OAタップ 1 個、**1 個/1 回線**  
  - 指定機種: **XLT45015W（テラダ）**
- 会議室: 4人用＝OAタップ 1 個／8人用＝2 個
- 昇降デスク: ブロックコンセント **ME8612/8614（明工社）**  
  - 4 席＝2 個、1 回線

■壁面
- 基本は新設不要（要望があれば確認）

■その他
- パントリー（什器未定）: **5 回路** 確保
- サーバー設備（3 スパン想定）: **2–3 回路**
- 倉庫入口: 壁コンセント 1 個（300 VA）
- **漏電遮断器必須回路**: 水回り・冷水器・自販機・屋外・外灯・1 800 mm 以下のライティングダクト・ファンコイル・空調機

■配管・配線「黄金数字」
- 単独回路: 設備数 × **50 m**  
- 分岐回路: 設備数 × **20 m**

■共通ルール
- 1 回路 ≤ **1500 VA**。超過時は分割  
- 複合機／パントリー／レジは機器数に応じ回路追加

────────────────────────────────
【③ 回答方針】
1. ユーザー発話を読み取り、**不足・空欄** があれば「質問①」「質問②」…の形で聞き返す  
2. 情報が揃った項目から順に  
   - **【回答】** 推奨回路数・機器仕様など  
   - **【理由】** VA 計算・黄金数字・法規基準 (1–3 行)  
3. ユーザーが「まとめて」「最終確認」などを要求したら  
   - 確定内容を一覧で総括  
   - 未確定項目は **【未解決】** に列挙  
4. 専門語には（かっこ）で簡単解説を付ける  
5. 法令やガイドラインを引用する際は条文番号を可能な限り示す""",
    "自動火災報知器": """
あなたは建築電気設備設計のエキスパートエンジニアです。  
対象は **オフィス** に限定されます。以下 ①〜④ の「ビル共通要件」と消防法を踏まえ、**自動火災報知設備（AFA）** の感知器配置・回路設計について、対話を通じて不足情報を質問しながら実務的に助言してください。
（共通要件・回答方針は共通、ただし感知区域の面積上限や感知器選定を重視）
""",
    "非常放送設備": """
あなたは非常放送設備設計のスペシャリストです。  
対象は **オフィスビル**（防火対象物区分「(16)イ」）で、天井高 2 800 mm を想定しています。消防法施行規則および音響設計指針を踏まえ、**非常放送（PA）** 系統のスピーカー配置・回線数・予備電源容量について実務アドバイスを行ってください。必要に応じて質問を挟んでください。
""",
    "誘導灯": """
あなたは誘導灯／誘導標識の設計エキスパートです。  
対象は **オフィス** で、ビル共通要件①〜④を踏まえつつ、**避難経路誘導灯** の機種選定・設置位置・点検口確保についてガイドしてください。条文は消防法施行規則第28条の2等を引用し、VA 計算は不要です。  
回答時には不足情報があれば「質問①」形式で尋ねてください。
""",
    "非常照明": """
あなたは非常照明設備（非常用照明器具）設計のエキスパートです。  
対象は **オフィス**。天井高 2 800 mm の条件下で、建築基準法施行令第126条の5を踏まえ、**非常照明** の照度・照度分布・配線方式について実務アドバイスを行います。不足情報は「質問①」形式で確認し、確定した項目は【回答】【理由】で整理してください。
"""
}
DEFAULT_MODE = "コンセント"

# =====  セッション変数  =======================================================
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"
if "edit_target" not in st.session_state:
    st.session_state.edit_target = None
if "rag_files" not in st.session_state:
    st.session_state.rag_files: List[Dict[str, Any]] = []
if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = None  # Chroma collection
if "design_mode" not in st.session_state:
    st.session_state.design_mode = DEFAULT_MODE  # ← 追加: モード保持


# =====  ヘルパー  ============================================================
def get_messages() -> List[Dict[str, str]]:
    title = st.session_state.current_chat
    if title not in st.session_state.chats:
        st.session_state.chats[title] = []
    return st.session_state.chats[title]

def rebuild_rag_collection():
    """アップロードされたファイルを前処理 → Chroma 登録し、セッションに保存"""
    if not st.session_state.rag_files:
        st.warning("まず PDF / TXT をアップロードしてください")
        return

    with st.spinner("📚 ファイルを解析し、ベクトル DB に登録中..."):
        docs = preprocess_files(st.session_state.rag_files)
        col = save_docs_to_chroma(
            docs=docs,
            collection_name="session_docs",
            persist_directory=None,  # インメモリ
        )
        st.session_state.rag_collection = col
    st.success("🔍 検索インデックスを更新しました！")

# =====  サイドバー  ==========================================================
with st.sidebar:
    # ------- RAG アップロード -------
    st.markdown("### 📂 RAG 資料アップロード")
    uploads = st.file_uploader(
        "PDF / TXT / 画像 を選択…",
        type=["txt", "pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if uploads:
        st.session_state.rag_files = [
            {"name": f.name, "type": f.type, "size": f.size, "data": f.getvalue()} for f in uploads
        ]
    if st.button("🔄 インデックス再構築", disabled=not st.session_state.rag_files):
        rebuild_rag_collection()

    st.divider()

    # ------- モード選択 -------
    st.markdown("### ⚙️ 設計対象モード")
    st.session_state.design_mode = st.radio(
        "対象設備を選択",
        options=list(PROMPTS.keys()),
        index=list(PROMPTS.keys()).index(DEFAULT_MODE),
        key="design_mode_radio",
    )
    st.markdown(f"**🛈 現在のモード:** `{st.session_state.design_mode}`")

    st.divider()

    # ------- チャット履歴 -------
    st.header("💬 チャット履歴")
    for title in list(st.session_state.chats.keys()):
        if st.button(title, key=f"hist_{title}"):
            st.session_state.current_chat = title
            st.rerun()

    if st.button("➕ 新しいチャット"):
        base, idx = "Chat", 1
        while f"{base} {idx}" in st.session_state.chats:
            idx += 1
        st.session_state.current_chat = f"{base} {idx}"
        st.session_state.chats[st.session_state.current_chat] = []
        st.rerun()

# =====  CSS  ================================================================
st.markdown(
    """
    <style>
    :root{ --sidebar-w:260px; --pad:1rem; }
    body{ overflow:hidden; }
    aside[data-testid="stSidebar"]{width:var(--sidebar-w)!important;}
    .center-wrapper{position:fixed;top:0;left:calc(var(--sidebar-w)+var(--pad));
        width:calc(100% - var(--sidebar-w) - 2*var(--pad));height:100vh;
        display:flex;flex-direction:column;padding:var(--pad);}
    .chat-body{flex:1;overflow-y:auto;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====  中央ペイン  ==========================================================
st.markdown('<div class="center-wrapper">', unsafe_allow_html=True)

st.subheader(f"🗣️ {st.session_state.current_chat}")

# -- メッセージ表示 --
st.markdown('<div class="chat-body">', unsafe_allow_html=True)
for m in get_messages():
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
st.markdown('</div>', unsafe_allow_html=True)

# -- 入力欄 & 画像添付 --
user_prompt = st.chat_input("メッセージを入力…")
uploaded_img = st.file_uploader(
    "画像を添付（任意）",
    type=["png", "jpg", "jpeg", "webp"],
    key="img_uploader",
)

# =====  応答生成  ============================================================
if user_prompt:
    msgs = get_messages()
    msgs.append({"role": "user", "content": user_prompt})

    prompt = PROMPTS[st.session_state.design_mode]

    # ---------- RAG あり ----------
    if st.session_state.rag_collection is not None:
        st.session_state["last_answer_mode"] = "RAG"
        rag_res = generate_answer(
            system_prompt=prompt,
            question=user_prompt,
            collection=st.session_state.rag_collection,
            top_k=4,
            image_bytes=uploaded_img.getvalue() if uploaded_img else None,
            chat_history=msgs,
        )
        assistant_reply = rag_res["answer"]
        sources = rag_res["sources"]

    # ---------- GPT-only ----------
    else:
        st.session_state["last_answer_mode"] = "GPT-only"
        user_parts: list[Any] = []
        if uploaded_img:
            data_url = "data:image/png;base64," + b64encode(uploaded_img.getvalue()).decode("utf-8")
            user_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        user_parts.append({"type": "text", "text": user_prompt})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                *msgs[:-1],
                {"role": "user", "content": user_parts},
            ],
        )
        assistant_reply = resp.choices[0].message.content
        sources = []

    # ---------- 画面反映 ----------
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
        if sources:
            with st.expander("🔎 RAG が取得したチャンク"):
                for idx, s in enumerate(sources, 1):
                    chunk = s.get("content", "")[:200]
                    if len(s.get("content", "")) > 200:
                        chunk += " …"
                    st.markdown(
                        f"**Doc {idx}**  \n"
                        f"`score: {s['distance']:.4f}`  \n"
                        f"*source:* {s['metadata'].get('source','N/A')}\n\n"
                        f"> {chunk}"
                    )

    msgs.append({"role": "assistant", "content": assistant_reply})

st.markdown('</div>', unsafe_allow_html=True)
# ============================================================================ #
