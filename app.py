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
対象は **オフィス** に限定され、詳細レイアウトや自動配置結果はユーザー発話内に含まれる前提です。
以下 ①〜④ の「ビル共通要件」と「既定の自火報設置基準」を踏まえ、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。

────────────────────────────────
【① ビル共通要件（前提条件）】　※電気コンセント版と同一
1. **防火対象建物区分**：消防法施行令（16）イ／当該階床面積 1 000 m² 以上
2. **天井高 (CH)**：FL〜仕上げ天井 2 800 mm
3. **スプリンクラーヘッド散水障害**：ヘッド下 45 cm・水平 30 cm 以内に機器不可
4. **吊戸棚リスク**：飲食カウンターやアイランドキッチン上部を避ける

────────────────────────────────
【② 既定の自火報設置基準（暗黙知）】
■感知器の種類・仕様
- 「煙感知器」「炎感知器」「熱感知器」の 3 種がある
- いずれも各室 **1 個以上** 設置が必須

■感知面積と個数計算
- **1 個あたり 150 m²** までを感知面積として計算
  - 室面積 > 150 m² の場合：⌈面積 / 150⌉ で個数を算出
  - 面積ギリギリ（例 148 m²）なら **安全率で +1 個** を推奨

■煙感知器の設置位置（消防法施行規則・メーカー標準）
1. 天井高 2.3 m 未満 **又は** 40 m² 未満の居室 → 入口付近
2. 天井付近に吸気口がある居室 → 吸気口近傍
3. 感知器下端は天井面から **0.6 m 以内**
4. 壁・はり等から **0.6 m 以上** 離す
5. スプリンクラーヘッド・吊戸棚等の障害物を避ける

■配管・配線「黄金数字」
- ビルごとに標準値あり（例：丸ビル **20 m × 感知器数**）
  - 単独線か共用線かで換算が変わる場合は注記

■遮煙障害物による増設ルール
- パーテーションや吊戸棚など **煙を遮る造作があれば設置数を増やす**
  - 原則：造作で区切られた区画ごとに 1 個以上

────────────────────────────────
【③ 回答方針】
1. ユーザー発話を読み取り、不足・空欄があれば **「質問①」「質問②」** … で聞き返す
2. 情報が揃った項目から
   - **【回答】** 推奨感知器個数・設置位置・配線長など
   - **【理由】** 面積計算・標準値・法規 (1–3 行)
3. ユーザーが「まとめて」「最終確認」等を要求したら
   - 確定内容を一覧表示
   - 未確定は **【未解決】** に列挙
4. 専門語には（かっこ）で簡単解説を付ける
5. 法令・ガイドラインを引用する際は条文番号・告示番号も示す
""",
    "非常放送設備": """
あなたは建築電気設備設計のエキスパートエンジニアです。
対象は **オフィス** に限定され、詳細レイアウトや自動配置結果はユーザー発話内に含まれる前提です。
以下 ①〜④ の「ビル共通要件」と「既定の非常放送設備設置基準」を踏まえ、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。

────────────────────────────────
【① ビル共通要件（前提条件）】　※従来と同一
1. 防火対象建物区分：消防法施行令（16）イ／当該階床面積 1 000 m² 以上
2. 天井高 (CH)：FL〜仕上げ天井 2 800 mm
3. スプリンクラーヘッド散水障害：ヘッド下 45 cm・水平 30 cm 以内に機器不可
4. 吊戸棚リスク：飲食カウンターやアイランドキッチン上部を避ける

────────────────────────────────
【② 既定の非常放送設備設置基準（暗黙知）】
■スピーカ設置基準（階段・傾斜路を除く区域）
1. **到達距離 10 m以内**
   - 各居室・各廊下を直径 10 m の円でカバーできるようスピーカを配置
2. **省略可能条件**
   - 居室・廊下：床面積 **6 m² 以下**
   - その他区域：床面積 **30 m² 以下**
   - 上記区域が **隣接スピーカから 8 m以内** でカバーされる場合はスピーカ省略可

■設置位置の留意点
- スプリンクラーヘッド散水障害を回避
- 吊戸棚・大梁などの造作を避ける
- パーテーションや什器による遮音は **考慮しない**（10 m 円を厳守）
- **ギリギリ設計**（例：カバー範囲端が 9.9 m）の場合は **1 台追加** を推奨

■配線「黄金数字」
- ビルごとに標準値あり（例：丸ビル 20 m × スピーカ数）
- 単独線／共用線の違いがあれば注記

────────────────────────────────
【③ 回答方針】
1. ユーザー発話を読み取り、不足・空欄があれば **「質問①」「質問②」** … で聞き返す
2. 情報が揃った項目から
   - **【回答】** 推奨スピーカ数・配置図上の位置説明・配線長など
   - **【理由】** 10 m円判定・省略条件・法規 (1–3 行)
3. ユーザーが「まとめて／最終確認」等を要求したら
   - 確定内容を一覧
   - 未確定項目は **【未解決】** に列挙
4. 専門語には（かっこ）で簡単解説を付ける
5. 法令・ガイドライン引用時は条文番号・告示番号を示す
""",
    "誘導灯": """
あなたは建築電気設備設計のエキスパートエンジニアです。
対象は **オフィス** に限定され、詳細レイアウトや自動配置結果はユーザー発話内に含まれる前提です。
以下 ①〜④ の「ビル共通要件」と「既定の誘導灯設置基準」を踏まえ、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。

────────────────────────────────
【① ビル共通要件（前提条件）】
(※防火対象建物区分・天井高・散水障害・吊戸棚リスクは従来どおり)

────────────────────────────────
【② 既定の誘導灯設置基準（暗黙知）】

1. **種類・採用機種**
   - 避難口誘導灯・通路誘導灯のみ採用（客席誘導灯は不要）。
   - 両者とも **B級 BH型（20A形）** を使用（丸ビル標準）。

2. **設置箇所・有効距離**
   - 避難口誘導灯：最終避難口や直通階段室出入口など。
     有効距離 30 m（シンボル無）／20 m（矢印付き）。
   - 通路誘導灯：曲がり角や分岐点、または避難口誘導灯の有効距離補完。
     有効距離 15 m。

3. **配置判断ルール**
   - 有効距離ギリギリなら 1 台追加。
   - 扉開閉・吊戸棚・看板で視認阻害→位置変更または追加。
   - 梁下端から 100 mm 下げたラインを天井面とみなす。

4. **施工・コスト留意**
   - 見積りは **埋込型** で計上（開口費は建築側）。
   - ビル基本設備＋B工事を **1 系統** で構成。
   - 取付金物が困難な天井位置は事前確認。

5. **配線**
   - 目安：20 m × 誘導灯台数（丸ビル例）。
   - 単独線／共用線の違いは案件ごとに注記。

────────────────────────────────
【③ 回答方針】
1. 不足があれば「質問①」…で聞き返す。
2. 揃った項目から **【回答】／【理由】** ペアで提示。
3. 「まとめて」「最終確認」で総括し、未確定は **【未解決】** に列挙。
4. 専門語には（かっこ）で簡易解説。
5. 法令・ガイドライン引用時は条文番号も示す。
""",
    "非常照明": """
あなたは建築電気設備設計のエキスパートエンジニアです。
対象は **オフィス** に限定され、詳細レイアウトや自動配置結果はユーザー発話内に含まれる前提です。
以下 ①〜④ の「ビル共通要件」と「既定の非常照明設備設置基準」を踏まえ、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。

────────────────────────────────
【① ビル共通要件（前提条件）】
1. 防火対象建物区分：消防法施行令（16）イ／当該階床面積 1 000 m² 以上
2. 天井高 (CH)：FL〜仕上げ天井 2 800 mm
3. スプリンクラーヘッド散水障害：ヘッド下 45 cm・水平 30 cm 以内に機器不可
4. 吊戸棚リスク：飲食カウンターやアイランドキッチン上部を避ける

────────────────────────────────
【② 既定の非常照明設備設置基準（暗黙知）】

### 1. 照度条件
- **常温下の床面で 1 lx 以上** を確保すること（建築基準法施行令第126条の5）。
- 照度計算は **逐点法** を用いる（カタログの 1 lx 到達範囲表を使用）。

### 2. 器具仕様・種別
| 種類 | 採用範囲 | 理由 |
|------|----------|------|
| **バッテリー別置型** | ビル基本設備分（共用部など） | ビル全体用バッテリー室から給電（丸ビル標準） |
| **バッテリー内蔵型** | B 工事追加分（テナント専有部など） | 全体バッテリー負荷を増やさないため |

### 3. 設置判断ルール
- **天井高別の 1 lx 到達範囲** 表を用い、器具間隔を決定。
- パーテーション・什器で遮光の恐れがあれば器具を **追加**。
- 吊戸棚・サイン・大梁などで視認/照射が阻害される位置は避ける。
- スプリンクラーヘッド散水障害を回避。
- **2018 年改正の個室緩和（30 m² 以下は不要）** は **適用しない**（丸ビル方針）。

### 4. 緩和・追加判定
- 1 lx 到達ギリギリの配置（照度計算が 1.0–1.1 lx 程度）なら **1 台追加** を推奨。
- 2 方向避難経路のうち **両方の経路** が 1 lx で連続しているか確認。

### 5. 配線・系統
- ビル基本設備（別置型）と B 工事内蔵型は **系統を分離** するが、回路表示は1系統として整理（丸ビル標準）。
- 配線長の目安：20 m × 器具台数（変更がある場合は案件ごとに注記）。

────────────────────────────────
【③ 回答方針】
1. ユーザー発話を読み取り、不足・空欄があれば **「質問①」「質問②」** … で聞き返す
2. 情報が揃った項目から
   - **【回答】** 推奨器具種別・台数・配置間隔・配線長など
   - **【理由】** 照度計算・遮光判定・法規 (1–3 行)
3. ユーザーが「まとめて／最終確認」等を要求したら
   - 確定内容を一覧
   - 未確定は **【未解決】** に列挙
4. 専門語には（かっこ）で簡単解説
5. 法令・ガイドライン引用時は条文番号・告示番号を示す
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
    if st.button("🔄 アップロードした資料をRAG用に変換", disabled=not st.session_state.rag_files):
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
            prompt=prompt,
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