import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any
import time, functools

from src.rag_preprocess import preprocess_files
from src.rag_vector import save_docs_to_chroma
from src.rag_qa import generate_answer

import yaml
import streamlit_authenticator as stauth

st.set_page_config(page_title="GPT + RAG Chatbot", page_icon="💬", layout="wide")

# =====  認証設定の読み込み ============================================================
with open('./config.yaml') as file:
    config = yaml.safe_load(file)

# 認証インスタンスの作成
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# =====  基本設定  ============================================================
client = OpenAI()

# =====  ログインUIの表示  ============================================================
authenticator.login()

if st.session_state["authentication_status"]:
    name = st.session_state["name"]
    username = st.session_state["username"]
    # --------------------------------------------------------------------------- #
    #                         ★ 各モード専用プロンプト ★                           #
    # --------------------------------------------------------------------------- #
    DEFAULT_PROMPTS: Dict[str, str] = {
        "暗黙知法令チャットモード": """
    あなたは建築電気設備設計のエキスパートエンジニアです。
    今回の対象は **複合用途ビルのオフィス入居工事** に限定されます。
    以下の知識と技術をもとに、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。
    専門用語は必要に応じて解説を加え、判断の背景にある理由を丁寧に説明します。
    なお、各事項は「代表的なビル（丸の内ビルディング）」を想定して記載しています。

    ────────────────────────────────
    【ビル共通要件】
    1. **防火対象建物区分**
        - 消防法施行令 区分「（16）複合用途防火対象物　イ」
        - 当該階床面積 1 000 m² 以上

    ────────────────────────────────
    【コンセント設計】
    ■ 図面の指示と基本的な割り振り
    - 図面や要望書の指示を優先（単独回路や専用ELB等）
    - 機器やデスクの配置をもとにグループ化
    - 一般的なオフィス机は複数の座席をまとめて1回路
    - 機器の消費電力が高い場合や同時使用想定で回路分割

    ■ 机・椅子（デスク周り）の標準設計
    - 個人用デスク： 1席ごとにOAタップ1個、6席ごとに1回路（300VA/席）
    - フリーアドレスデスク：1席ごとにOAタップ1個、8席ごとに1回路（150VA/席）
    - 昇降デスク：1席ごとにOAタップ1個、2席ごとに1回路（600VA/席）
    - 会議室テーブル：4席ごとにOAタップ1個、12席ごとに1回路（150VA/席）

    ■ 設備機器の設計
    - 単独回路が必要な機器（コピー機、プリンター、電子レンジ、冷蔵庫、等）
    - 水気のある機器にはELB必須
    - 300〜1200VA程度の機器は近い位置で1回路にまとめ可能（1500VA上限）

    ■ 特殊エリアの電源
    - パントリー：最低OAタップ5個と5回路
    - エントランス：最低OAタップ1個と1回路
    - プリンター台数：40人に1台が目安

    ────────────────────────────────
    【自動火災報知器設計】
    ■ 感知器の種類・仕様
    - 基本的に廊下も居室も「煙感知器スポット型2種」を使用（丸ビル標準）
    - 天井面中央付近、または障害を避けて煙が集まりやすい位置に設置

    ■ 設置基準
    - 廊下：端点から15m以内、感知器間30m以内
    - 居室：面積150m²ごとに1個（切り上げ）
    - 障害物がある場合は基本個数+1
    - 天井高2.3m未満または40m²未満の居室は入口付近
    - 吸気口付近に設置、排気口付近は避ける
    - 厨房は定温式スポット型（1種）、防火シャッター近くは専用感知器

    ────────────────────────────────
    【非常放送設備設計】
    ■ スピーカ設置基準
    - 到達距離10m以内（各居室・廊下を半径10mの円でカバー）
    - 省略可能条件：居室・廊下は6m²以下、その他区域は30m²以下、かつ隣接スピーカから8m以内
    - パーテーションや什器による遮音は考慮しない（半径10mの円は不変）

    ────────────────────────────────
    【誘導灯設計】
    ■ 種類・採用機種
    - 避難口誘導灯・通路誘導灯のみ使用
    - 両者ともB級BH型（20A形）のみ使用（丸ビル標準）

    ■ 設置箇所・有効距離
    - 避難口誘導灯：最終避難口、または最終避難口に通じる経路上の扉
        有効距離30m（シンボル無）／20m（矢印付き）
    - 通路誘導灯：廊下の曲がり角や分岐点、または避難口誘導灯の有効距離補完
        有効距離15m

    ■ 配置判断
    - 扉開閉・パーテーション・背の高い棚などで視認阻害→位置変更または追加

    ────────────────────────────────
    【非常照明設計】
    ■ 照度条件
    - 常温下の床面で1lx以上を確保（建築基準法施行令第126条の5）
    - 照度計算は逐点法を用いる（カタログの1lx到達範囲表使用）

    ■ 器具仕様・種別
    - バッテリー別置型：ビル基本設備分（入居前既設分）
    - バッテリー内蔵型：B工事追加分（間仕切り変更などで追加した分）

    ■ 設置判断ルール
    - 天井高別の1lx到達範囲表を用い、器具間隔を決定
    - パーテーション・什器で遮光の恐れがあれば器具を追加
    - 2018年改正の個室緩和（30m²以下は不要）は適用しない（丸ビル方針）
    """,

        "質疑応答書添削モード": """
    あなたは建築電気設備分野における質疑応答書作成の専門家です。
    ユーザーが入力した文章を、見積根拠図や見積書と一緒に提出する質疑応答書として最適な文章に添削してください。

    【重要】添削文のみを出力し、添削内容の説明は一切不要です。

    【添削・整形の仕様】
    1. **誤字脱字の修正**
        - 一般的な誤字・脱字を検出し、修正します

    2. **表現の統一・調整**
        - 質疑応答書として適切かつ丁寧な表現に統一・調整します
        - 敬体（です・ます調）を基本とします
        - 過度な敬語や冗長な表現は避け、簡潔で分かりやすい表現に修正します
        - 専門用語の表記を統一します

    3. **見積・提案の文脈に合わせた表現**
        - 「指示がない部分」については「見積依頼図に基づき想定で見込んでいます」という表現を基本とします
        - 「具体的な指示をいただけますか」といった質問は「〜という内容で見込んでいます」という確認形式に変換します
        - 提案や確認の際は「〜と考えておりますが、いかがでしょうか」「〜で想定しておりますがよろしいでしょうか」といった表現を使用します

    4. **クローズドクエスチョンへの変換**
        - オープンクエスチョン（「どうしますか？」「何ですか？」など）を、クローズドクエスチョン（「〜でよろしいでしょうか？」など）に変換します
        - 「ご指示ください」→「〜で見込んでいます」
        - 「いかがいたしましょうか」→「〜と考えておりますが、よろしいでしょうか」
        - 決定や承認を求める場合は「〜とさせていただきたいと思いますが、よろしいでしょうか」

    【変換例】
    修正前：会議室のコンセントは指示がないですが、どうしますか？
    修正後：会議室のコンセントについてはご指示がなかったため、見積依頼図に基づき想定で見込んでいます。

    修正前：家具コンセント・テレキューブが設置される場所に関してはOAタップを設置する位置をご指示いただけますでしょうか。
    修正後：家具コンセント・テレキューブが設置される場所に関しては、OA内にOAタップを設置する想定で見込んでいますが、よろしいでしょうか。

    修正前：照明用回路が設計上1回路で設定されておりますが、容量が1800Ｗ想定ということですので安全を考えると回路数の変更が必要かとおもっておりますがいかがいたしましょうか。
    修正後：照明用回路が設計上1回路で設定されておりますが、容量が1800Ｗ想定ということですので安全を考え2回路へ変更させていただきたいと思いますが、よろしいでしょうか。

    【出力】
    添削内容を1つだけ出力してください。説明や理由などの付加情報は一切不要です。
    出力は添削した質疑応答書の文章のみとしてください。
    """
    }

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
        st.session_state.design_mode = list(DEFAULT_PROMPTS.keys())[0]  # デフォルトは「全設備モード」
    if "prompts" not in st.session_state:
        st.session_state.prompts = DEFAULT_PROMPTS.copy()  # プロンプトを変更可能に
    if "gpt_model" not in st.session_state:
        st.session_state.gpt_model = "gpt-4.1"  # デフォルトモデルをgpt-4.1に変更


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

    # ----- チャットタイトル自動生成機能 -----
    def generate_chat_title(messages):
        if len(messages) >= 2:  # ユーザー質問と回答が1往復以上ある場合
            prompt = f"以下の会話の内容を25文字以内の簡潔なタイトルにしてください:\n{messages[0]['content'][:200]}"
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-nano",  # 軽量モデルで十分
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                )
                return resp.choices[0].message.content.strip('"').strip()
            except:
                return f"Chat {len(st.session_state.chats) + 1}"
        return f"Chat {len(st.session_state.chats) + 1}"

    # =====  編集機能用のヘルパー関数  ==============================================
    def handle_save_prompt(mode_name, edited_text):
        st.session_state.prompts[mode_name] = edited_text
        st.session_state.edit_target = None
        st.success(f"「{mode_name}」のプロンプトを更新しました")
        time.sleep(1)
        st.rerun()

    def handle_reset_prompt(mode_name):
        if mode_name in DEFAULT_PROMPTS:
            st.session_state.prompts[mode_name] = DEFAULT_PROMPTS[mode_name]
            st.success(f"「{mode_name}」のプロンプトをデフォルトに戻しました")
            time.sleep(1)
            st.rerun()
        else:
            st.error("このモードにはデフォルト設定がありません")

    def handle_cancel_edit():
        st.session_state.edit_target = None
        st.rerun()

    # =====  CSS  ================================================================
    # CSSを改善してダークモード対応
    st.markdown(
        """
        <style>
        :root{ --sidebar-w:260px; --pad:1rem; }
        aside[data-testid="stSidebar"]{width:var(--sidebar-w)!important;}
        .chat-body{max-height:70vh;overflow-y:auto;}
        .stButton button {font-size: 16px; padding: 8px 16px;}

        /* モバイル対応 */
        @media (max-width: 768px) {
            :root{ --sidebar-w:100%; --pad:0.5rem; }
            .chat-body {max-height: 60vh;}
            .stButton button {font-size: 14px; padding: 6px 12px;}
        }

        /* ダークモード対応メッセージスタイル - カスタム背景色は削除 */
        .user-message, .assistant-message {
            border-radius: 10px;
            padding: 8px 12px;
            margin: 5px 0;
            border-left: 3px solid;
        }
        .user-message {
            border-left-color: #4c8bf5;
        }
        .assistant-message {
            border-left-color: #ff7043;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =====  サイドバー  ==========================================================
    with st.sidebar:
        st.markdown(f"👤 ログインユーザー: `{name}`")
        authenticator.logout('ログアウト', 'sidebar')

        # ------- RAG アップロード -------
        st.markdown("### 📂 RAG 資料アップロード")
        uploads = st.file_uploader(
            "PDF / TXT を選択…",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )
        if uploads:
            st.session_state.rag_files = [
                {"name": f.name, "type": f.type, "size": f.size, "data": f.getvalue()} for f in uploads
            ]
        if st.button("🔄 インデックス再構築", disabled=not st.session_state.rag_files):
            rebuild_rag_collection()

        st.divider()

        # ------- モデル選択 -------
        st.markdown("### 🤖 GPTモデル選択")
        model_options = {
            "gpt-4.1": "GPT-4.1 (標準・最新世代)",
            "gpt-4.1-mini": "GPT-4.1-mini (小・最新世代)",
            "gpt-4.1-nano": "GPT-4.1-nano (超小型・高速)",
            "gpt-4o": "GPT-4o (標準・高性能)",
            "gpt-4o-mini": "GPT-4o-mini (小・軽量)"
        }
        st.session_state.gpt_model = st.selectbox(
            "使用するモデルを選択",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.gpt_model) if st.session_state.gpt_model in model_options else 0,
        )
        st.markdown(f"**🛈 現在のモデル:** `{model_options[st.session_state.gpt_model]}`")

        # ------- モデル詳細設定 -------
        with st.expander("🔧 詳細設定"):
            st.slider("応答の多様性",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,  # OpenAIのデフォルト値
                    step=0.1,
                    key="temperature",
                    help="値が高いほど創造的、低いほど一貫した回答になります（OpenAIデフォルト: 1.0）")

            max_tokens_options = {
                "未設定（モデル上限）": None,
                "500": 500,
                "1000": 1000,
                "2000": 2000,
                "4000": 4000,
                "8000": 8000
            }
            selected_max_tokens = st.selectbox(
                "最大応答長",
                options=list(max_tokens_options.keys()),
                index=0,  # デフォルトは「未設定（モデル上限）」
                key="max_tokens_select",
                help="生成される回答の最大トークン数（OpenAIデフォルト: モデル上限）"
            )
            # sessionの値を更新
            st.session_state["max_tokens"] = max_tokens_options[selected_max_tokens]

        st.divider()

        # ------- モード選択 -------
        st.markdown("### ⚙️ 設計対象モード")
        st.session_state.design_mode = st.radio(
            "対象設備を選択",
            options=list(st.session_state.prompts.keys()),
            index=0,  # デフォルトは「全設備モード」
            key="design_mode_radio",
        )
        st.markdown(f"**🛈 現在のモード:** `{st.session_state.design_mode}`")

        # ------- プロンプト編集ボタン -------
        if st.button("✏️ 現在のプロンプトを編集"):
            st.session_state.edit_target = st.session_state.design_mode

        st.divider()

        # ------- チャット履歴 -------
        st.header("💬 チャット履歴")
        for title in list(st.session_state.chats.keys()):
            if st.button(title, key=f"hist_{title}"):
                st.session_state.current_chat = title
                st.rerun()

        if st.button("➕ 新しいチャット"):
            base = f"Chat {len(st.session_state.chats) + 1}"
            st.session_state.current_chat = base
            st.session_state.chats[st.session_state.current_chat] = []
            st.rerun()

    # =====  プロンプト編集画面  =================================================
    if st.session_state.edit_target:
        mode_name = st.session_state.edit_target

        # 完全にクリーンなコンテナでプロンプト編集UI
        st.title(f"✏️ プロンプト編集: {mode_name}")

        # 編集用フォーム - フォームを使うことで確実に入力を受け付ける
        with st.form(key=f"prompt_edit_form_{mode_name}"):
            # テキストエリア
            prompt_text = st.text_area(
                "プロンプトを編集してください",
                value=st.session_state.prompts[mode_name],
                height=400
            )

            # フォーム内のボタン
            col1, col2, col3 = st.columns(3)
            with col1:
                save_button = st.form_submit_button(label="✅ 保存")
            with col2:
                reset_button = st.form_submit_button(label="🔄 デフォルトに戻す")
            with col3:
                cancel_button = st.form_submit_button(label="❌ キャンセル")

        # フォーム送信後の処理
        if save_button:
            handle_save_prompt(mode_name, prompt_text)
        elif reset_button:
            handle_reset_prompt(mode_name)
        elif cancel_button:
            handle_cancel_edit()

    # =====  中央ペイン  ==========================================================
    # プロンプト編集モードでない場合のみチャットインターフェースを表示
    if not st.session_state.edit_target:
        # 現在のモデルとモードを表示
        st.title("💬 GPT + RAG チャットボットv2")
        st.subheader(f"🗣️ {st.session_state.current_chat}")
        st.markdown(f"**モデル:** {st.session_state.gpt_model} | **モード:** {st.session_state.design_mode}")

        # -- メッセージ表示 --
        st.markdown('<div class="chat-body">', unsafe_allow_html=True)
        for m in get_messages():
            message_class = "user-message" if m["role"] == "user" else "assistant-message"
            with st.chat_message(m["role"]):
                st.markdown(f'<div class="{message_class}">{m["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # -- 入力欄 --
        user_prompt = st.chat_input("メッセージを入力…")
    else:
        # プロンプト編集モード時は入力欄を無効化
        user_prompt = None


    # =====  応答生成  ============================================================
    if user_prompt and not st.session_state.edit_target:  # 編集モード時は応答生成をスキップ
        # メッセージリストに現在の質問を追加
        msgs = get_messages()
        msgs.append({"role": "user", "content": user_prompt})

        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{user_prompt}</div>', unsafe_allow_html=True)

        # シンプルなステータス表示 - 折りたたみなし
        with st.status(f"🤖 {st.session_state.gpt_model} で回答を生成中...", expanded=True) as status:
            # プロンプト取得
            prompt = st.session_state.prompts[st.session_state.design_mode]

            # ---------- RAG あり ----------
            if st.session_state.rag_collection is not None:
                st.session_state["last_answer_mode"] = "RAG"
                rag_res = generate_answer(
                        prompt=prompt,
                        question=user_prompt,
                        collection=st.session_state.rag_collection,
                        rag_files=st.session_state.rag_files,  # ← ここを追加
                        top_k=4,
                        model=st.session_state.gpt_model,
                        chat_history=msgs,
                    )
                assistant_reply = rag_res["answer"]
                sources = rag_res["sources"]

            # ---------- GPT-only ----------
            else:
                st.session_state["last_answer_mode"] = "GPT-only"
                # API呼び出し部分（条件付き）
                params = {
                    "model": st.session_state.gpt_model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        *msgs[:-1],
                        {"role": "user", "content": user_prompt},
                    ]
                }

                # カスタム設定があれば追加
                if st.session_state.get("temperature") != 1.0:
                    params["temperature"] = st.session_state.temperature
                if st.session_state.get("max_tokens") is not None:
                    params["max_tokens"] = st.session_state.max_tokens

                # APIを呼び出し
                resp = client.chat.completions.create(**params)

                assistant_reply = resp.choices[0].message.content
                sources = []

            # ---------- 画面反映 ----------
            with st.chat_message("assistant"):
                # モデル情報を応答に追加
                model_info = f"\n\n---\n*このレスポンスは `{st.session_state.gpt_model}` で生成されました*"
                full_reply = assistant_reply + model_info
                st.markdown(full_reply)

            # チャットメッセージ外で expander 表示
            # if sources:
            #     st.markdown("### 🔎 RAG が取得したチャンク")  # タイトルとして使う
            #     for idx, s in enumerate(sources, 1):
            #         chunk = s.get("content", "")[:200]
            #         if len(s.get("content", "")) > 200:
            #             chunk += " …"
            #         with st.expander(f"Doc {idx} - {s['metadata'].get('source','N/A')} (score: {s['distance']:.4f})"):
            #             st.markdown(f"> {chunk}")

            # 保存するのは元の応答（モデル情報なし）
            msgs.append({"role": "assistant", "content": assistant_reply})

            # チャットタイトル自動生成（初回応答後）
            if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                new_title = generate_chat_title(msgs)
                if new_title and new_title != st.session_state.current_chat:
                    old_title = st.session_state.current_chat
                    st.session_state.chats[new_title] = st.session_state.chats[old_title]
                    del st.session_state.chats[old_title]
                    st.session_state.current_chat = new_title
            
            st.rerun()

elif st.session_state["authentication_status"] is False:
    st.error('ユーザー名またはパスワードが間違っています。')
elif st.session_state["authentication_status"] is None:
    st.warning("ユーザー名とパスワードを入力してください。")
    st.stop()
