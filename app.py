import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any
import time, functools, requests
from concurrent.futures import ThreadPoolExecutor
import copy

from src.rag_preprocess import preprocess_files
from src.rag_vector import save_docs_to_chroma
from src.rag_qa import generate_answer
from src.startup_loader import initialize_chroma_from_input
from src.logging_utils import init_logger

import yaml
import streamlit_authenticator as stauth
import uuid

st.set_page_config(page_title="GPT + RAG Chatbot", page_icon="💬", layout="wide")

logger = init_logger()

# グローバルで使い回す実行基盤（CPU 2core 制限を考慮し 3スレッド程度に抑える）
EXECUTOR = ThreadPoolExecutor(max_workers=3)

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

WEBHOOK_URL = st.secrets["WEBHOOK_URL"]

# ログ送信用の関数
def post_log(
    input_text: str,
    output_text: str,
    prompt: str,
):
    """Apps Script Webhook へ 1 行 POST する（302 を成功扱い）"""
    payload = {
        "mode": st.session_state["design_mode"],       # ex: "chat", "eval"
        "model": st.session_state['gpt_model'],        # ex: "gpt-4", "gpt-4o"
        "title": st.session_state["current_chat"],     # 会話タイトル
        "input": input_text,                           # 入力メッセージ
        "output": output_text,                         # 出力メッセージ（LLM応答）
        "prompt": prompt,                              # システムプロンプトなど
    }

    for wait in (0, 1, 3):  # 最大3回リトライ
        try:
            r = requests.post(
                WEBHOOK_URL,
                json=payload,
                timeout=4,
                allow_redirects=False,  # 302 Found を許容するため
            )
            if r.status_code in (200, 302):
                logger.info("✅ post_log ok — status=%s", r.status_code)
                return
            logger.warning("⚠️ post_log status=%s body=%s",
                           r.status_code, r.text[:120])
        except requests.RequestException as e:
            logger.error("❌ post_log error — %s", e, exc_info=True)
        time.sleep(wait)

    st.warning("⚠️ Sheets へのログ送信に失敗しました")

# =====  基本設定  ============================================================
client = OpenAI()

# =====  ログインUIの表示  ============================================================
authenticator.login()

if st.session_state["authentication_status"]:
    name = st.session_state["name"]
    username = st.session_state["username"]

    logger.info("🔐 login success — user=%s  username=%s", name, username)

    # Chromaコレクションを input_data から自動初期化（persist_directory=None → インメモリ）
    if st.session_state.get("rag_collection") is None:
        try:
            res = initialize_chroma_from_input(
                input_dir="rag_data",
                persist_dir=None,  # 永続化しない
                collection_name="session_docs"
            )
            st.session_state.rag_collection = res["collection"]
            st.session_state.rag_files = res["rag_files"]

            logger.info("📂 Chroma init — chunks=%d  files=%d",
                    res["collection"].count(), len(res["rag_files"]))
            
        except Exception as e:
            logger.exception("❌ Chroma init failed — %s", e)
            st.warning(f"RAG初期化中にエラーが発生しました: {e}")

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
    - 当該階床面積 1000 m² 以上

    ────────────────────────────────
    【コンセント設備（床）】
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
    - 単独回路が必要な機器（複合機（コピー機）、プリンター、シュレッダー、テレブース、自動販売機、冷蔵庫、ウォーターサーバー、電子レンジ、食器洗い乾燥機、コーヒーメーカー、ポット、造作家具（什器用コンセント）、インターホン親機、サーバーラック、セキュリティシステム、等）
    - 水気のある機器にはELB必須
    - 分岐回路でもよい機器（ディスプレイ（会議室、応接室、役員室）、テレビ（共用）、スタンド照明、ロッカー（電源供給機能付）、等）
    - 300〜1200VA程度の機器は近い位置で1回路にまとめ可能（1500VA上限）

    ■ 特殊エリアの電源
    - パントリー：最低OAタップ5個と5回路
    - エントランス：最低OAタップ1個と1回路
    - プリンター台数：20人に1台が目安、40人に1台が確保できてなければ電源の追加を提案

    ────────────────────────────────
    【コンセント設備（壁）※一般用コンセント】
    ■ 用途と設置考え方
    - 清掃時に掃除機を接続するための電源（入居企業は使用不可）
    - 見積図面では提案するが、入居企業の要望により省略も可能
    - 設置位置は主に扉横

    ■ 配置判断ルール
    - 清掃時の動線（≒避難経路）を考慮して配置
    - 扉を挟んだどちら側に設置するかの精査が必要
    - 各部屋の入口付近に最低1箇所

    ────────────────────────────────
    【コンセント設備（壁）※客先指示電源】
    ■ 設置基準
    - 見積依頼図に指示された場所、指示された仕様で設置
    - 客先からの特殊指示（単独回路、専用ELB等）を最優先
    - 図面上の明示がなくても打合せ記録等で指示があれば対応

    ■ 追加提案判断
    - 見積図に指示がなくても、使用目的が明確な場合は追加提案
    - 特殊機器（給湯器、加湿器等）の近くには設置を提案

    ────────────────────────────────
    【コンセント設備（天井）】
    ■ 設置基準
    - 見積依頼図に指示があった場所に設置
    - 電源が必要な天井付近の機器がある場合に1個設置

    ■ 対象機器
    - プロジェクター
    - 電動スクリーン
    - 電動ブラインド
    - 壁面発光サイン
    - その他天井付近に設置される電気機器

    ────────────────────────────────
    【自動火災報知設備】
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
    【非常放送設備】
    ■ スピーカ設置基準
    - 到達距離10m以内（各居室・廊下を半径10mの円でカバー）
    - パーテーションや什器による遮音は考慮しない（半径10mの円は不変）

    ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷200㎡）の切り上げ
    - 200㎡は「L級スピーカー」の有効範囲円（半径10m）に内接する正方形の面積

    ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 省略可能条件（居室・廊下は6m²以下、その他区域は30m²以下、かつ隣接スピーカから8m以内なら省略可能）は適用しない（丸ビル方針）
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置

    ────────────────────────────────
    【誘導灯設備】
    ■ 種類・採用機種
    - 避難口誘導灯・通路誘導灯のみ使用
    - 両者ともB級BH型（20A形）のみ使用（丸ビル標準）

    ■ 設置箇所・有効距離
    - 避難口誘導灯：最終避難口、または最終避難口に通じる避難経路上の扉
    有効距離30m（シンボル無）／20m（矢印付き）
    - 通路誘導灯：廊下の曲がり角や分岐点、または避難口誘導灯の有効距離補完
    有効距離15m

    ■ 配置判断
    - 扉開閉・パーテーション・背の高い棚などで視認阻害→位置変更または追加

    ────────────────────────────────
    【非常照明設備】
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

    ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷50㎡）の切り上げ
    - 50㎡は新丸ビルにおける非常照明設備の有効範囲円（半径5.0m）に内接する正方形の面積

    ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置

    ────────────────────────────────
    【照明制御設備（照度センサ）】
    ■ 設置判断ルール
    - 天井高別の有効範囲表を用い、器具間隔を決定
    - パーテーション・什器で遮られる恐れがあれば器具を追加

    ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷28㎡）の切り上げ
    - 28㎡は新丸ビルにおける照度センサの有効範囲円（半径3.75m）に内接する正方形の面積

    ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置

    ────────────────────────────────
    【照明制御設備（スイッチ）】
    ■ 設置判断ルール
    - 新規に間仕切りされた領域に対してそれぞれ設置
    - 設置するスイッチ数は領域の大きさや扉の配置、制御の分け方による

    ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷20㎡）の切り上げ
    - 算出個数に基づき、「最終避難口」以外の「扉」の横に配置（最終避難口の横にはビル基本のスイッチがあるため追加設置不要）

    ■ 配置ルール
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 配置数2個以上かつ扉数2個以上の場合は、領域内の「扉」の横に均等に配置
    - 扉数＞個数の場合は最終避難口への距離が短い扉から優先的に配置
    - 本来は入退室ルート（≒避難経路）に基づく動線計画に従い、設置位置を精査

    ────────────────────────────────
    【テレビ共聴設備】
    ■ 設置基準
    - 見積依頼図に指示があった場所に設置
    - テレビ共聴設備が必要な什器がある場所に1個設置

    ■ 設置が必要な部屋・什器
    - 会議室：最低1個は設置
    - 応接室：最低1個は設置
    - 役員室：最低1個は設置
    - ディスプレイ（会議室、応接室、役員室にあるもの）
    - テレビ（共用のもの）

    ────────────────────────────────
    【電話・LAN設備（配管）】【防犯設備（配管）】
    ■ 業務基本原則
    - 基本的には客先から図面を受領して見積りを作成
    - C工事会社から配管の設置のみ依頼される場合が多い

    ■ 概算見積りの考え方
    - 概算段階では配線図を作成せず、細部計算を省略することが一般的
    - 「設備数×○m」という形式で概算を算出
    - 各ビル・各設備ごとの「黄金数字（○m）」を考慮した設計が必要

    ────────────────────────────────
    【動力設備（配管、配線）】
    ■ 適用場面と業務原則
    - 基本的には客先から図面を受領して見積りを作成
    - 店舗（特に飲食店）では必要性が高い
    - オフィスでも稀に必要となるケースがある

    ■ 概算見積りの特徴
    - 概算段階では配置平面図よりも、必要な動力設備の種類と数をまとめた表から算出
    - 表を読み解いて必要数を算出し見積りに反映

    ■ オフィスでの対応
    - オフィスで必要な場合：動力用の分電盤、配線・配管を設置
    - 詳細な設計検討が必要（概算見積対応はできない）

    ────────────────────────────────
    【注意点】
    検索結果の言い回しをそのまま複製することを避け、直接引用以外のすべてを自分の言葉で表現します。
    """,

        "質疑応答書添削モード": """
    あなたは建築電気設備分野における質疑応答書作成の専門家です。
    ユーザーが入力した文章を、見積根拠図や見積書と一緒に提出する質疑応答書として最適な文章に添削してください。

    【重要】添削文のみを出力し、添削内容の説明は一切不要です。

    【添削・整形の仕様】
    1. **誤字脱字の修正**
        - 一般的な誤記、表記揺れを修正し、読みやすく整えます。

    2. **表現の統一・調整**
        - 質疑応答書として適切かつ丁寧な表現に統一・調整します
        - 文体は敬体（です・ます調）に統一します
        - 過度な敬語や冗長な表現は避け、簡潔で分かりやすい表現に修正します
        - 専門用語は業界標準に則って表記統一します

    3. **見積・提案の文脈に合わせた表現**
        - 指示がなくても合理的に見積もれる内容であれば、**確認文を使わずに断定的に表現**してください。
        例：「○○については□□として見込んでおります。」
        - 情報が明らかに不足しており、仕様決定の判断ができない場合のみ、
        **前提を提示したうえで控えめに確認を促す表現**としてください。
        例：「図面記載がないため、○○として想定しておりますが、仕様のご確認をお願いいたします。」

    4. **クローズドクエスチョンへの変換**
        - 「〜でよろしいでしょうか？」「〜でしょうか？」といった**クローズドクエスチョン表現は使用しないでください。**
        - 「〜と見込んでおります」や「〜とさせていただきたいと考えております」といった**先方のリアクションがなくてもそのまま見積を行えるような文章**が理想です。

    【変換例】
    変換前：
    家具コンセント・テレキューブが設置される場所に関してはOA内にOAタップを設置する認識でよろしいでしょうか。
    変換後：
    家具コンセント・テレキューブが設置される場所については、OA内にOAタップを設置する前提としております。

    変換前：
    NW工事（光ケーブル、電話含め）、AV工事は全てC工事という認識でよろしいですね。
    変換後：
    NW工事（光ケーブル、電話含む）およびAV工事は、全てC工事区分として想定しております。

    変換前：
    ＴＶ共聴信号については、壁埋め込みとしコンセントと２連での設置でよろしいでしょうか。また、口数はいくつ必要でしょうか。
    変換後：
    TV共聴信号については、コンセントと2連の壁埋め込み型で設置する想定です。必要な口数は未記載のため、ご指示をお願いいたします。

    【出力】
    添削内容を1つだけ出力してください。説明や理由などの付加情報は一切不要です。
    出力は添削した質疑応答書の文章のみとしてください。

    【注意点】
    検索結果の言い回しをそのまま複製することを避け、直接引用以外のすべてを自分の言葉で表現します。
    """
    }

    # =====  セッション変数  =======================================================
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "chat_sids"   not in st.session_state:                        # ★ 追加
        st.session_state.chat_sids = {"New Chat": str(uuid.uuid4())}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"
    if "sid"         not in st.session_state:                        # ★ 追加
        st.session_state.sid = st.session_state.chat_sids["New Chat"]
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
    if "sid" not in st.session_state:          # 追加
        import uuid
        st.session_state.sid = str(uuid.uuid4())
    if "use_rag" not in st.session_state:
        st.session_state["use_rag"] = False  # ← デフォルトでRAGを使わない
    if "comparison_results" not in st.session_state:
        # {(chat_sid, turn_no): {model_name: answer_text}}
        st.session_state.comparison_results = {}


    # =====  ヘルパー  ============================================================
    def get_messages() -> List[Dict[str, str]]:
        title = st.session_state.current_chat
        return st.session_state.chats.setdefault(title, [])
    
    # ★ 新しいチャットを作成
    def new_chat():
        title = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[title] = []
        st.session_state.chat_sids[title] = str(uuid.uuid4())   # 新sid
        st.session_state.current_chat = title
        st.session_state.sid = st.session_state.chat_sids[title]

        logger.info("➕ new_chat — sid=%s  title='%s'", st.session_state.sid, title)

        st.rerun()

    # ★ 既存チャットへ切替
    def switch_chat(title: str):
        if title not in st.session_state.chat_sids:          # ★ 安全化
            st.session_state.chat_sids[title] = str(uuid.uuid4())
        st.session_state.current_chat = title
        st.session_state.sid = st.session_state.chat_sids[title]

        logger.info("🔀 switch_chat — sid=%s  title='%s'", st.session_state.sid, title)

        st.rerun()

    def rebuild_rag_collection():
        """
        アップロードされたファイルを前処理 → Chroma 登録し、セッションに保持
        """
        if not st.session_state.rag_files:
            st.warning("まず PDF / TXT をアップロードしてください")
            logger.warning("📚 RAG rebuild aborted — no files")
            return

        total_files = len(st.session_state.rag_files)
        logger.info("📚 RAG rebuild start — files=%d", total_files)

        import time
        t0 = time.perf_counter()            # 所要時間計測

        try:
            with st.spinner("📚 ファイルを解析し、ベクトル DB に登録中..."):
                docs = preprocess_files(st.session_state.rag_files)
                col = save_docs_to_chroma(
                    docs=docs,
                    collection_name="session_docs",
                    persist_directory=None,   # インメモリ
                )
                st.session_state.rag_collection = col

            chunk_count = col.count()
            elapsed = time.perf_counter() - t0
            logger.info("✅ RAG rebuild done — chunks=%d  files=%d  elapsed=%.2fs",
                        chunk_count, total_files, elapsed)

            st.success("🔍 検索インデックスを更新しました！")

        except Exception as e:
            logger.exception("❌ RAG rebuild failed — %s", e)
            st.error(f"RAG 初期化中にエラーが発生しました: {e}")

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
    
    # =====  チャット応答生成  =========================================
    def generate_with_model(model_name: str,
                            system_prompt: str,
                            user_text: str,
                            msgs_history: List[Dict[str, str]],
                            temperature: float = 1.0) -> str:   # ← 追加
        """
        指定モデルで回答を生成し、プレーンテキストを返す。
        """

        # ----- RAG あり -----
        if st.session_state.get("use_rag", True):
            res = generate_answer(
                prompt=system_prompt,
                question=user_text,
                collection=st.session_state.rag_collection,
                rag_files=st.session_state.rag_files,
                top_k=4,
                model=model_name,
                chat_history=msgs_history,
                temperature=temperature          # ← ここを追加
            )
            return res["answer"]

        # ----- GPT only -----
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                *msgs_history[:-1],
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature          # ← ここを追加
        }
        if st.session_state.get("max_tokens") is not None:
            params["max_tokens"] = st.session_state.max_tokens

        resp = client.chat.completions.create(**params)
        return resp.choices[0].message.content
    
    def launch_background_comparisons(prompt: str,
                                    user_text: str,
                                    msgs_history: List[Dict[str, str]]):
        """比較対象を温度に応じて自動決定して非同期実行する"""
        this_sid   = st.session_state.sid
        turn_no    = len(msgs_history)
        key        = (this_sid, turn_no)

        if key not in st.session_state.comparison_results:
            st.session_state.comparison_results[key] = {}

        main_model     = st.session_state.gpt_model
        main_temp_used = float(st.session_state.get("temperature", 1.0))
        compare_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]

        # 温度候補
        temperature_set = [0.0, 1.0]

        def _worker(m_name, temp):
            try:
                answer = generate_with_model(m_name, prompt, user_text, msgs_history, temp)
                st.session_state.comparison_results[key][(m_name, temp)] = answer
                st.experimental_rerun()
            except Exception as e:
                st.session_state.comparison_results[key][(m_name, temp)] = f"❌ Error: {e}"

        for model in compare_models:
            if model == main_model:
                for temp in temperature_set:
                    # 自分の出力済み温度と「異なる」ものだけ実行
                    if abs(temp - main_temp_used) > 1e-6:
                        EXECUTOR.submit(_worker, model, temp)
            else:
                for temp in temperature_set:
                    EXECUTOR.submit(_worker, model, temp)

    # =====  編集機能用のヘルパー関数  ==============================================
    def handle_save_prompt(mode_name, edited_text):
        st.session_state.prompts[mode_name] = edited_text
        st.session_state.edit_target = None

        logger.info("✏️ prompt_saved — mode=%s  len=%d",
                mode_name, len(edited_text))
        
        st.success(f"「{mode_name}」のプロンプトを更新しました")
        time.sleep(1)
        st.rerun()

    def handle_reset_prompt(mode_name):
        if mode_name in DEFAULT_PROMPTS:
            st.session_state.prompts[mode_name] = DEFAULT_PROMPTS[mode_name]

            logger.info("🔄 prompt_reset — mode=%s", mode_name)

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
                switch_chat(title)

        if st.button("➕ 新しいチャット"):
            new_chat()
        
        st.divider()

        # ===== サイドバー（モデル選択などの下が最適） =====
        st.markdown("### 🧠 RAG 検索の使用設定")

        st.session_state["use_rag"] = st.checkbox(
            "検索資料（ベクトルDB）を活用する",
            value=st.session_state["use_rag"],
            help="OFFにすると、プロンプトと履歴のみで応答を生成します"
        )

        # ✅ 現在のモードを明示表示
        if st.session_state["use_rag"]:
            st.success("現在のモード: RAG使用中")
        else:
            st.info("現在のモード: GPTのみ（検索なし）")

        # サイドバー下部など、rag_collection の表示
        st.markdown("### 🗂 ベクトルDBステータス")

        if st.session_state.get("rag_collection"):
            st.success("✔️ ベクトルDBは初期化済みです")
            try:
                count = st.session_state.rag_collection.count()
                st.markdown(f"📄 登録チャンク数: `{count}`")
            except Exception as e:
                st.warning(f"⚠️ 件数取得失敗: {e}")
        else:
            st.error("❌ ベクトルDBがまだ初期化されていません")

        # ------- RAG アップロード -------
        st.markdown("### 📂 追加RAG 資料アップロード")
        uploads = st.file_uploader(
            "PDF / TXT を選択…",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )
        if uploads:
            st.session_state.rag_files = [
                {"name": f.name, "type": f.type, "size": f.size, "data": f.getvalue()} for f in uploads
            ]

            logger.info("📥 file_uploaded — files=%d  total_bytes=%d",
                len(uploads), sum(f.size for f in uploads))
            
        if st.button("🔄 インデックス再構築", disabled=not st.session_state.rag_files):
            rebuild_rag_collection()

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
        st.title("💬 GPT + RAG チャットボット")
        st.subheader(f"🗣️ {st.session_state.current_chat}")
        st.markdown(f"**モデル:** {st.session_state.gpt_model} | **モード:** {st.session_state.design_mode}")

        # -- メッセージ表示 --
        st.markdown('<div class="chat-body">', unsafe_allow_html=True)
        for m in get_messages():
            message_class = "user-message" if m["role"] == "user" else "assistant-message"
            with st.chat_message(m["role"]):
                st.markdown(f'<div class="{message_class}">{m["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        msgs = get_messages()  # 現在のメッセージリストを取得

        # --- ボタン常設ロジック ---
        if msgs and msgs[-1]["role"] == "assistant":
            last_turn_key = (st.session_state.sid, len(msgs))
            if st.button("🧪 他モデルと比較する", key=f"compare_{last_turn_key}"):
                st.session_state["compare_dialog_open"] = last_turn_key
                st.rerun()

        # ==== ダイアログ的な比較結果表示 ====
        turn_key_current = st.session_state.get("compare_dialog_open")

        if turn_key_current:
            comp = st.session_state.comparison_results.get(turn_key_current, {})
            with st.expander("🧪 モデル比較結果", expanded=True):
                if not comp:
                    st.info("⏳ 比較結果を取得中です。数秒後に再表示してください。")
                else:
                    for (mdl, temp), ans in comp.items():
                        st.markdown(f"#### ⮞ `{mdl}` (temperature={temp})")
                        st.markdown(ans)
                        st.divider()

            if st.button("❌ 閉じる", key=f"close_{turn_key_current}"):
                del st.session_state["compare_dialog_open"]
                st.rerun()

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

            logger.info("💬 gen_start — mode=%s model=%s use_rag=%s sid=%s",
                st.session_state.design_mode,
                st.session_state.gpt_model,
                st.session_state.get("use_rag", True),
                st.session_state.sid)

            try:
                # ---------- RAG あり ----------
                if st.session_state.get("use_rag", True):
                    st.session_state["last_answer_mode"] = "RAG"

                    t_api = time.perf_counter()
                    rag_res = generate_answer(
                            prompt=prompt,
                            question=user_prompt,
                            collection=st.session_state.rag_collection,
                            rag_files=st.session_state.rag_files,  # ← ここを追加
                            top_k=4,
                            model=st.session_state.gpt_model,
                            chat_history=msgs,
                        )
                    api_elapsed = time.perf_counter() - t_api
                    assistant_reply = rag_res["answer"]
                    sources = rag_res["sources"]

                    logger.info("💬 GPT done — tokens≈%d  api_elapsed=%.2fs  sources=%d",
                                    len(assistant_reply.split()), api_elapsed, len(sources))

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

                    import time
                    t_api = time.perf_counter()

                    # APIを呼び出し
                    resp = client.chat.completions.create(**params)

                    api_elapsed = time.perf_counter() - t_api

                    assistant_reply = resp.choices[0].message.content
                    sources = []

                    logger.info("💬 GPT done — tokens≈%d  api_elapsed=%.2fs",
                                    len(assistant_reply.split()), api_elapsed)
                    
            except Exception as e:
                logger.exception("❌ answer_gen failed — %s", e)
                st.error("回答生成時にエラーが発生しました")

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

            # ✅ ここで非同期比較を開始  ---------------------------
            launch_background_comparisons(prompt, user_prompt, msgs)

            # 会話内容を Sheets へ記録
            post_log(user_prompt, assistant_reply, prompt)

            # チャットタイトル自動生成（初回応答後）
            # if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
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
