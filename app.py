
import streamlit as st
import boto3
from typing import List, Dict, Any
import time, functools
import os
import pandas as pd
import json

from src.rag_preprocess import preprocess_files
from src.rag_qa import generate_answer_with_equipment, detect_equipment_from_question
from src.startup_loader import initialize_equipment_data
from src.logging_utils import init_logger
from src.sheets_manager import log_to_sheets, get_sheets_manager, send_prompt_to_model_comparison

# === 🔥 LangChain統合のインポートを追加 ===
from src.langchain_chains import generate_smart_answer_with_langchain

import yaml
import streamlit_authenticator as stauth
from streamlit.components.v1 import html
import uuid

import threading
import queue
from typing import Optional
import atexit
import copy
import base64

# Azure OpenAI関連のインポートを追加
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

st.set_page_config(page_title="Claude + RAG Chatbot", page_icon="💬", layout="wide")

logger = init_logger()

# AWS Bedrock設定を追加
def setup_bedrock_client():
    """AWS Bedrock設定"""
    try:
        aws_access_key_id = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
        aws_secret_access_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
        aws_region = st.secrets.get("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
    except:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    if not aws_access_key_id or not aws_secret_access_key:
        st.error("AWS Bedrock の設定が不足しています。環境変数またはSecrets.tomlを確認してください。")
        st.stop()
    
    return boto3.client(
        'bedrock-runtime',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

# Azure OpenAI設定を追加
def setup_azure_client():
    """Azure OpenAI設定"""
    if not AZURE_OPENAI_AVAILABLE:
        st.error("Azure OpenAI ライブラリがインストールされていません。pip install openai を実行してください。")
        st.stop()
    
    try:
        azure_endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT"))
        azure_api_key = st.secrets.get("AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))
        azure_api_version = st.secrets.get("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
    except:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    if not azure_endpoint or not azure_api_key:
        st.error("Azure OpenAI の設定が不足しています。環境変数またはSecrets.tomlを確認してください。")
        st.stop()
    
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version
    )

# Claude用のモデル名マッピング
CLAUDE_MODEL_MAPPING = {
    "claude-4-sonnet": "apac.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-3.7": "apac.anthropic.claude-3-7-sonnet-20250219-v1:0"
}

# Azure OpenAI用のモデル名マッピングを追加
AZURE_MODEL_MAPPING = {
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o"
}

def get_claude_model_name(model_name: str) -> str:
    """Claude表示名をBedrockモデルIDに変換"""
    return CLAUDE_MODEL_MAPPING.get(model_name, model_name)

def normalize_filename(filename: str) -> str:
    return filename.strip().lower().replace(" ", "").replace("（", "(").replace("）", ")")

def call_claude_bedrock(client, model_id: str, messages: List[Dict], max_tokens: int = 4096, temperature: float = 0.0):
    """AWS Bedrock Converse API経由でClaudeを呼び出し"""
    system_prompts = []
    conversation_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_prompts.append({"text": msg["content"]})
        else:
            conversation_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })
    
    converse_params = {
        "modelId": model_id,
        "messages": conversation_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    }
    
    if system_prompts:
        converse_params["system"] = system_prompts
    
    response = client.converse(**converse_params)
    
    if response.get('stopReason') == 'error':
        raise Exception(f"Claude API Error: {response.get('output', {}).get('message', 'Unknown error')}")
    
    return response['output']['message']['content'][0]['text']

def call_azure_gpt(client, model_name: str, messages: List[Dict], max_tokens: int = 4096, temperature: float = 0.0):
    """Azure OpenAI経由でGPTを呼び出し"""
    formatted_messages = []
    
    for msg in messages:
        if msg["role"] in ["system", "user", "assistant"]:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    response = client.chat.completions.create(
        model=AZURE_MODEL_MAPPING.get(model_name, model_name),
        messages=formatted_messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message.content

# =====  認証設定の読み込み ============================================================
with open('./config.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ===== 非同期ログ処理クラス =====
class StreamlitAsyncLogger:
    def __init__(self):
        self.log_queue = queue.Queue(maxsize=100)
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.stats = {"processed": 0, "errors": 0, "last_process_time": None, "last_error_time": None, "last_error_msg": None}
        self._lock = threading.Lock()
        self.start_worker()
    
    def start_worker(self):
        with self._lock:
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.shutdown_event.clear()
                self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="StreamlitAsyncLogger")
                self.worker_thread.start()
                logger.info("🚀 StreamlitAsyncLogger worker started")
    
    def _worker_loop(self):
        while not self.shutdown_event.is_set():
            try:
                log_data = self.log_queue.get(timeout=2.0)
                if log_data is None:
                    break
                self._process_log_safe(log_data)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                with self._lock:
                    self.stats["errors"] += 1
                    self.stats["last_error_time"] = time.time()
                    self.stats["last_error_msg"] = str(e)
                logger.error("❌ AsyncLogger worker error — %s", e, exc_info=True)

    def _process_log_safe(self, log_data: dict):
        try:
            # 実際のログ処理
            manager = get_sheets_manager()
            if manager and manager.is_connected:
                success = log_to_sheets(
                    input_text=log_data["input_text"],
                    output_text=log_data["output_text"],
                    prompt=log_data["prompt"],
                    chat_title=log_data.get("chat_title", "未設定"),
                    user_id=log_data.get("user_id", "unknown"),
                    session_id=log_data.get("session_id", "unknown"),
                    mode=log_data.get("mode", "unknown"),
                    model=log_data.get("model", "unknown"),
                    temperature=log_data.get("temperature", 0.0),
                    max_tokens=log_data.get("max_tokens"),
                    use_rag=log_data.get("use_rag", False)
                )
            
            with self._lock:
                self.stats["processed"] += 1
                self.stats["last_process_time"] = time.time()
            
        except Exception as e:
            with self._lock:
                self.stats["errors"] += 1
                self.stats["last_error_time"] = time.time()
                self.stats["last_error_msg"] = str(e)
            logger.error("❌ Async log processing failed — %s", e, exc_info=True)
    
    def post_log_async(self, input_text: str, output_text: str, prompt: str, send_to_model_comparison: bool = False):
        if not self.worker_thread or not self.worker_thread.is_alive():
            self.start_worker()
        
        log_data = {
            "input_text": input_text,
            "output_text": output_text,
            "prompt": prompt,
            "timestamp": time.time(),
            "chat_title": st.session_state.get("current_chat", "未設定"),
            "user_id": st.session_state.get("username", "unknown"),
            "session_id": st.session_state.get("sid", "unknown"),
            "mode": st.session_state.get("design_mode", "unknown"),
            "model": st.session_state.get("claude_model", "unknown"),
            "temperature": st.session_state.get("temperature", 0.0),
            "max_tokens": st.session_state.get("max_tokens"),
            "use_rag": st.session_state.get("use_rag", False)
        }
        
        try:
            self.log_queue.put_nowait(log_data)
            logger.info("📝 Log queued — queue_size=%d", self.log_queue.qsize())
        except queue.Full:
            logger.error("❌ Log queue is full — dropping log entry")
    
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queue_size": self.log_queue.qsize(),
                "worker_alive": self.worker_thread.is_alive() if self.worker_thread else False,
                "shutdown_requested": self.shutdown_event.is_set(),
                "stats": self.stats.copy()
            }

def get_async_logger() -> StreamlitAsyncLogger:
    if "async_logger" not in st.session_state:
        st.session_state.async_logger = StreamlitAsyncLogger()
    
    async_logger = st.session_state.async_logger
    if not async_logger.worker_thread or not async_logger.worker_thread.is_alive():
        logger.warning("⚠️ AsyncLogger instance invalid, creating new one")
        st.session_state.async_logger = StreamlitAsyncLogger()
        async_logger = st.session_state.async_logger
    
    return async_logger

def post_log_async(input_text: str, output_text: str, prompt: str, send_to_model_comparison: bool = False):
    try:
        logger_instance = get_async_logger()
        logger_instance.post_log_async(input_text, output_text, prompt, send_to_model_comparison)
    except Exception as e:
        logger.error("❌ post_log_async failed — %s", e)

# =====  基本設定  ============================================================
bedrock_client = setup_bedrock_client()

# =====  ログインUIの表示  ============================================================
authenticator.login()

if st.session_state["authentication_status"]:
    name = st.session_state["name"]
    username = st.session_state["username"]

    logger.info("🔐 login success — user=%s  username=%s", name, username)

    # 設備データ初期化
    if st.session_state.get("equipment_data") is None:
        logger.info("🔍 設備データ初期化開始")
        
        try:
            drive_folder_id = None
            try:
                drive_folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
                if drive_folder_id:
                    drive_folder_id = drive_folder_id.strip()
            except Exception as secrets_error:
                logger.error("🔍 secrets取得エラー: %s", secrets_error)
            
            if drive_folder_id:
                st.info("📁 Google Driveからファイルを読み込み中...")
                param = f"gdrive:{drive_folder_id}"
                res = initialize_equipment_data(param)
                logger.info("📂 Google Driveから設備データ初期化完了")
            else:
                st.info("📂 ローカル rag_data フォルダからファイルを読み込み中...")
                res = initialize_equipment_data("rag_data")
                logger.info("📂 ローカルディレクトリから設備データ初期化完了")
            
            st.session_state.equipment_data = res["equipment_data"]
            st.session_state.equipment_list = res["equipment_list"]
            st.session_state.category_list = res["category_list"]
            st.session_state.rag_files = res["file_list"]

            logger.info("📂 設備データ初期化完了 — 設備数=%d  ファイル数=%d",
                    len(res["equipment_list"]), len(res["file_list"]))
            
        except Exception as e:
            logger.exception("❌ 設備データ初期化失敗 — %s", e)
            st.error(f"設備データ初期化中にエラーが発生しました: {e}")
    else:
        logger.info("🔍 設備データは既に初期化済み")

    # --------------------------------------------------------------------------- #
    #                         ★ 各モード専用プロンプト ★                           #
    # --------------------------------------------------------------------------- #
    DEFAULT_PROMPTS: Dict[str, str] = {
        "暗黙知法令チャットモード": """
    あなたは建築電気設備設計のエキスパートエンジニアです。
    今回の対象は **複合用途ビルのオフィス入居工事（B工事）** に限定されます。
    以下の知識と技術をもとに、対話を通じて不足情報を質問しつつ、根拠を示した実務アドバイスを行ってください。
    専門用語は必要に応じて解説を加え、判断の背景にある理由を丁寧に説明します。
    ────────────────────────────────
    ## 【回答方針】
    **重要：以下の各事項は「代表的なビル（丸の内ビルディング）」を想定して記載しています。他のビルでは仕様や基準が異なる可能性があることを、回答時には必ず言及してください。**
    **注意：過度に込み入った条件の詳細説明をユーザーに求めることは避け、一般的な設計基準に基づく実務的な回答を心がけてください。**
    ### ■ 暗黙知情報不足時の対応プロセス
    現在保有している暗黙知情報では適切な回答ができない場合は、以下の手順で対応してください：
    1. **現状把握の明示**
    - 「現在の暗黙知情報では、○○ビルの△△設備について十分な情報がございません」と明確に伝える
    - 一般的な設計基準に基づく暫定的な回答がある場合は、その旨を明記して提供する
    2. **逆質問の実行**
    - ユーザーの実務経験や現場知識を活用するため、具体的な逆質問を行う
    - 質問例：「○○ビルでは△△設備についてどのような仕様・基準をお使いでしょうか？」
    - 「過去の類似案件では、どのような対応をされましたか？」
    3. **暗黙知情報の記録**
    - ユーザーから有効な回答が得られた場合、以下のフォーマットで情報を記録する：
    ```
    【暗黙知情報：ビル名、設備名】
    内容：（ユーザーから得られた情報の要約）
    適用条件：（どのような条件下で適用されるか）
    ```
    4. **情報活用とフィードバック**
    - 得られた暗黙知情報を元に、改めて適切な回答を提供する
    - 「この情報は今後の設計業務改善に活用させていただきます」と感謝の意を示す
    ────────────────────────────────
    ## 【工事区分について】
    - **B工事**：本システムが対象とする工事。入居者負担でビル側が施工する工事
    - **C工事**：入居者が独自に施工する工事（電話・LAN・防犯設備など）
    - 本システムでは、C工事設備については配管類の数量算出のみを行います
    ────────────────────────────────
    ## 【今回の設計対象ビル】
    - ビル正式名称: 丸の内ビルディング
    - ビル略称: （記載なし）
    - 床面積: 1,949.00㎡ (589.57坪)～2,284.00㎡ (690.91坪)
    - 床荷重: 500kg/㎡HDZ(重荷重対応)800kg/㎡
    - 天井高: 2,800mm（ＯＡフロア設置後）
    - コンセント容量: 75VA/㎡ 増設対応可能
    - OAフロア: 高さ100mm
    - トイレ: 男女各1カ所
    - 給湯室: 各2カ所
    - 管轄の消防本部/局:東京消防庁
    - 管轄の消防署:丸の内消防署
    - 自火報設備（基準階） メーカー: 能美防災㈱
    - 自火報設備（基準階） 型　番: （記載なし）
    - 自火報設備（基準階） 感知器種別: 煙感知器
    - 自火報設備（基準階） 受信機 更新時期: （記載なし）
    - 自火報設備（基準階） 請負対象: （記載なし）
    - 非常放送（基準階） メーカー: ホーチキ(株)(TOA)
    - 非常放送（基準階） 型　番: （記載なし）
    - 非常放送（基準階） 放送設備 更新時期: （記載なし）
    - 非常照明（基準階） メーカー: パナソニック
    - 非常照明（基準階） 型　番: （記載なし）
    - 非常照明（基準階） バッテリー 内臓・別置型: 内臓・別置
    - 非常照明（基準階） 店舗　BAT　(白図): 店舗　BAT　(白図)
    - 誘導灯（基準階） メーカー: パナソニック
    - 誘導灯（基準階） 型　番: （記載なし）
    - 誘導灯（基準階） 型式 B級・C級: B級 BH型
    - 誘導灯（基準階） 東京消防庁基準: 出入口や誘導灯が障害物により視認できない場合であっても、人が概ね５ｍ移動することにより出入 口や誘導灯を視認できる場合は、容易に見とおしできるものとみなす
    - 所在地: 東京都千代田区丸の内二丁目4番1号
    - アクセス: JR各線、丸ノ内線「東京駅」 直結
    - 竣工年月: 2002年8月
    - リニューアル年月: 2025年2月
    - 規模地上: 37階・地下 4階・塔屋 2階
    - 構造: 鉄骨造・一部鉄骨鉄筋コンクリート造
    - 延床面積: 159,901.38㎡ (48,370.17坪)
    - 有効面積: 49,195.720㎡ (14,881.71坪)
    - 敷地面積: 10,029.45㎡ (3,033.91坪)
    - 建物高さ: 最高部 179.0ｍ
    - 設計監理: 三菱地所設計
    - エレベーター: 乗用 23台　貨物用 2台
    - 空調設備: 各階8分割にゾーニング
    - 基準冷暖房供給時間（事務所）: 平日 8:30-19:00、土曜 設定無し、日祝 設定無し
    - 基準冷暖房供給時間（店舗）: 全日 9:30-24:00
    - 通信インフラ: 光ファイバ敷設済
    - 開閉館時間: 全日 6:00-24:30
    - 管理形態: 24時間有人管理
    - 夜間入退館方法: インターフォンによる呼出
    - ガレージ収容台数: 376台 (高さ 3,000mm)
    - ガレージ営業時間: 全日 6:00-24:00、定期契約車は24時間入出庫可
    - ビル管理室電話番号: 03-3240-8251
    - 用途区分（消防）: 16項(イ)
    ────────────────────────────────
    ## 【コンセント設備（床）】
    ### ■ 図面の指示と基本的な割り振り
    - 図面や要望書の指示を優先（単独回路や専用ELB等）
    - 機器やデスクの配置をもとにグループ化
    - 一般的なオフィス机は複数の座席をまとめて1回路
    - 機器の消費電力が高い場合や同時使用想定で回路分割
    ### ■ 机・椅子（デスク周り）の標準設計とOAタップ仕様
    - **個人用デスク**： 1席ごとにOAタップ1個、6席ごとに1回路（300VA/席）
    - 使用機器：分岐 4口タップ（300VA）
    - **フリーアドレスデスク**：1席ごとにOAタップ1個、8席ごとに1回路（150VA/席）
    - 使用機器：分岐 4口タップ（150VA）
    - **昇降デスク**：1席ごとにOAタップ1個、2席ごとに1回路（600VA/席）
    - 使用機器：分岐 4口タップ（600VA）
    - **会議室テーブル**：4席ごとにOAタップ1個、12席ごとに1回路（150VA/席）
    - 使用機器：分岐 4口タップ（600VA）
    ### ■ 設備機器の設計とOAタップ仕様
    - **単独回路が必要な機器**
    - 通常機器用：単独 2口タップ（単独回路）
    - 水気のある機器用：単独 ELB 2口タップ（単独回路、ELB付き）
    - 対象機器：複合機（コピー機）、プリンター、シュレッダー、テレブース、自動販売機、冷蔵庫、ウォーターサーバー、電子レンジ、食器洗い乾燥機、コーヒーメーカー、ポット、造作家具（什器用コンセント）、インターホン親機、セキュリティシステム、等
    - **サーバーラック**：什器1つにつき2回路必要（単独 2口タップを2個設置）
    - **分岐回路でもよい機器**
    - 使用機器：分岐 4口タップ（150VA/300VA/600VA/1200VA）（容量に応じて選択）
    - 対象機器：ディスプレイ（会議室、応接室、役員室）、テレビ（共用）、スタンド照明、ロッカー（電源供給機能付）、等
    - 300〜1200VA程度の機器は近い位置で1回路にまとめ可能（1500VA上限）
    ### ■ 特殊エリアの電源
    - パントリー：最低OAタップ5個と5回路
    - エントランス：最低OAタップ1個と1回路
    - プリンター台数：20人に1台が目安、40人に1台が確保できてなければ電源の追加を提案
    ────────────────────────────────
    ## 【コンセント設備（壁）※清掃用電源】
    ### ■ 用途と設置考え方
    - 清掃時に掃除機を接続するための電源（入居企業は使用不可）
    - 見積図面では提案するが、入居企業の要望により省略も可能
    - 設置位置は主に扉横
    ### ■ 配置判断ルール
    - 清掃時の動線（≒避難経路）を考慮して配置
    - 扉を挟んだどちら側に設置するかの精査が必要
    - 各部屋の入口付近に最低1箇所
    ────────────────────────────────
    ## 【コンセント設備（壁）※客先指示】
    ### ■ 設置基準
    - 見積依頼図に指示された場所、指示された仕様で設置
    - 客先からの特殊指示（単独回路、専用ELB等）を最優先
    - 図面上の明示がなくても打合せ記録等で指示があれば対応
    ### ■ 追加提案判断
    - 見積図に指示がなくても、使用目的が明確な場合は追加提案
    - 特殊機器（給湯器、加湿器等）の近くには設置を提案
    ────────────────────────────────
    ## 【コンセント設備（天井）】
    ### ■ 設置基準
    - 見積依頼図に指示があった場所に設置
    - 電源が必要な天井付近の機器がある場合に1個設置
    ### ■ 対象機器
    - プロジェクター
    - 電動スクリーン
    - 電動ブラインド
    - 壁面発光サイン
    - その他天井付近に設置される電気機器
    ────────────────────────────────
    ## 【自動火災報知設備】
    ### ■ 感知器の種類・仕様
    - **丸ビル標準**
    - 廊下：煙感知器スポット型2種（全ビル共通）
    - 居室：煙感知器スポット型2種（丸ビル標準）
    - 厨房：定温式スポット型（1種）
    - **他ビルでの例**
    - 居室で熱感知器（差動式スポット型1種等）を使用するビルもある
    - 天井面中央付近、または障害を避けて煙が集まりやすい位置に設置
    ### ■ 設置基準
    - 廊下：端点から15m以内、感知器間30m以内
    - 居室：面積150m²ごとに1個（切り上げ）
    - 煙を遮る障害物がある場合は個数増
    - 天井高2.3m未満または40m²未満の居室は入口付近
    - 吸気口付近に設置、排気口付近は避ける
    - 防火シャッター近くは専用感知器（煙感知器スポット型3種等）
    ### ■ 特殊条件での判断指針
    - **欄間オープン内の侵入防止バー**：面積阻害の程度と補償措置について消防署への事前相談を推奨
    ────────────────────────────────
    ## 【非常放送設備】
    ### ■ スピーカ設置基準
    - 到達距離10m以内（各居室・廊下を半径10mの円でカバー）
    - パーテーションや什器による遮音は考慮しない（半径10mの円は不変）
    ### ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷200㎡）の切り上げ
    - 200㎡は「L級スピーカー」の有効範囲円（半径10m）に内接する正方形の面積
    ### ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 省略可能条件（居室・廊下は6m²以下、その他区域は30m²以下、かつ隣接スピーカから8m以内なら省略可能）は適用しない（丸ビル方針）
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置
    ────────────────────────────────
    ## 【誘導灯設備】
    ### ■ 種類・採用機種
    - 避難口誘導灯・通路誘導灯のみ使用
    - 両者ともB級BH型（20A形）のみ使用（丸ビル標準）
    ### ■ 設置箇所・有効距離
    - 避難口誘導灯：最終避難口、または最終避難口に通じる避難経路上の扉
    有効距離30m（シンボル無）／20m（矢印付き）
    - 通路誘導灯：廊下の曲がり角や分岐点、または避難口誘導灯の有効距離補完
    有効距離15m
    ### ■ 配置判断
    - 扉開閉・パーテーション・背の高い棚などで視認阻害→位置変更または追加
    ### ■ 特殊ケースの判断指針
    - **扉の直上扱いの範囲**：扉周辺3m程度が一般的な目安だが、具体的な設置可能範囲については消防署の担当者判断となるため、境界的なケースでは事前相談を推奨
    ────────────────────────────────
    ## 【非常照明設備】
    ### ■ 照度条件
    - 常温下の床面で1lx以上を確保（建築基準法施行令第126条の5）
    - 照度計算は逐点法を用いる（カタログの1lx到達範囲表使用）
    ### ■ 器具仕様・種別
    - バッテリー別置型：ビル基本設備分（入居前既設分）
    - バッテリー内蔵型：B工事追加分（間仕切り変更などで追加した分）
    ### ■ 設置判断ルール
    - 天井高別の1lx到達範囲表を用い、器具間隔を決定
    - パーテーション・什器で遮光の恐れがあれば器具を追加
    - 2018年改正の個室緩和（30m²以下は不要）は適用しない（丸ビル方針）
    ### ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷50㎡）の切り上げ
    - 50㎡は新丸ビルにおける非常照明設備の有効範囲円（半径5.0m）に内接する正方形の面積
    ### ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置
    ────────────────────────────────
    ## 【照明制御設備（照度センサ）】
    ### ■ 設置判断ルール
    - 天井高別の有効範囲表を用い、器具間隔を決定
    - パーテーション・什器で遮られる恐れがあれば器具を追加
    ### ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷28㎡）の切り上げ
    - 28㎡は新丸ビルにおける照度センサの有効範囲円（半径3.75m）に内接する正方形の面積
    ### ■ 設置に関する注意点
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 本来はビル仕様や他設備との取り合いを考慮し、全域が有効範囲内に収まるよう配置
    ────────────────────────────────
    ## 【照明制御設備（スイッチ）】
    ### ■ 設置判断ルール
    - 新規に間仕切りされた領域に対してそれぞれ設置
    - 設置するスイッチ数は領域の大きさや扉の配置、制御の分け方による
    ### ■ 概算数量計算
    - 概算個数＝（領域面積◯㎡÷20㎡）の切り上げ
    - 算出個数に基づき、「最終避難口」以外の「扉」の横に配置（最終避難口の横にはビル基本のスイッチがあるため追加設置不要）
    ### ■ 配置ルール
    - 初期概算見積り段階では、あえて数量に余裕をもたせる計算式を採用
    - 配置数2個以上かつ扉数2個以上の場合は、領域内の「扉」の横に均等に配置
    - 扉数＞個数の場合は最終避難口への距離が短い扉から優先的に配置
    - 本来は入退室ルート（≒避難経路）に基づく動線計画に従い、設置位置を精査
    ────────────────────────────────
    ## 【テレビ共聴設備】
    ### ■ 設置基準
    - 見積依頼図に指示があった場所に設置
    - テレビ共聴設備が必要な什器がある場所に1個設置
    ### ■ 設置が必要な部屋・什器
    - 会議室：最低1個は設置
    - 応接室：最低1個は設置
    - 役員室：最低1個は設置
    - ディスプレイ（会議室、応接室、役員室にあるもの）
    - テレビ（共用のもの）
    ────────────────────────────────
    ## 【電話・LAN設備（配管）】【防犯設備（配管）】（C工事設備）
    ### ■ 業務基本原則
    - 本設備はC工事のため、B工事では配管の設置のみを行う
    - 基本的には客先から図面を受領して見積りを作成
    - C工事会社から配管の設置のみ依頼される場合が多い
    ### ■ 概算見積りの考え方
    - 概算段階では配管図を作成せず、細部計算を省略することが一般的
    - 「設備数×○m」という形式で概算を算出
    - 各ビル・各設備ごとの「黄金数字（○m）」を考慮した設計が必要
    ────────────────────────────────
    ## 【動力設備（配管、配線）】
    ### ■ 適用場面と業務原則
    - 基本的には客先から図面を受領して見積りを作成
    - 店舗（特に飲食店）では必要性が高い
    - オフィスでも稀に必要となるケースがある
    ### ■ 概算見積りの特徴
    - 概算段階では配置平面図よりも、必要な動力設備の種類と数をまとめた表から算出
    - 表を読み解いて必要数を算出し見積りに反映
    ### ■ オフィスでの対応
    - オフィスで必要な場合：動力用の分電盤、配線・配管を設置
    - 詳細な設計検討が必要（概算見積対応はできない）
    ────────────────────────────────
    ## 【消防署事前相談の指針】
    ### ■ 事前相談が必要な状況
    法令のルールが競合する場合や細かな仕様で判断が分かれる場合は、**必ず消防署への事前相談を行う**ことを推奨してください。
    ### ■ 事前相談のタイミング
    - **着工届出書提出時**：通常の手続きの中で相談
    - **軽微な工事で着工届が不要な場合**：別途消防署に出向いて相談
    ### ■ 法令競合の典型例
    1. **自火報（煙感知器）関連**
    - 狭い部屋内で「吹き出しから離して設置」「吸込口付近に設置」「入口付近に設置」を同時に満たす場所がない場合
    ### ■ 細かな仕様判断の典型例
    1. **自火報（煙感知器）関連**
    - 欄間オープン内に侵入防止バーがあって面積が阻害されている場合
    - 阻害された面積分を補うように欄間オープンの面積を広げている場合の扱い
    2. **避難口誘導灯関連**
    - 扉の直上扱いとして矢印シンボルなしを設置してよい範囲（扉周辺3m程度が目安だが、最終的には担当者判断）
    - パーテーション等による視認阻害の程度と補完誘導灯設置の要否
    ────────────────────────────────
    ## 【重要な注意事項】
    1. **ビル仕様の違い**：上記の内容は丸の内ビルディングを基準としています。他のビルでは異なる仕様・基準が適用される可能性があります。
    2. **過度な詳細要求の回避**：ユーザーに対して、込み入った条件の詳細説明を過度に求めることは避けてください。
    3. **工事区分の明確化**：B工事とC工事の区分を常に意識し、C工事設備については配管類のみを扱うことを明確にしてください。
    4. **法令準拠**：検索結果の言い回しをそのまま複製することを避け、直接引用以外のすべてを自分の言葉で表現します。
    5. **判断困難時の対応**：法令競合や細かな仕様判断で迷いが生じた場合は、必ず消防署への事前相談を推奨し、一般的な傾向は示しつつも最終判断は消防署見解に委ねることを明記してください。
    6. **暗黙知収集**：現在の知識で対応できない質問については、積極的にユーザーの実務経験を活用し、将来の暗黙知データベース拡充に貢献してください。
    7. **資料からの原文抜粋の禁止**：ユーザーから提供された資料や図面からの原文抜粋は行わず、必ず自分の言葉で説明してください。
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

    # セッション変数
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "chat_sids" not in st.session_state:
        st.session_state.chat_sids = {"New Chat": str(uuid.uuid4())}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"
    if "sid" not in st.session_state:
        st.session_state.sid = st.session_state.chat_sids["New Chat"]
    if "edit_target" not in st.session_state:
        st.session_state.edit_target = None
    if "rag_files" not in st.session_state:
        st.session_state.rag_files: List[Dict[str, Any]] = []
    if "design_mode" not in st.session_state:
        st.session_state.design_mode = list(DEFAULT_PROMPTS.keys())[0]
    if "prompts" not in st.session_state:
        st.session_state.prompts = DEFAULT_PROMPTS.copy()
    if "claude_model" not in st.session_state:
        st.session_state.claude_model = "claude-4-sonnet"
    if "selected_equipment" not in st.session_state:
        st.session_state.selected_equipment = None
    if "selection_mode" not in st.session_state:
        st.session_state.selection_mode = "manual"
    
    user_prompt: str | None = None

    # ヘルパー関数
    def get_messages() -> List[Dict[str, str]]:
        title = st.session_state.current_chat
        return st.session_state.chats.setdefault(title, [])
    
    def new_chat():
        title = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[title] = []
        st.session_state.chat_sids[title] = str(uuid.uuid4())
        st.session_state.current_chat = title
        st.session_state.sid = st.session_state.chat_sids[title]
        logger.info("➕ new_chat — sid=%s  title='%s'", st.session_state.sid, title)
        st.rerun()

    def switch_chat(title: str):
        if title not in st.session_state.chat_sids:
            st.session_state.chat_sids[title] = str(uuid.uuid4())
        st.session_state.current_chat = title
        st.session_state.sid = st.session_state.chat_sids[title]
        logger.info("🔀 switch_chat — sid=%s  title='%s'", st.session_state.sid, title)
        st.rerun()

    def generate_chat_title(messages):
        if len(messages) >= 2:
            prompt = f"以下の会話の内容を25文字以内の簡潔なタイトルにしてください:\n{messages[0]['content'][:200]}"
            try:
                title_messages = [{"role": "user", "content": prompt}]
                response = call_claude_bedrock(
                    bedrock_client, 
                    get_claude_model_name("claude-4-sonnet"),
                    title_messages,
                    max_tokens=30
                )
                return response.strip('"').strip()
            except Exception as e:
                logger.error(f"Chat title generation failed: {e}")
                return f"Chat {len(st.session_state.chats) + 1}"
        return f"Chat {len(st.session_state.chats) + 1}"

    # CSS
    st.markdown("""
        <style>
        :root{ --sidebar-w:260px; --pad:1rem; }
        aside[data-testid="stSidebar"]{width:var(--sidebar-w)!important;}
        .chat-body{max-height:70vh;overflow-y:auto;}
        .stButton button {font-size: 16px; padding: 8px 16px;}
        .user-message, .assistant-message {
            border-radius: 10px; padding: 8px 12px; margin: 5px 0; border-left: 3px solid;
        }
        .user-message { border-left-color: #4c8bf5; }
        .assistant-message { border-left-color: #ff7043; }
        </style>
        """, unsafe_allow_html=True)

    # サイドバー
    with st.sidebar:
        st.markdown(f"👤 ログインユーザー: `{name}`")
        authenticator.logout('ログアウト', 'sidebar')
        st.divider()

        # チャット履歴
        st.header("💬 チャット履歴")
        for title in list(st.session_state.chats.keys()):
            if st.button(title, key=f"hist_{title}"):
                switch_chat(title)
        if st.button("➕ 新しいチャット"):
            new_chat()
        st.divider()

        # モデル選択
        st.markdown("### 🤖 モデル選択")
        model_options = {
            "claude-4-sonnet": "Claude 4 Sonnet (最高性能・推奨)",
            "claude-3.7": "Claude 3.7 Sonnet (高性能)",
            "gpt-4.1": "GPT-4.1 (最新・高性能)",
            "gpt-4o": "GPT-4o(高性能)"
        }
        st.session_state.claude_model = st.selectbox(
            "使用するモデルを選択",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.claude_model) if st.session_state.claude_model in model_options else 0,
        )

        # 詳細設定
        with st.expander("🔧 詳細設定"):
            st.slider("応答の多様性", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if "max_tokens" not in st.session_state or st.session_state.get("max_tokens") is None:
                    st.session_state["max_tokens"] = 4096
                max_tokens_text = st.text_input("最大応答長（トークン数）", value=str(st.session_state.get("max_tokens", 4096)), key="max_tokens_text")
            with col2:
                apply_button = st.button("✅ 適用", key="apply_max_tokens")
            
            if apply_button:
                if max_tokens_text.strip() == "":
                    st.session_state["max_tokens"] = None
                else:
                    try:
                        max_tokens_value = int(max_tokens_text.strip())
                        if max_tokens_value > 0:
                            st.session_state["max_tokens"] = max_tokens_value
                            st.success(f"✅ 最大トークン数を {max_tokens_value:,} に設定")
                            st.rerun()
                    except ValueError:
                        st.error("❌ 有効な数値を入力してください")

        # === 🔥 LangChain設定 ===
        st.divider()
        st.markdown("### 🔗 LangChain設定")
        use_langchain = st.checkbox("LangChainを使用する", value=st.session_state.get("use_langchain", False), key="use_langchain")
        
        if use_langchain:
            st.info("🔗 LangChainモード: より標準化された処理を使用")
        else:
            st.info("🔧 従来モード: 既存の直接API呼び出しを使用")

        st.divider()

        # モード選択
        st.markdown("### ⚙️ 設計対象モード")
        st.session_state.design_mode = st.radio("対象設備を選択", options=list(st.session_state.prompts.keys()), index=0)

        st.divider()

        # 設備選択
        st.markdown("### 🔧 対象設備選択")
        available_equipment = st.session_state.get("equipment_list", [])
        
        if not available_equipment:
            st.error("❌ 設備データが読み込まれていません")
        else:
            selection_mode = st.radio("選択方式", ["設備名で選択", "カテゴリから選択", "自動推定"], index=0)
            
            if selection_mode == "設備名で選択":
                selected_equipment = st.selectbox("設備を選択してください", options=[""] + available_equipment, index=0)
                st.session_state["selected_equipment"] = selected_equipment if selected_equipment else None
                st.session_state["selection_mode"] = "manual"
            elif selection_mode == "カテゴリから選択":
                st.session_state["selection_mode"] = "category"
            else:
                st.info("🤖 質問文から設備を自動推定して回答します")
                st.session_state["selected_equipment"] = None
                st.session_state["selection_mode"] = "auto"

        # LangChain診断
        st.divider()
        st.markdown("### 🧪 LangChain診断")
        
        if st.button("🔗 LangChain接続テスト"):
            try:
                from src.langchain_models import test_model_creation
                from src.langchain_chains import test_chain_creation
                
                model_success = test_model_creation()
                chain_success = test_chain_creation()
                
                if model_success and chain_success:
                    st.success("🎉 LangChain統合テスト完了！")
                else:
                    st.error("⚠️ 一部のテストが失敗しました")
            except Exception as e:
                st.error(f"❌ LangChainテストエラー: {e}")

    # メイン画面表示
    if not st.session_state.get("edit_target"):
        st.title("💬 Claude + 設備資料チャットボット")
        st.subheader(f"🗣️ {st.session_state.current_chat}")
        st.markdown(f"**モデル:** {st.session_state.claude_model} | **モード:** {st.session_state.design_mode}")

        # メッセージ表示
        for m in get_messages():
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # 入力欄
        user_prompt = st.chat_input("メッセージを入力…")

    # 応答生成
    if user_prompt and not st.session_state.get("edit_target"):
        msgs = get_messages()
        msgs.append({"role": "user", "content": user_prompt})

        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.status(f"🤖 {st.session_state.claude_model} で回答を生成中...", expanded=True):
            prompt = st.session_state.prompts[st.session_state.design_mode]

            try:
                use_langchain = st.session_state.get("use_langchain", False)
                
                if use_langchain:
                    # === LangChain統一処理 ===
                    langchain_params = {
                        "prompt": prompt,
                        "question": user_prompt,
                        "model": st.session_state.claude_model,
                        "equipment_data": st.session_state.equipment_data,
                        "chat_history": msgs,
                    }
                    
                    if st.session_state.get("temperature") != 0.0:
                        langchain_params["temperature"] = st.session_state.temperature
                    if st.session_state.get("max_tokens") is not None:
                        langchain_params["max_tokens"] = st.session_state.max_tokens
                    
                    rag_res = generate_smart_answer_with_langchain(**langchain_params)
                    assistant_reply = rag_res["answer"]
                    used_equipment = rag_res["used_equipment"]
                    used_files = rag_res.get("selected_files", [])
                    
                else:
                    # === 従来処理 ===
                    # 設備の決定
                    target_equipment = None
                    selection_mode = st.session_state.get("selection_mode", "manual")
                    
                    if selection_mode == "auto":
                        available_equipment = st.session_state.get("equipment_list", [])
                        target_equipment = detect_equipment_from_question(user_prompt, available_equipment)
                    else:
                        target_equipment = st.session_state.get("selected_equipment")

                    if target_equipment:
                        # RAG処理
                        rag_params = {
                            "prompt": prompt,
                            "question": user_prompt,
                            "equipment_data": st.session_state.equipment_data,
                            "target_equipment": target_equipment,
                            "model": st.session_state.claude_model,
                            "chat_history": msgs,
                        }
                        
                        if st.session_state.get("temperature") != 0.0:
                            rag_params["temperature"] = st.session_state.temperature
                        if st.session_state.get("max_tokens") is not None:
                            rag_params["max_tokens"] = st.session_state.max_tokens
                        
                        rag_res = generate_answer_with_equipment(**rag_params)
                        assistant_reply = rag_res["answer"]
                        used_equipment = rag_res["used_equipment"]
                        used_files = rag_res.get("selected_files", [])
                        
                    else:
                        # 設備なしモード
                        messages = []
                        
                        system_msg = {"role": "system", "content": prompt}
                        messages.append(system_msg)
                        
                        if len(msgs) > 1:
                            safe_history = [
                                {"role": m.get("role"), "content": m.get("content")}
                                for m in msgs[:-1]
                                if isinstance(m, dict) and m.get("role") and m.get("content")
                            ]
                            messages.extend(safe_history)
                        
                        user_msg = {
                            "role": "user",
                            "content": f"【質問】\n{user_prompt}\n\n設備資料は利用せず、あなたの知識に基づいて回答してください。"
                        }
                        messages.append(user_msg)
                        
                        max_tokens = st.session_state.get("max_tokens") or 4096
                        temperature = st.session_state.get("temperature", 0.0)
                        
                        if st.session_state.claude_model.startswith("gpt"):
                            azure_client = setup_azure_client()
                            assistant_reply = call_azure_gpt(
                                azure_client,
                                st.session_state.claude_model,
                                messages,
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                        else:
                            assistant_reply = call_claude_bedrock(
                                bedrock_client,
                                get_claude_model_name(st.session_state.claude_model),
                                messages,
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                        
                        used_equipment = "なし（一般知識による回答）"
                        used_files = []

            except Exception as e:
                logger.exception("❌ answer_gen failed — %s", e)
                st.error(f"回答生成時にエラーが発生しました: {e}")
                st.stop()

            # 画面反映
            with st.chat_message("assistant"):
                langchain_info = " (LangChain)" if st.session_state.get("use_langchain", False) else ""
                
                if used_files:
                    file_info = f"（{len(used_files)}ファイル使用）"
                    model_info = f"\n\n---\n*このレスポンスは `{st.session_state.claude_model}` と設備「{used_equipment}」{file_info}で生成されました{langchain_info}*"
                else:
                    model_info = f"\n\n---\n*このレスポンスは `{st.session_state.claude_model}` で生成されました（設備資料なし）{langchain_info}*"
                
                full_reply = assistant_reply + model_info
                st.markdown(full_reply)

            # 保存
            msg_to_save = {
                "role": "assistant",
                "content": assistant_reply,
            }
            
            if target_equipment and target_equipment != "なし（一般知識による回答）":
                msg_to_save["used_equipment"] = used_equipment
                msg_to_save["used_files"] = used_files

            msgs.append(msg_to_save)

            # ログ保存
            post_log_async(user_prompt, assistant_reply, prompt, send_to_model_comparison=True)

            # チャットタイトル生成
            try:
                new_title = generate_chat_title(msgs)
                if new_title and new_title != st.session_state.current_chat:
                    old_title = st.session_state.current_chat
                    st.session_state.chats[new_title] = st.session_state.chats[old_title]
                    del st.session_state.chats[old_title]
                    st.session_state.current_chat = new_title
                    logger.info("📝 Chat title updated: %s -> %s", old_title, new_title)
            except Exception as e:
                logger.warning("⚠️ Chat title generation failed (non-critical): %s", e)

            time.sleep(2) 
            st.rerun()

elif st.session_state["authentication_status"] is False:
    st.error('ユーザー名またはパスワードが間違っています。')
elif st.session_state["authentication_status"] is None:
    st.warning("ユーザー名とパスワードを入力してください。")
    st.stop()