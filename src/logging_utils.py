# logging_utils.py
import logging, os, tempfile, streamlit as st
from logging.handlers import RotatingFileHandler

def init_logger() -> logging.Logger:
    """
    - Streamlit UI に流すハンドラ (_StHandler)
    - 書き込み可能なら一時ファイル (/tmp など) へローテーションログ
    """
    logger = logging.getLogger("app")
    if logger.handlers:            # 再実行時の重複防止
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # ── A. Streamlit-handler（最新500行を session_state にキープ） ──
    class _StHandler(logging.Handler):
        def emit(self, record):
            logs = st.session_state.setdefault("__logs", [])
            logs.append(self.format(record))
            if len(logs) > 500:       # 上限 500 行
                logs.pop(0)

    sh = _StHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # ── B. ローカル一時ファイル（書ける環境だけ） ──
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), "app.log")
        fh = RotatingFileHandler(
            tmp_path, maxBytes=300_000, backupCount=2, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except (PermissionError, OSError) as e:
        # ファイル書き込み不可でも UI ログは生きる
        print("⚠️  File log disabled:", e)

    return logger