# logging_utils.py
import logging, os, tempfile, streamlit as st
from logging.handlers import RotatingFileHandler

def init_logger() -> logging.Logger:
    logger = logging.getLogger("app")
    if logger.handlers:          # 再実行時の重複防止
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            "%Y-%m-%d %H:%M:%S")

    # ── ① Cloud Logs 用 (標準出力) ───────────────────────────
    ch = logging.StreamHandler()     # これが Cloud の Logs ビューに流れる
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── ② 書ける環境なら一時ファイルにも残す（任意） ────────
    try:
        tmp_path = os.path.join(tempfile.gettempdir(), "app.log")
        fh = RotatingFileHandler(tmp_path, maxBytes=300_000, backupCount=2,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except (PermissionError, OSError):
        # ファイルに書けない環境でも動作を止めない
        pass

    return logger