import logging, streamlit as st, os
from logging.handlers import RotatingFileHandler

def init_logger() -> logging.Logger:
    logger = logging.getLogger("app")
    if logger.handlers:                       # 再実行時の重複防止
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            "%Y-%m-%d %H:%M:%S")

    # A. Streamlit-handler（最新500行だけ session_state に保持）
    class _StHandler(logging.Handler):
        def emit(self, record):
            logs = st.session_state.setdefault("__logs", [])
            logs.append(self.format(record))
            if len(logs) > 500:
                logs.pop(0)
    sh = _StHandler(); sh.setFormatter(fmt); logger.addHandler(sh)

    # B. /tmp に一時ファイル（デバッグ用・揮発OK）
    tmp_dir = pathlib.Path(__file__).resolve().parent.parent / "logs"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / "app.log"
    os.makedirs("/tmp", exist_ok=True)
    fh = RotatingFileHandler(tmp_path, maxBytes=300_000, backupCount=2,
                             encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)

    return logger