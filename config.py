"""전역 설정 - API 키, DB 경로, 스케줄 주기 등."""

import os
from pathlib import Path

# .env 파일 로드 (python-dotenv 없이 직접 파싱)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# ── 데이터베이스 ──────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db/xrp_data.db")

# ── 타임존 ────────────────────────────────────────────────────
TIMEZONE = "America/New_York"

# ── 바이낸스 API (키 불필요, 무료) ────────────────────────────
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")

CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")  # 무료 키 (cryptocompare.com에서 발급)
LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")  # LunarCrush API 키
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Google Gemini API 키

# Bithomp 제거 — XRPL 공식 JSON-RPC API 사용 (키 불필요)
XRPL_RPC_URL = os.getenv("XRPL_RPC_URL", "https://s1.ripple.com:51234/")

# ── 수집 스케줄 주기 (초 단위) ────────────────────────────────
SCHEDULE_INTERVAL_TECHNICAL = 3600      # 기술 지표: 1시간
SCHEDULE_INTERVAL_PAIR = 3600           # 페어 데이터: 1시간
SCHEDULE_INTERVAL_SENTIMENT = 86400     # 심리 데이터: 1일
SCHEDULE_INTERVAL_ONCHAIN = 86400       # 온체인 데이터: 1일

# ── 재학습 주기 ───────────────────────────────────────────────
RETRAINING_INTERVAL_DAYS = 1            # 기본 일별 재학습

# ── 수집기 재시도 설정 ────────────────────────────────────────
MAX_RETRIES = 3
CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 3

# ── 상관 자산 페어 ────────────────────────────────────────────
CRYPTO_PAIRS = ["XRP/BTC", "XRP/ETH", "XRP/USD", "BTC/USD", "ETH/USD"]
INDICES = ["S&P500", "NASDAQ", "DXY", "VIX"]

# ── XRP ETF 티커 목록 ─────────────────────────────────────────
XRP_ETF_TICKERS = {
    "XRP": "Bitwise XRP ETF",
    "XRPZ": "Franklin XRP ETF",
    "XRPR": "REX-Osprey XRP ETF",
    "XXRP": "Teucrium 2x Long XRP ETF",
    "XRPI": "Volatility Shares XRP ETF",
    "XRPT": "Volatility Shares XRP 2X ETF",
}

# 2x 레버리지 ETF 목록 (가격/2로 프리미엄 비교)
XRP_ETF_LEVERAGED = {"XXRP", "XRPT"}

SCHEDULE_INTERVAL_ETF = 86400  # ETF 데이터: 1일

# ── 온체인 설정 ───────────────────────────────────────────────
WHALE_THRESHOLD_XRP = 100_000           # 고래 거래 기준: 10만 XRP

# ── 예측 타임프레임 ───────────────────────────────────────────
TIMEFRAMES = ["short", "mid", "long"]

# ── ML 모델 목록 ──────────────────────────────────────────────
MODEL_NAMES = ["lstm", "xgboost", "rf", "transformer", "lgbm"]

# ── 모델 저장 경로 ────────────────────────────────────────────
CHAMPION_MODEL_DIR = "models/champion"
CHALLENGER_MODEL_DIR = "models/challenger"

# ── 적중률 허용 오차 ──────────────────────────────────────────
HIT_RATE_TOLERANCE = {"short": 0.03, "mid": 0.05, "long": 0.10}
