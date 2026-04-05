"""ORM 모델 정의 — 설계 문서의 데이터 모델 섹션 기반."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PriceData(Base):
    """XRP OHLCV 시간봉 가격 데이터."""

    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class DailyPriceData(Base):
    """XRP OHLCV 일봉 가격 데이터 (엘리엇 파동/와이코프 분석용)."""

    __tablename__ = "daily_price_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class TechnicalIndicator(Base):
    """계산된 기술 지표."""

    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    sma_5 = Column(Float)
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    obv = Column(Float)
    sma_100 = Column(Float)
    disparity_50 = Column(Float)
    disparity_100 = Column(Float)
    disparity_200 = Column(Float)


class ElliottWaveData(Base):
    """엘리엇 파동 분석 결과."""

    __tablename__ = "elliott_wave_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    wave_number = Column(String, nullable=False)  # "1","2","3","4","5","A","B","C"
    wave_type = Column(String, nullable=False)  # "impulse" or "corrective"
    start_price = Column(Float, nullable=False)
    end_price = Column(Float)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    wave_degree = Column(String)  # "Primary","Intermediate" 등
    is_valid = Column(Boolean, default=True)
    fibonacci_targets = Column(JSON)  # {"0.236": x, "0.382": y, ...}


class WyckoffData(Base):
    """와이코프 패턴 분석 결과."""

    __tablename__ = "wyckoff_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(String, nullable=False)  # "PS","SC","AR","ST","Spring" 등
    pattern_type = Column(String, nullable=False)  # "accumulation" or "distribution"
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    market_phase = Column(String)  # "Accumulation","Markup" 등
    wyckoff_phase = Column(String)  # "A","B","C","D","E"
    confidence_score = Column(Float)  # 0~100
    is_trend_reversal = Column(Boolean, default=False)


class PairData(Base):
    """상관 자산 가격 데이터."""

    __tablename__ = "pair_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    # "XRP/BTC","XRP/ETH","XRP/USD","BTC/USD","ETH/USD","S&P500","NASDAQ"
    asset_name = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    correlation_with_xrp = Column(Float)  # 상관계수


class SentimentData(Base):
    """시장 심리 데이터."""

    __tablename__ = "sentiment_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    google_trend_score = Column(Float)  # 0~100 정규화 (기본 XRP 키워드)
    sns_mention_score = Column(Float)  # 0~100 정규화
    sns_sentiment_score = Column(Float)  # 0~100 정규화
    fear_greed_index = Column(Float)  # 0~100

    # ── 확장 Google Trends 키워드 그룹 점수 ──
    # 그룹1: 거시경제 ("Fed Interest Rate", "Inflation", "CPI", "Recession", "Rate Cut")
    trend_macro_score = Column(Float)
    # 그룹2: ETF/규제 ("XRP ETF", "Spot ETF", "RLUSD", "SEC", "Crypto Regulation")
    trend_etf_regulatory_score = Column(Float)
    # 그룹3: 심리 ("Buy Crypto", "Sell Crypto", "Bitcoin", "Crypto Scam")
    trend_sentiment_score = Column(Float)

    # ── 파생 피처 ──
    # Attention Ratio: XRP trend / Bitcoin trend
    attention_ratio = Column(Float)
    # FOMO Spread: "Buy Crypto" - "Sell Crypto"
    fomo_spread = Column(Float)
    # 거시경제 종합 점수 (macro 키워드 가중 평균)
    macro_aggregate = Column(Float)


class OnchainData(Base):
    """온체인 데이터."""

    __tablename__ = "onchain_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    active_wallets = Column(Integer)
    new_wallets = Column(Integer)
    transaction_count = Column(Integer)
    total_volume_xrp = Column(Float)
    whale_tx_count = Column(Integer)  # 100만 XRP 이상 거래 건수
    whale_tx_volume = Column(Float)  # 고래 거래 총량


class PredictionRecord(Base):
    """예측 결과 기록."""

    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_name = Column(String, nullable=False)  # "lstm","xgboost","rf","transformer","ensemble"
    timeframe = Column(String, nullable=False)  # "short","mid","long"
    predicted_price = Column(Float, nullable=False)
    predicted_direction = Column(String, nullable=False)  # "up" or "down"
    up_probability = Column(Float)  # 0~100
    down_probability = Column(Float)  # 0~100
    confidence = Column(Float)  # 0~100
    actual_price = Column(Float)  # 실제 가격 (나중에 업데이트)
    feature_importance = Column(JSON)


class HitRateRecord(Base):
    """적중률 기록."""

    __tablename__ = "hit_rate_records"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    model_name = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    direction_hit_rate = Column(Float)  # 0~100
    range_hit_rate = Column(Float)  # 0~100


class EnsembleWeight(Base):
    """앙상블 가중치 이력."""

    __tablename__ = "ensemble_weights"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_name = Column(String, nullable=False)
    weight = Column(Float, nullable=False)
    timeframe = Column(String, nullable=False)


class RetrainingHistory(Base):
    """재학습 이력."""

    __tablename__ = "retraining_history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    model_name = Column(String, nullable=False)
    method = Column(String, nullable=False)  # "incremental" or "full"
    data_period_start = Column(DateTime)
    data_period_end = Column(DateTime)
    champion_mae = Column(Float)
    champion_mape = Column(Float)
    champion_direction_accuracy = Column(Float)
    challenger_mae = Column(Float)
    challenger_mape = Column(Float)
    challenger_direction_accuracy = Column(Float)
    replaced = Column(Boolean, nullable=False)
    error_message = Column(String)


class EtfData(Base):
    """XRP 현물 ETF 데이터."""

    __tablename__ = "etf_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ticker = Column(String, nullable=False)  # XRPR, XRPZ, XRP, XXRP, XRPI, XRPT
    name = Column(String)
    price = Column(Float)
    volume = Column(Integer)
    premium_discount = Column(Float)  # XRP 현물 대비 프리미엄/디스카운트 %


class CollectionLog(Base):
    """데이터 수집 로그."""

    __tablename__ = "collection_logs"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)  # "success" or "failure"
    error_message = Column(String)
    consecutive_failures = Column(Integer, default=0)


class AiAnalysisRecord(Base):
    """AI 시장 분석 리포트 저장."""

    __tablename__ = "ai_analysis_records"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    analysis = Column(String, nullable=False)  # 마크다운 분석 텍스트
    model_used = Column(String)  # "gemini-2.5-flash" 등
