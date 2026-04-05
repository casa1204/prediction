"""FastAPI 라우트 정의 — 모든 API 엔드포인트 구현.

각 엔드포인트는 DB에서 데이터를 조회하여 JSON으로 반환한다.

Requirements: 6.1~6.20
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from sqlalchemy import desc

from db.database import get_session
from db.models import (
    ElliottWaveData,
    EnsembleWeight,
    EtfData,
    HitRateRecord,
    OnchainData,
    PairData,
    PredictionRecord,
    PriceData,
    RetrainingHistory,
    SentimentData,
    TechnicalIndicator,
    WyckoffData,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/current-price")
async def get_current_price() -> dict:
    """XRP 현재 가격, 24h 변동률, 거래량."""
    session = get_session()
    try:
        latest = (
            session.query(PriceData)
            .order_by(desc(PriceData.timestamp))
            .first()
        )
        if not latest:
            return {"price": None, "change_24h": None, "volume": None}

        # 24시간 전 데이터 조회
        from datetime import timedelta

        target_time = latest.timestamp - timedelta(hours=24)
        prev = (
            session.query(PriceData)
            .filter(PriceData.timestamp <= target_time)
            .order_by(desc(PriceData.timestamp))
            .first()
        )

        change_24h = None
        if prev and prev.close and prev.close != 0:
            change_24h = round(
                ((latest.close - prev.close) / prev.close) * 100, 2
            )

        return {
            "price": latest.close,
            "change_24h": change_24h,
            "volume": latest.volume,
            "timestamp": latest.timestamp.isoformat(),
        }
    finally:
        session.close()


@router.get("/api/predictions/{timeframe}")
async def get_predictions(timeframe: str) -> dict:
    """타임프레임별 예측 결과 (개별 모델 + 앙상블)."""
    if timeframe not in ("short", "mid", "long"):
        raise HTTPException(
            status_code=400,
            detail="timeframe must be 'short', 'mid', or 'long'",
        )

    session = get_session()
    try:
        records = (
            session.query(PredictionRecord)
            .filter(PredictionRecord.timeframe == timeframe)
            .order_by(desc(PredictionRecord.timestamp))
            .limit(10)
            .all()
        )

        results = {}
        for r in records:
            if r.model_name not in results:
                results[r.model_name] = {
                    "model_name": r.model_name,
                    "predicted_price": r.predicted_price,
                    "predicted_direction": r.predicted_direction,
                    "up_probability": r.up_probability,
                    "down_probability": r.down_probability,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp.isoformat(),
                }

        return {"timeframe": timeframe, "predictions": results}
    finally:
        session.close()


@router.get("/api/technical-indicators")
async def get_technical_indicators() -> dict:
    """최신 기술 지표 데이터."""
    session = get_session()
    try:
        latest = (
            session.query(TechnicalIndicator)
            .order_by(desc(TechnicalIndicator.timestamp))
            .first()
        )
        if not latest:
            return {"indicators": None}

        return {
            "timestamp": latest.timestamp.isoformat(),
            "indicators": {
                "rsi_14": latest.rsi_14,
                "macd": latest.macd,
                "macd_signal": latest.macd_signal,
                "macd_histogram": latest.macd_histogram,
                "bb_upper": latest.bb_upper,
                "bb_middle": latest.bb_middle,
                "bb_lower": latest.bb_lower,
                "sma_5": latest.sma_5,
                "sma_10": latest.sma_10,
                "sma_20": latest.sma_20,
                "sma_50": latest.sma_50,
                "sma_200": latest.sma_200,
                "ema_12": latest.ema_12,
                "ema_26": latest.ema_26,
            },
        }
    finally:
        session.close()



@router.get("/api/price-history")
async def get_price_history() -> dict:
    """시간봉 OHLCV 가격 히스토리 (Lightweight Charts 캔들스틱용)."""
    session = get_session()
    try:
        rows = (
            session.query(PriceData)
            .order_by(PriceData.timestamp)
            .all()
        )
        candles = [
            {
                "time": int(r.timestamp.timestamp()),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
        return {"candles": candles}
    finally:
        session.close()


@router.get("/api/daily-price-history")
async def get_daily_price_history() -> dict:
    """일봉 OHLCV 가격 히스토리 (엘리엇 파동/와이코프 차트용)."""
    from db.models import DailyPriceData

    session = get_session()
    try:
        rows = (
            session.query(DailyPriceData)
            .order_by(DailyPriceData.timestamp)
            .all()
        )
        candles = [
            {
                "time": int(r.timestamp.timestamp()),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
        return {"candles": candles}
    finally:
        session.close()


@router.get("/api/elliott-wave")
async def get_elliott_wave() -> dict:
    """엘리엇 파동 분석 결과."""
    session = get_session()
    try:
        waves = (
            session.query(ElliottWaveData)
            .order_by(desc(ElliottWaveData.timestamp))
            .limit(20)
            .all()
        )
        if not waves:
            return {"waves": [], "current_position": None}

        latest = waves[0]
        wave_list = [
            {
                "wave_number": w.wave_number,
                "wave_type": w.wave_type,
                "start_price": w.start_price,
                "end_price": w.end_price,
                "start_time": w.start_time.isoformat() if w.start_time else None,
                "end_time": w.end_time.isoformat() if w.end_time else None,
                "wave_degree": w.wave_degree,
                "is_valid": w.is_valid,
                "fibonacci_targets": w.fibonacci_targets,
            }
            for w in waves
        ]

        return {
            "waves": wave_list,
            "current_position": {
                "wave_number": latest.wave_number,
                "wave_type": latest.wave_type,
                "timestamp": latest.timestamp.isoformat(),
            },
        }
    finally:
        session.close()


@router.get("/api/wyckoff")
async def get_wyckoff() -> dict:
    """와이코프 분석 결과."""
    session = get_session()
    try:
        events = (
            session.query(WyckoffData)
            .order_by(desc(WyckoffData.timestamp))
            .limit(20)
            .all()
        )
        if not events:
            return {
                "events": [],
                "market_phase": None,
                "wyckoff_phase": None,
                "confidence_score": None,
            }

        latest = events[0]
        event_list = [
            {
                "event_type": e.event_type,
                "pattern_type": e.pattern_type,
                "price": e.price,
                "volume": e.volume,
                "timestamp": e.timestamp.isoformat(),
                "is_trend_reversal": e.is_trend_reversal,
            }
            for e in events
        ]

        return {
            "events": event_list,
            "market_phase": latest.market_phase,
            "wyckoff_phase": latest.wyckoff_phase,
            "confidence_score": latest.confidence_score,
        }
    finally:
        session.close()


@router.get("/api/correlations")
async def get_correlations() -> dict:
    """상관 자산 상관계수."""
    session = get_session()
    try:
        from sqlalchemy import func

        # 각 자산별 최신 상관계수 조회
        subquery = (
            session.query(
                PairData.asset_name,
                func.max(PairData.timestamp).label("max_ts"),
            )
            .group_by(PairData.asset_name)
            .subquery()
        )

        results = (
            session.query(PairData)
            .join(
                subquery,
                (PairData.asset_name == subquery.c.asset_name)
                & (PairData.timestamp == subquery.c.max_ts),
            )
            .all()
        )

        correlations = {
            r.asset_name: {
                "price": r.price,
                "correlation_with_xrp": r.correlation_with_xrp,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in results
        }

        return {"correlations": correlations}
    finally:
        session.close()


@router.get("/api/sentiment")
async def get_sentiment() -> dict:
    """시장 심리 데이터."""
    session = get_session()
    try:
        latest = (
            session.query(SentimentData)
            .order_by(desc(SentimentData.timestamp))
            .first()
        )
        if not latest:
            return {"sentiment": None}

        return {
            "timestamp": latest.timestamp.isoformat(),
            "sentiment": {
                "google_trend_score": latest.google_trend_score,
                "sns_mention_score": latest.sns_mention_score,
                "sns_sentiment_score": latest.sns_sentiment_score,
                "fear_greed_index": latest.fear_greed_index,
                "trend_macro_score": latest.trend_macro_score,
                "trend_etf_regulatory_score": latest.trend_etf_regulatory_score,
                "trend_sentiment_score": latest.trend_sentiment_score,
                "attention_ratio": latest.attention_ratio,
                "fomo_spread": latest.fomo_spread,
                "macro_aggregate": latest.macro_aggregate,
            },
        }
    finally:
        session.close()


@router.get("/api/onchain")
async def get_onchain() -> dict:
    """온체인 데이터."""
    session = get_session()
    try:
        latest = (
            session.query(OnchainData)
            .order_by(desc(OnchainData.timestamp))
            .first()
        )
        if not latest:
            return {"onchain": None}

        return {
            "timestamp": latest.timestamp.isoformat(),
            "onchain": {
                "active_wallets": latest.active_wallets,
                "new_wallets": latest.new_wallets,
                "transaction_count": latest.transaction_count,
                "total_volume_xrp": latest.total_volume_xrp,
                "whale_tx_count": latest.whale_tx_count,
                "whale_tx_volume": latest.whale_tx_volume,
            },
        }
    finally:
        session.close()


@router.get("/api/hit-rates")
async def get_hit_rates() -> dict:
    """적중률 데이터 (모델별, 타임프레임별)."""
    session = get_session()
    try:
        records = (
            session.query(HitRateRecord)
            .order_by(desc(HitRateRecord.date))
            .limit(100)
            .all()
        )

        hit_rates = [
            {
                "date": r.date.isoformat(),
                "model_name": r.model_name,
                "timeframe": r.timeframe,
                "direction_hit_rate": r.direction_hit_rate,
                "range_hit_rate": r.range_hit_rate,
            }
            for r in records
        ]

        return {"hit_rates": hit_rates}
    finally:
        session.close()


@router.get("/api/feature-importance/{model_name}")
async def get_feature_importance(model_name: str) -> dict:
    """모델별 피처 기여도."""
    session = get_session()
    try:
        record = (
            session.query(PredictionRecord)
            .filter(
                PredictionRecord.model_name == model_name,
                PredictionRecord.feature_importance.isnot(None),
            )
            .order_by(desc(PredictionRecord.timestamp))
            .first()
        )

        if not record:
            return {
                "model_name": model_name,
                "feature_importance": None,
            }

        return {
            "model_name": model_name,
            "feature_importance": record.feature_importance,
            "timestamp": record.timestamp.isoformat(),
        }
    finally:
        session.close()


@router.get("/api/retraining-history")
async def get_retraining_history() -> list[dict]:
    """재학습 이력."""
    session = get_session()
    try:
        records = (
            session.query(RetrainingHistory)
            .order_by(desc(RetrainingHistory.timestamp))
            .limit(50)
            .all()
        )

        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "model_name": r.model_name,
                "method": r.method,
                "data_period_start": (
                    r.data_period_start.isoformat() if r.data_period_start else None
                ),
                "data_period_end": (
                    r.data_period_end.isoformat() if r.data_period_end else None
                ),
                "champion_mae": r.champion_mae,
                "champion_mape": r.champion_mape,
                "champion_direction_accuracy": r.champion_direction_accuracy,
                "challenger_mae": r.challenger_mae,
                "challenger_mape": r.challenger_mape,
                "challenger_direction_accuracy": r.challenger_direction_accuracy,
                "replaced": r.replaced,
                "error_message": r.error_message,
            }
            for r in records
        ]
    finally:
        session.close()


@router.get("/api/etf")
async def get_etf() -> dict:
    """각 ETF의 최신 데이터 반환."""
    session = get_session()
    try:
        from sqlalchemy import func

        # 각 티커별 최신 레코드 조회
        subquery = (
            session.query(
                EtfData.ticker,
                func.max(EtfData.timestamp).label("max_ts"),
            )
            .group_by(EtfData.ticker)
            .subquery()
        )

        results = (
            session.query(EtfData)
            .join(
                subquery,
                (EtfData.ticker == subquery.c.ticker)
                & (EtfData.timestamp == subquery.c.max_ts),
            )
            .all()
        )

        etfs = [
            {
                "ticker": r.ticker,
                "name": r.name,
                "price": r.price,
                "volume": r.volume,
                "premium_discount": r.premium_discount,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in results
        ]

        return {"etfs": etfs}
    finally:
        session.close()


@router.get("/api/model-weights")
async def get_model_weights() -> dict:
    """앙상블 가중치."""
    session = get_session()
    try:
        from sqlalchemy import func

        # 각 모델+타임프레임별 최신 가중치 조회
        subquery = (
            session.query(
                EnsembleWeight.model_name,
                EnsembleWeight.timeframe,
                func.max(EnsembleWeight.timestamp).label("max_ts"),
            )
            .group_by(EnsembleWeight.model_name, EnsembleWeight.timeframe)
            .subquery()
        )

        results = (
            session.query(EnsembleWeight)
            .join(
                subquery,
                (EnsembleWeight.model_name == subquery.c.model_name)
                & (EnsembleWeight.timeframe == subquery.c.timeframe)
                & (EnsembleWeight.timestamp == subquery.c.max_ts),
            )
            .all()
        )

        weights = {}
        for r in results:
            if r.timeframe not in weights:
                weights[r.timeframe] = {}
            weights[r.timeframe][r.model_name] = {
                "weight": r.weight,
                "timestamp": r.timestamp.isoformat(),
            }

        return {"weights": weights}
    finally:
        session.close()


@router.get("/api/ai-analysis")
async def get_ai_analysis() -> dict:
    """최신 AI 시장 분석 리포트 (DB에서 조회, 없으면 실시간 생성)."""
    from db.models import AiAnalysisRecord

    session = get_session()
    try:
        latest = (
            session.query(AiAnalysisRecord)
            .order_by(desc(AiAnalysisRecord.timestamp))
            .first()
        )
        if latest:
            return {
                "analysis": latest.analysis,
                "timestamp": latest.timestamp.isoformat(),
                "model_used": latest.model_used,
            }
    finally:
        session.close()

    # DB에 없으면 실시간 생성
    from ai.analyzer import generate_and_save_analysis
    analysis = await generate_and_save_analysis()
    return {"analysis": analysis, "timestamp": None, "model_used": "gemini-2.5-flash"}


@router.get("/api/ai-analysis/history")
async def get_ai_analysis_history() -> list[dict]:
    """AI 분석 리포트 이력 (최근 30건)."""
    from db.models import AiAnalysisRecord

    session = get_session()
    try:
        records = (
            session.query(AiAnalysisRecord)
            .order_by(desc(AiAnalysisRecord.timestamp))
            .limit(30)
            .all()
        )
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "analysis": r.analysis,
                "model_used": r.model_used,
            }
            for r in records
        ]
    finally:
        session.close()
