"""APScheduler 기반 작업 스케줄러 — 데이터 수집 및 재학습 자동화.

데이터 수집 스케줄:
- 기술 지표/페어: 1시간 간격
- 심리/온체인: 1일 간격

재학습 트리거:
- 일별 수집 완료 후 자동 시작

Requirements: 7.1, 7.2, 7.3, 7.5, 7.12
"""

import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config import (
    MODEL_NAMES,
    SCHEDULE_INTERVAL_ETF,
    SCHEDULE_INTERVAL_ONCHAIN,
    SCHEDULE_INTERVAL_PAIR,
    SCHEDULE_INTERVAL_SENTIMENT,
    SCHEDULE_INTERVAL_TECHNICAL,
    TIMEFRAMES,
)

logger = logging.getLogger(__name__)


async def collect_technical() -> None:
    """기술 지표 + 엘리엇 파동 + 와이코프 분석 수집 작업."""
    from collectors.price_collector import PriceCollector
    from analysis.technical_indicators import TechnicalIndicatorModule
    from analysis.elliott_wave import ElliottWaveModule
    from analysis.wyckoff import WyckoffModule
    from db.database import get_session
    from db.models import PriceData

    logger.info("[scheduler] 기술 지표 수집 시작")
    try:
        collector = PriceCollector()
        await collector.collect_with_retry()

        session = get_session()
        try:
            import pandas as pd

            # 일봉 데이터로 기술 지표 계산 (1000일치 → 학습 데이터와 일치)
            from db.models import DailyPriceData
            daily_rows = session.query(DailyPriceData).order_by(DailyPriceData.timestamp).all()
            if daily_rows:
                daily_df = pd.DataFrame([
                    {"timestamp": r.timestamp, "open": r.open, "high": r.high,
                     "low": r.low, "close": r.close, "volume": r.volume}
                    for r in daily_rows
                ])

                # 기술 지표 계산 (일봉 기준)
                ti_module = TechnicalIndicatorModule()
                result = ti_module.calculate_all(daily_df)
                ti_module.save_to_db(result, session)
                logger.info("[scheduler] 일봉 기준 기술 지표 %d건 계산", len(result))

                # 엘리엇 파동 분석 (일봉 200개 이상이면 실행)
                if len(daily_rows) >= 200:
                    try:
                        # 기존 분석 결과 삭제 (중복 방지)
                        from db.models import ElliottWaveData, WyckoffData
                        session.query(ElliottWaveData).delete()
                        session.query(WyckoffData).delete()
                        session.commit()

                        ew_module = ElliottWaveModule()
                        waves = ew_module.detect_waves(daily_df)
                        if waves:
                            ew_module.save_to_db(waves, session)
                            logger.info("[scheduler] 엘리엇 파동 %d개 감지 (일봉 %d개)", len(waves), len(daily_df))
                    except Exception:
                        logger.exception("[scheduler] 엘리엇 파동 분석 실패")

                    # 와이코프 분석 (일봉 기준)
                    try:
                        wk_module = WyckoffModule()
                        accum = wk_module.detect_accumulation(daily_df)
                        dist = wk_module.detect_distribution(daily_df)
                        phase = wk_module.determine_market_phase(daily_df)
                        wp = wk_module.determine_wyckoff_phase(daily_df)
                        all_events = accum + dist
                        if all_events:
                            wk_module.save_to_db(all_events, session, phase, wp)
                            logger.info("[scheduler] 와이코프 이벤트 %d개 감지 (phase=%s)", len(all_events), phase.phase)
                    except Exception:
                        logger.exception("[scheduler] 와이코프 분석 실패")
                else:
                    daily_count = len(daily_rows)
                    logger.info("[scheduler] 일봉 %d개 — 엘리엇/와이코프 분석에 200개 이상 필요", daily_count)
            else:
                logger.warning("[scheduler] 일봉 데이터 없음 — 기술 지표 계산 건너뜀")
        finally:
            session.close()

        logger.info("[scheduler] 기술 지표 수집 완료")
    except Exception:
        logger.exception("[scheduler] 기술 지표 수집 실패")


async def collect_pair() -> None:
    """페어 데이터 수집 + 상관계수 계산."""
    from collectors.pair_collector import PairCollector, PairDataModule
    from db.database import get_session
    from db.models import PairData, PriceData
    import pandas as pd

    logger.info("[scheduler] 페어 데이터 수집 시작")
    try:
        collector = PairCollector()
        await collector.collect_with_retry()

        # 상관계수 계산
        session = get_session()
        try:
            # XRP 일별 종가
            xrp_rows = session.query(PriceData).order_by(PriceData.timestamp).all()
            if not xrp_rows:
                return
            xrp_df = pd.DataFrame([{"date": r.timestamp.date(), "close": r.close} for r in xrp_rows])
            xrp_df = xrp_df.groupby("date", as_index=False).last()

            module = PairDataModule()
            from config import CRYPTO_PAIRS, INDICES
            all_assets = CRYPTO_PAIRS + INDICES

            for asset in all_assets:
                asset_rows = session.query(PairData).filter(PairData.asset_name == asset).order_by(PairData.timestamp).all()
                if not asset_rows:
                    continue
                pair_df = pd.DataFrame([{"date": r.timestamp.date(), "close": r.price} for r in asset_rows])
                pair_df = pair_df.groupby("date", as_index=False).last()

                # 최근 90일만 사용
                xrp_df_90 = xrp_df.tail(90)
                pair_df_90 = pair_df.tail(90)
                corr = module.calculate_correlation(xrp_df_90, pair_df_90)

                # 최신 레코드의 상관계수 업데이트
                latest = session.query(PairData).filter(PairData.asset_name == asset).order_by(PairData.timestamp.desc()).first()
                if latest:
                    latest.correlation_with_xrp = corr

            session.commit()
            logger.info("[scheduler] 상관계수 계산 완료")
        finally:
            session.close()

        logger.info("[scheduler] 페어 데이터 수집 완료")
    except Exception:
        logger.exception("[scheduler] 페어 데이터 수집 실패")


async def collect_sentiment() -> None:
    """심리 데이터 수집 작업 (일별)."""
    from collectors.sentiment_collector import SentimentCollector

    logger.info("[scheduler] 심리 데이터 수집 시작")
    try:
        collector = SentimentCollector()
        await collector.collect_with_retry()
        logger.info("[scheduler] 심리 데이터 수집 완료")
    except Exception:
        logger.exception("[scheduler] 심리 데이터 수집 실패")


async def collect_etf() -> None:
    """ETF 데이터 수집 작업 (일별)."""
    from collectors.etf_collector import EtfCollector

    logger.info("[scheduler] ETF 데이터 수집 시작")
    try:
        collector = EtfCollector()
        await collector.collect_with_retry()
        logger.info("[scheduler] ETF 데이터 수집 완료")
    except Exception:
        logger.exception("[scheduler] ETF 데이터 수집 실패")


async def collect_onchain() -> None:
    """온체인 데이터 수집 작업 (일별)."""
    from collectors.onchain_collector import OnchainCollector

    logger.info("[scheduler] 온체인 데이터 수집 시작")
    try:
        collector = OnchainCollector()
        await collector.collect_with_retry()
        logger.info("[scheduler] 온체인 데이터 수집 완료")
    except Exception:
        logger.exception("[scheduler] 온체인 데이터 수집 실패")


def _build_model_instance(model_name: str):
    """모델 이름으로 BaseModel 인스턴스를 생성한다."""
    if model_name == "lstm":
        from prediction.lstm_model import LSTMModel
        return LSTMModel(epochs=50)
    elif model_name == "xgboost":
        from prediction.xgboost_model import XGBoostModel
        return XGBoostModel()
    elif model_name == "rf":
        from prediction.random_forest_model import RandomForestModel
        return RandomForestModel()
    elif model_name == "transformer":
        from prediction.transformer_model import TransformerModel
        return TransformerModel(epochs=50)
    elif model_name == "lgbm":
        from prediction.lightgbm_model import LightGBMModel
        return LightGBMModel()
    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")


def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """모델 평가 함수 — RetrainingScheduler.run_retraining에 전달."""
    from monitoring.retraining_scheduler import ModelMetrics
    from monitoring.hit_rate_tracker import HitRateTracker

    predictions = []
    for i in range(len(X_test)):
        row = X_test[i : i + 1]
        try:
            result = model.predict(row, "short")
            predictions.append(result.predicted_price)
        except Exception:
            predictions.append(y_test[i])

    actuals = y_test.tolist()
    mae = HitRateTracker.calculate_mae(predictions, actuals)
    mape = HitRateTracker.calculate_mape(predictions, actuals)

    pred_dirs = [
        "up" if predictions[i] > actuals[i - 1] else "down"
        for i in range(1, len(predictions))
    ]
    actual_dirs = [
        "up" if actuals[i] > actuals[i - 1] else "down"
        for i in range(1, len(actuals))
    ]
    dir_acc = HitRateTracker.calculate_direction_hit_rate(pred_dirs, actual_dirs)

    return ModelMetrics(mae=mae, mape=mape, direction_accuracy=dir_acc)


async def _run_retraining() -> None:
    """재학습을 별도 스레드에서 실행하여 API 응답을 블로킹하지 않는다."""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_retraining_sync)

    # 재학습 완료 후 AI 분석 자동 생성
    try:
        from ai.analyzer import generate_and_save_analysis
        await generate_and_save_analysis()
        logger.info("[scheduler] AI 분석 리포트 자동 생성 완료")
    except Exception:
        logger.exception("[scheduler] AI 분석 리포트 생성 실패")


def _run_retraining_sync() -> None:
    """모델 재학습 실행 (동기) — 타임프레임별로 별도 모델을 학습하고 예측한다.

    각 타임프레임(short/mid/long)마다:
    1. FeatureEngineering으로 해당 타임프레임 타겟의 데이터셋 구성
    2. 각 모델(LSTM, XGBoost, RF, Transformer) 학습
    3. 해당 타임프레임 전용 앙상블로 예측
    4. PredictionRecord에 저장
    """
    logger.info("[scheduler] 모델 재학습 시작")

    from db.database import get_session
    from db.models import PredictionRecord, EnsembleWeight
    from prediction.feature_engineering import FeatureEngineering
    from prediction.ensemble import EnsembleEngine
    from monitoring.retraining_scheduler import RetrainingScheduler

    session = get_session()
    try:
        fe = FeatureEngineering()
        now = datetime.now(timezone.utc)
        retraining_scheduler = RetrainingScheduler(session=session)

        for timeframe in TIMEFRAMES:
            logger.info("[scheduler] === 타임프레임 '%s' 학습 시작 ===", timeframe)

            # 1) 타임프레임별 데이터셋 구성
            X, y, feature_names, insufficient_data = fe.build_dataset(
                session, timeframe=timeframe
            )

            if X.size == 0 or y.size == 0:
                logger.warning("[scheduler] tf='%s': 학습 데이터 없음, 건너뜀", timeframe)
                continue

            # 학습/테스트 분할 (80/20)
            split_idx = max(1, int(len(X) * 0.8))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            if len(X_test) == 0:
                logger.warning("[scheduler] tf='%s': 테스트 데이터 없음, 건너뜀", timeframe)
                continue

            # 2) 이 타임프레임 전용 모델들 학습
            tf_models: dict = {}
            tf_accuracy: dict[str, float] = {}

            for model_name in MODEL_NAMES:
                try:
                    model = _build_model_instance(model_name)
                    model.feature_names = list(feature_names)

                    min_samples = getattr(model, "seq_length", 0) + 1
                    if len(X_train) <= min_samples:
                        logger.warning(
                            "[scheduler] '%s' (tf=%s): 데이터 부족(%d<=%d), 건너뜀",
                            model_name, timeframe, len(X_train), min_samples,
                        )
                        continue

                    record = retraining_scheduler.run_retraining(
                        model_name=f"{model_name}_{timeframe}",
                        champion=model,
                        method="full",
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        data_period_start=now,
                        data_period_end=now,
                        evaluate_fn=_evaluate_model,
                    )

                    if record.error_message:
                        logger.error("[scheduler] '%s' (tf=%s) 오류: %s", model_name, timeframe, record.error_message)
                        continue

                    tf_models[model_name] = model
                    if record.challenger_metrics:
                        tf_accuracy[model_name] = record.challenger_metrics.direction_accuracy

                    logger.info("[scheduler] '%s' (tf=%s) 학습 완료", model_name, timeframe)

                except Exception as e:
                    logger.error("[scheduler] '%s' (tf=%s) 예외: %s", model_name, timeframe, e)

            if not tf_models:
                logger.warning("[scheduler] tf='%s': 학습된 모델 없음", timeframe)
                continue

            # 3) 이 타임프레임 전용 앙상블 구성 및 예측
            ensemble = EnsembleEngine(tf_models)
            if tf_accuracy:
                ensemble.update_weights(tf_accuracy)

            # 가중치 저장
            for m_name, weight in ensemble.weights.items():
                session.add(EnsembleWeight(
                    timestamp=now.replace(tzinfo=None),
                    model_name=m_name,
                    weight=weight,
                    timeframe=timeframe,
                ))

            # 4) 예측 수행 (전체 피처 데이터 사용)
            try:
                ensemble_result = ensemble.predict(X, timeframe)

                for m_name, pred in ensemble_result.individual_results.items():
                    session.add(PredictionRecord(
                        timestamp=now.replace(tzinfo=None),
                        model_name=m_name,
                        timeframe=timeframe,
                        predicted_price=pred.predicted_price,
                        predicted_direction="up" if pred.up_probability > pred.down_probability else "down",
                        up_probability=pred.up_probability,
                        down_probability=pred.down_probability,
                        confidence=pred.confidence,
                        feature_importance=pred.feature_importance,
                    ))

                session.add(PredictionRecord(
                    timestamp=now.replace(tzinfo=None),
                    model_name="ensemble",
                    timeframe=timeframe,
                    predicted_price=ensemble_result.final_price,
                    predicted_direction=ensemble_result.final_direction,
                    up_probability=ensemble_result.integrated_up_probability,
                    down_probability=ensemble_result.integrated_down_probability,
                    confidence=0.0,
                    feature_importance=None,
                ))

                logger.info(
                    "[scheduler] tf='%s' 예측 완료: ensemble=$%.4f %s",
                    timeframe, ensemble_result.final_price, ensemble_result.final_direction,
                )

            except Exception as e:
                logger.error("[scheduler] tf='%s' 예측 실패: %s", timeframe, e)

        session.commit()
        logger.info("[scheduler] 모델 재학습 및 예측 완료")

    except Exception as e:
        logger.exception("[scheduler] 재학습 파이프라인 전체 오류: %s", e)
        session.rollback()
    finally:
        session.close()



def create_scheduler() -> AsyncIOScheduler:
    """APScheduler 인스턴스를 생성하고 작업을 등록한다.

    서버 시작 시 모든 수집 작업을 즉시 1회 실행한 뒤,
    이후 정해진 주기로 반복 실행한다.

    Returns:
        설정된 AsyncIOScheduler 인스턴스.
    """
    scheduler = AsyncIOScheduler(job_defaults={"misfire_grace_time": 60})
    now = datetime.now(timezone.utc)

    # 기술 지표 수집: 즉시 1회 + 1시간 간격
    scheduler.add_job(
        collect_technical,
        trigger=IntervalTrigger(seconds=SCHEDULE_INTERVAL_TECHNICAL),
        id="collect_technical",
        name="기술 지표 수집",
        replace_existing=True,
        next_run_time=now,  # 즉시 실행
    )

    # 페어 데이터 수집: 즉시 1회 + 1시간 간격
    scheduler.add_job(
        collect_pair,
        trigger=IntervalTrigger(seconds=SCHEDULE_INTERVAL_PAIR),
        id="collect_pair",
        name="페어 데이터 수집",
        replace_existing=True,
        next_run_time=now,  # 즉시 실행
    )

    # 심리 데이터 수집: 즉시 1회 + 1일 간격
    scheduler.add_job(
        collect_sentiment,
        trigger=IntervalTrigger(seconds=SCHEDULE_INTERVAL_SENTIMENT),
        id="collect_sentiment",
        name="심리 데이터 수집",
        replace_existing=True,
        next_run_time=now,  # 즉시 실행
    )

    # 온체인 데이터 수집: 즉시 1회 + 1일 간격
    scheduler.add_job(
        collect_onchain,
        trigger=IntervalTrigger(seconds=SCHEDULE_INTERVAL_ONCHAIN),
        id="collect_onchain",
        name="온체인 데이터 수집",
        replace_existing=True,
        next_run_time=now,  # 즉시 실행
    )

    # ETF 데이터 수집: 즉시 1회 + 1일 간격
    scheduler.add_job(
        collect_etf,
        trigger=IntervalTrigger(seconds=SCHEDULE_INTERVAL_ETF),
        id="collect_etf",
        name="ETF 데이터 수집",
        replace_existing=True,
        next_run_time=now,  # 즉시 실행
    )

    # 재학습: 매일 UTC 21:00 (한국시간 오전 6:00) + 서버 시작 5분 후 1회
    from datetime import timedelta
    first_retraining = now + timedelta(minutes=5)
    scheduler.add_job(
        _run_retraining,
        trigger=CronTrigger(hour=21, minute=0, timezone="UTC"),
        id="daily_retraining",
        name="일별 재학습 (KST 06:00)",
        replace_existing=True,
        next_run_time=first_retraining,  # 서버 시작 5분 후 (수집 완료 대기)
    )

    logger.info(
        "[scheduler] 스케줄러 생성 완료 — "
        "기술지표/페어: %ds, 심리: %ds, 온체인: %ds, ETF: %ds, "
        "재학습: 매일 KST 06:00",
        SCHEDULE_INTERVAL_TECHNICAL,
        SCHEDULE_INTERVAL_PAIR,
        SCHEDULE_INTERVAL_SENTIMENT,
        SCHEDULE_INTERVAL_ONCHAIN,
        SCHEDULE_INTERVAL_ETF,
    )

    return scheduler
