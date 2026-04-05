"""통합 테스트 — API 엔드포인트 → DB 조회 → JSON 응답 형식 검증, 수집 → 저장 → 조회 흐름 검증.

Requirements: 6.16, 8.4
"""

import os

os.environ["DATABASE_URL"] = "sqlite:///tests/_test_integration.db"

from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router
from db.database import engine, get_session, init_db
from db.models import (
    Base,
    EnsembleWeight,
    HitRateRecord,
    OnchainData,
    PairData,
    PredictionRecord,
    PriceData,
    RetrainingHistory,
    SentimentData,
    TechnicalIndicator,
)

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def _reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


class TestCollectStoreQueryFlow:
    """수집 → 저장 → API 조회 흐름 검증."""

    @classmethod
    def setup_class(cls):
        _reset_db()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cls.now = now
        session = get_session()

        # 가격 데이터 시드 (여러 시점)
        for i in range(5):
            ts = now - timedelta(hours=25 - i)
            session.add(PriceData(
                timestamp=ts,
                open=0.50 + i * 0.01,
                high=0.52 + i * 0.01,
                low=0.49 + i * 0.01,
                close=0.51 + i * 0.01,
                volume=1e6 + i * 1e5,
            ))

        # 기술 지표
        session.add(TechnicalIndicator(
            timestamp=now,
            rsi_14=60.0, macd=0.02, macd_signal=0.01, macd_histogram=0.01,
            bb_upper=0.60, bb_middle=0.55, bb_lower=0.50,
            sma_5=0.54, sma_10=0.53, sma_20=0.52, sma_50=0.51, sma_200=0.49,
            ema_12=0.54, ema_26=0.53,
        ))

        # 심리 데이터
        session.add(SentimentData(
            timestamp=now,
            google_trend_score=72.0,
            sns_mention_score=65.0,
            sns_sentiment_score=58.0,
            fear_greed_index=55.0,
        ))

        # 온체인 데이터
        session.add(OnchainData(
            timestamp=now,
            active_wallets=200000,
            new_wallets=8000,
            transaction_count=1000000,
            total_volume_xrp=8e8,
            whale_tx_count=30,
            whale_tx_volume=6e7,
        ))

        # 페어 데이터
        session.add(PairData(
            timestamp=now, asset_name="XRP/BTC",
            price=1.6e-5, correlation_with_xrp=0.88,
        ))

        # 예측 레코드
        for m in ["lstm", "xgboost", "rf", "transformer", "ensemble"]:
            session.add(PredictionRecord(
                timestamp=now, model_name=m, timeframe="short",
                predicted_price=0.57, predicted_direction="up",
                up_probability=68.0, down_probability=32.0,
                confidence=72.0, feature_importance={"rsi_14": 0.25},
            ))

        # 적중률
        session.add(HitRateRecord(
            date=now, model_name="ensemble", timeframe="short",
            direction_hit_rate=75.0, range_hit_rate=58.0,
        ))

        # 앙상블 가중치
        for m in ["lstm", "xgboost", "rf", "transformer"]:
            session.add(EnsembleWeight(
                timestamp=now, model_name=m, weight=0.25, timeframe="short",
            ))

        # 재학습 이력
        session.add(RetrainingHistory(
            timestamp=now, model_name="xgboost", method="full",
            data_period_start=now - timedelta(days=90), data_period_end=now,
            champion_mae=0.025, champion_mape=4.0, champion_direction_accuracy=68.0,
            challenger_mae=0.020, challenger_mape=3.5, challenger_direction_accuracy=72.0,
            replaced=True,
        ))

        session.commit()
        session.close()

    # ── API 응답 형식 검증 ────────────────────────────────────

    def test_current_price_json_format(self):
        resp = client.get("/api/current-price")
        assert resp.status_code == 200
        data = resp.json()
        assert "price" in data
        assert "change_24h" in data
        assert "volume" in data
        assert "timestamp" in data
        assert isinstance(data["price"], (int, float))

    def test_predictions_json_format(self):
        resp = client.get("/api/predictions/short")
        assert resp.status_code == 200
        data = resp.json()
        assert data["timeframe"] == "short"
        assert "predictions" in data
        assert isinstance(data["predictions"], dict)
        # 개별 모델 + 앙상블 모두 존재
        assert "ensemble" in data["predictions"]
        pred = data["predictions"]["ensemble"]
        assert "predicted_price" in pred
        assert "predicted_direction" in pred
        assert "up_probability" in pred

    def test_technical_indicators_json_format(self):
        resp = client.get("/api/technical-indicators")
        assert resp.status_code == 200
        data = resp.json()
        assert "indicators" in data
        ind = data["indicators"]
        assert ind["rsi_14"] == 60.0
        assert "macd" in ind
        assert "bb_upper" in ind

    def test_sentiment_json_format(self):
        resp = client.get("/api/sentiment")
        assert resp.status_code == 200
        data = resp.json()
        assert "sentiment" in data
        s = data["sentiment"]
        assert s["google_trend_score"] == 72.0
        assert s["fear_greed_index"] == 55.0

    def test_onchain_json_format(self):
        resp = client.get("/api/onchain")
        assert resp.status_code == 200
        data = resp.json()
        assert "onchain" in data
        o = data["onchain"]
        assert o["active_wallets"] == 200000
        assert o["whale_tx_count"] == 30

    def test_correlations_json_format(self):
        resp = client.get("/api/correlations")
        assert resp.status_code == 200
        data = resp.json()
        assert "correlations" in data
        assert "XRP/BTC" in data["correlations"]

    def test_hit_rates_json_format(self):
        resp = client.get("/api/hit-rates")
        assert resp.status_code == 200
        data = resp.json()
        assert "hit_rates" in data
        assert len(data["hit_rates"]) >= 1
        hr = data["hit_rates"][0]
        assert "model_name" in hr
        assert "direction_hit_rate" in hr

    def test_model_weights_json_format(self):
        resp = client.get("/api/model-weights")
        assert resp.status_code == 200
        data = resp.json()
        assert "weights" in data
        assert "short" in data["weights"]

    def test_retraining_history_json_format(self):
        resp = client.get("/api/retraining-history")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        rec = data[0]
        assert rec["model_name"] == "xgboost"
        assert rec["replaced"] is True
        assert "champion_mae" in rec
        assert "challenger_mae" in rec

    def test_feature_importance_json_format(self):
        resp = client.get("/api/feature-importance/lstm")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "lstm"
        assert data["feature_importance"] is not None

    # ── 수집 → 저장 → 조회 흐름 ──────────────────────────────

    def test_price_store_and_query(self):
        """가격 데이터 저장 후 API로 조회 가능한지 검증."""
        session = get_session()
        try:
            count = session.query(PriceData).count()
            assert count >= 5
        finally:
            session.close()

        resp = client.get("/api/current-price")
        data = resp.json()
        assert data["price"] is not None
        assert data["price"] > 0

    def test_prediction_store_and_query(self):
        """예측 저장 후 API로 조회 가능한지 검증."""
        resp = client.get("/api/predictions/short")
        data = resp.json()
        preds = data["predictions"]
        assert len(preds) >= 5  # 4 models + ensemble

    @classmethod
    def teardown_class(cls):
        try:
            os.remove("tests/_test_integration.db")
        except OSError:
            pass


class TestRetrainingPipelineHelpers:
    """재학습 파이프라인 헬퍼 함수 단위 검증."""

    def test_build_model_instance(self):
        from scheduler.job_scheduler import _build_model_instance
        for name in ["lstm", "xgboost", "rf", "transformer"]:
            model = _build_model_instance(name)
            assert model is not None
            assert hasattr(model, "train")
            assert hasattr(model, "predict")

    def test_build_model_instance_invalid(self):
        from scheduler.job_scheduler import _build_model_instance
        import pytest
        with pytest.raises(ValueError):
            _build_model_instance("unknown_model")

    def test_evaluate_model_with_trained_model(self):
        from scheduler.job_scheduler import _evaluate_model
        from prediction.random_forest_model import RandomForestModel

        rng = np.random.RandomState(42)
        X = rng.rand(50, 5)
        y = X[:, 0] * 2 + 1 + rng.rand(50) * 0.1

        model = RandomForestModel(n_estimators=10)
        model.train(X, y)

        metrics = _evaluate_model(model, X[:10], y[:10])
        assert metrics.mae >= 0
        assert metrics.mape >= 0
        assert 0 <= metrics.direction_accuracy <= 100
