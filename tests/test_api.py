"""API endpoint tests."""
import os
os.environ["DATABASE_URL"] = "sqlite:///tests/_test_api.db"

from datetime import datetime, timedelta, timezone
from db.database import init_db, get_session, engine
from db.models import (
    Base, ElliottWaveData, EnsembleWeight, HitRateRecord,
    OnchainData, PairData, PredictionRecord, PriceData,
    RetrainingHistory, SentimentData, TechnicalIndicator, WyckoffData,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.routes import router


def _reset():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def _seed():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    s = get_session()
    s.add(PriceData(timestamp=now - timedelta(hours=25), open=0.5, high=0.52, low=0.49, close=0.51, volume=1e6))
    s.add(PriceData(timestamp=now, open=0.52, high=0.55, low=0.51, close=0.54, volume=1.2e6))
    s.add(TechnicalIndicator(timestamp=now, rsi_14=55.0, macd=0.01, macd_signal=0.005, macd_histogram=0.005, bb_upper=0.6, bb_middle=0.54, bb_lower=0.48, sma_5=0.53, sma_10=0.52, sma_20=0.51, sma_50=0.5, sma_200=0.48, ema_12=0.53, ema_26=0.52))
    s.add(ElliottWaveData(timestamp=now, wave_number="3", wave_type="impulse", start_price=0.45, end_price=0.55, start_time=now - timedelta(days=5), end_time=now, wave_degree="Primary", is_valid=True, fibonacci_targets={"0.236": 0.47}))
    s.add(WyckoffData(timestamp=now, event_type="Spring", pattern_type="accumulation", price=0.5, volume=5e5, market_phase="Accumulation", wyckoff_phase="C", confidence_score=75.0, is_trend_reversal=True))
    s.add(PairData(timestamp=now, asset_name="XRP/BTC", price=1.5e-5, correlation_with_xrp=0.85))
    s.add(PairData(timestamp=now, asset_name="BTC/USD", price=42000.0, correlation_with_xrp=0.72))
    s.add(SentimentData(timestamp=now, google_trend_score=65.0, sns_mention_score=70.0, sns_sentiment_score=55.0, fear_greed_index=60.0))
    s.add(OnchainData(timestamp=now, active_wallets=150000, new_wallets=5000, transaction_count=800000, total_volume_xrp=5e8, whale_tx_count=25, whale_tx_volume=5e7))
    for m in ["lstm", "xgboost", "rf", "transformer", "ensemble"]:
        s.add(PredictionRecord(timestamp=now, model_name=m, timeframe="short", predicted_price=0.56, predicted_direction="up", up_probability=65.0, down_probability=35.0, confidence=70.0, feature_importance={"rsi_14": 0.3}))
    s.add(HitRateRecord(date=now, model_name="lstm", timeframe="short", direction_hit_rate=72.0, range_hit_rate=55.0))
    s.add(HitRateRecord(date=now, model_name="ensemble", timeframe="short", direction_hit_rate=78.0, range_hit_rate=60.0))
    for m in ["lstm", "xgboost", "rf", "transformer"]:
        s.add(EnsembleWeight(timestamp=now, model_name=m, weight=0.25, timeframe="short"))
    s.add(RetrainingHistory(timestamp=now, model_name="lstm", method="incremental", data_period_start=now - timedelta(days=90), data_period_end=now, champion_mae=0.02, champion_mape=3.5, champion_direction_accuracy=70.0, challenger_mae=0.018, challenger_mape=3.2, challenger_direction_accuracy=73.0, replaced=True))
    s.commit()
    s.close()


app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestEmptyDB:
    @classmethod
    def setup_class(cls):
        _reset()

    def test_current_price(self):
        assert client.get("/api/current-price").json()["price"] is None

    def test_predictions(self):
        d = client.get("/api/predictions/short").json()
        assert d["timeframe"] == "short" and d["predictions"] == {}

    def test_predictions_invalid(self):
        assert client.get("/api/predictions/invalid").status_code == 400

    def test_indicators(self):
        assert client.get("/api/technical-indicators").json()["indicators"] is None

    def test_elliott(self):
        d = client.get("/api/elliott-wave").json()
        assert d["waves"] == [] and d["current_position"] is None

    def test_wyckoff(self):
        d = client.get("/api/wyckoff").json()
        assert d["events"] == [] and d["market_phase"] is None

    def test_correlations(self):
        assert client.get("/api/correlations").json()["correlations"] == {}

    def test_sentiment(self):
        assert client.get("/api/sentiment").json()["sentiment"] is None

    def test_onchain(self):
        assert client.get("/api/onchain").json()["onchain"] is None

    def test_hit_rates(self):
        assert client.get("/api/hit-rates").json()["hit_rates"] == []

    def test_feature_importance(self):
        d = client.get("/api/feature-importance/lstm").json()
        assert d["model_name"] == "lstm" and d["feature_importance"] is None

    def test_retraining(self):
        assert client.get("/api/retraining-history").json() == []

    def test_weights(self):
        assert client.get("/api/model-weights").json()["weights"] == {}


class TestWithData:
    @classmethod
    def setup_class(cls):
        _reset()
        _seed()

    def test_current_price(self):
        d = client.get("/api/current-price").json()
        assert d["price"] == 0.54 and d["change_24h"] is not None

    def test_predictions(self):
        d = client.get("/api/predictions/short").json()
        assert "lstm" in d["predictions"] and "ensemble" in d["predictions"]

    def test_indicators(self):
        assert client.get("/api/technical-indicators").json()["indicators"]["rsi_14"] == 55.0

    def test_elliott(self):
        assert len(client.get("/api/elliott-wave").json()["waves"]) == 1

    def test_wyckoff(self):
        assert client.get("/api/wyckoff").json()["market_phase"] == "Accumulation"

    def test_correlations(self):
        assert "XRP/BTC" in client.get("/api/correlations").json()["correlations"]

    def test_sentiment(self):
        assert client.get("/api/sentiment").json()["sentiment"]["google_trend_score"] == 65.0

    def test_onchain(self):
        assert client.get("/api/onchain").json()["onchain"]["active_wallets"] == 150000

    def test_hit_rates(self):
        assert len(client.get("/api/hit-rates").json()["hit_rates"]) == 2

    def test_feature_importance(self):
        assert client.get("/api/feature-importance/lstm").json()["feature_importance"] is not None

    def test_retraining(self):
        h = client.get("/api/retraining-history").json()
        assert len(h) == 1 and h[0]["replaced"] is True

    def test_weights(self):
        w = client.get("/api/model-weights").json()["weights"]
        assert "short" in w and w["short"]["lstm"]["weight"] == 0.25

    @classmethod
    def teardown_class(cls):
        try:
            os.remove("tests/_test_api.db")
        except OSError:
            pass
