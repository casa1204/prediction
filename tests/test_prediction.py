"""예측 모델 테스트 — BaseModel, LSTM, XGBoost, Random Forest, Transformer 모델 테스트.

단위 테스트와 속성 기반 테스트를 포함한다.
"""

import os
import tempfile

import numpy as np
import pytest

from prediction.base_model import BaseModel, PredictionResult
from prediction.lstm_model import LSTMModel
from prediction.random_forest_model import RandomForestModel
from prediction.transformer_model import TransformerModel
from prediction.xgboost_model import XGBoostModel


# ── 헬퍼 ─────────────────────────────────────────────────────

def _make_dataset(
    n_samples: int = 100,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """테스트용 합성 데이터셋 생성."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features) * 10
    # 간단한 선형 관계 + 노이즈
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.randn(n_samples) * 0.1
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


# ── PredictionResult 테스트 ──────────────────────────────────

class TestPredictionResult:
    def test_creation(self):
        result = PredictionResult(
            predicted_price=1.5,
            up_probability=60.0,
            down_probability=40.0,
            confidence=75.0,
            timeframe="short",
            feature_importance={"feat_0": 0.5, "feat_1": 0.5},
        )
        assert result.predicted_price == 1.5
        assert result.up_probability + result.down_probability == 100.0
        assert result.timeframe == "short"

    def test_default_feature_importance(self):
        result = PredictionResult(
            predicted_price=1.0,
            up_probability=50.0,
            down_probability=50.0,
            confidence=50.0,
            timeframe="mid",
        )
        assert result.feature_importance == {}


# ── BaseModel 테스트 ─────────────────────────────────────────

class TestBaseModel:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_default_feature_importance(self):
        """BaseModel의 기본 get_feature_importance는 균등 분배."""
        model = XGBoostModel(feature_names=["a", "b", "c"])
        # 학습 전이므로 기본 구현 사용
        importance = model.get_feature_importance()
        assert len(importance) == 3
        assert abs(sum(importance.values()) - 1.0) < 0.01


# ── 모델 공통 테스트 (파라미터화) ────────────────────────────

_MODELS_WITH_PARAMS = [
    ("lstm", lambda fn: LSTMModel(feature_names=fn, seq_length=10, epochs=5, hidden_size=16)),
    ("xgboost", lambda fn: XGBoostModel(feature_names=fn, n_estimators=10)),
    ("rf", lambda fn: RandomForestModel(feature_names=fn, n_estimators=10)),
    ("transformer", lambda fn: TransformerModel(feature_names=fn, seq_length=10, epochs=5, d_model=16, nhead=4)),
]


@pytest.mark.parametrize("name,model_factory", _MODELS_WITH_PARAMS, ids=[m[0] for m in _MODELS_WITH_PARAMS])
class TestModelCommon:
    """모든 모델에 공통으로 적용되는 테스트."""

    def test_train_and_predict(self, name, model_factory):
        """학습 후 예측이 PredictionResult를 반환한다."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)
        model.train(X, y)

        result = model.predict(X, "short")
        assert isinstance(result, PredictionResult)
        assert result.timeframe == "short"

    def test_prediction_probabilities_sum_to_100(self, name, model_factory):
        """up_probability + down_probability == 100."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)
        model.train(X, y)

        for tf in ("short", "mid", "long"):
            result = model.predict(X, tf)
            assert abs(result.up_probability + result.down_probability - 100.0) < 0.01

    def test_confidence_range(self, name, model_factory):
        """confidence는 0~100 범위."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)
        model.train(X, y)

        result = model.predict(X, "mid")
        assert 0.0 <= result.confidence <= 100.0

    def test_invalid_timeframe_raises(self, name, model_factory):
        """유효하지 않은 timeframe은 ValueError."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)
        model.train(X, y)

        with pytest.raises(ValueError):
            model.predict(X, "invalid")

    def test_predict_before_train_raises(self, name, model_factory):
        """학습 전 예측 시 RuntimeError."""
        _, _, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)

        with pytest.raises(RuntimeError):
            model.predict(np.zeros((10, 5)), "short")

    def test_save_and_load(self, name, model_factory):
        """저장 후 로드하면 동일한 예측 결과를 반환한다."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)
        model.train(X, y)

        result_before = model.predict(X, "short")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"{name}_model.pt")
            model.save(path)
            assert os.path.exists(path)

            model2 = model_factory(fn)
            model2.load(path)
            result_after = model2.predict(X, "short")

        assert abs(result_before.predicted_price - result_after.predicted_price) < 0.01

    def test_feature_importance_sums_to_one(self, name, model_factory):
        """학습 후 피처 기여도 합 ≈ 1.0."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)
        model.train(X, y)

        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert abs(sum(importance.values()) - 1.0) < 0.02

    def test_feature_importance_all_non_negative(self, name, model_factory):
        """피처 기여도는 모두 0 이상."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)
        model.train(X, y)

        importance = model.get_feature_importance()
        for v in importance.values():
            assert v >= 0.0

    def test_save_before_train_raises(self, name, model_factory):
        """학습 전 저장 시 RuntimeError."""
        _, _, fn = _make_dataset(n_samples=100, n_features=5)
        model = model_factory(fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            with pytest.raises(RuntimeError):
                model.save(path)


# ── LSTM 특화 테스트 ─────────────────────────────────────────

class TestLSTMModel:
    def test_insufficient_data_raises(self):
        """데이터가 시퀀스 길이보다 짧으면 ValueError."""
        model = LSTMModel(seq_length=50, epochs=1)
        X = np.random.rand(30, 5)
        y = np.random.rand(30)
        with pytest.raises(ValueError):
            model.train(X, y)

    def test_short_input_prediction_with_padding(self):
        """예측 시 입력이 시퀀스 길이보다 짧으면 패딩 처리."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = LSTMModel(feature_names=fn, seq_length=10, epochs=3, hidden_size=16)
        model.train(X, y)

        # 시퀀스 길이보다 짧은 입력
        short_X = X[:5]
        result = model.predict(short_X, "short")
        assert isinstance(result, PredictionResult)


# ── Transformer 특화 테스트 ──────────────────────────────────

class TestTransformerModel:
    def test_insufficient_data_raises(self):
        """데이터가 시퀀스 길이보다 짧으면 ValueError."""
        model = TransformerModel(seq_length=50, epochs=1)
        X = np.random.rand(30, 5)
        y = np.random.rand(30)
        with pytest.raises(ValueError):
            model.train(X, y)

    def test_short_input_prediction_with_padding(self):
        """예측 시 입력이 시퀀스 길이보다 짧으면 패딩 처리."""
        X, y, fn = _make_dataset(n_samples=100, n_features=5)
        model = TransformerModel(feature_names=fn, seq_length=10, epochs=3, d_model=16, nhead=4)
        model.train(X, y)

        short_X = X[:5]
        result = model.predict(short_X, "short")
        assert isinstance(result, PredictionResult)


# ── 속성 기반 테스트 (Property-Based Tests) ──────────────────

from hypothesis import given, settings, strategies as st


# Feature: xrp-price-prediction-dashboard, Property 14: 예측 결과 유효성
@settings(max_examples=100)
@given(
    seed=st.integers(min_value=0, max_value=10000),
    timeframe=st.sampled_from(["short", "mid", "long"]),
    model_type=st.sampled_from(["xgboost", "rf"]),
)
def test_property_14_prediction_result_validity(seed, timeframe, model_type):
    """Property 14: predict가 반환하는 PredictionResult의 상승 확률과 하락 확률의 합은 100이고,
    신뢰도는 0~100 범위이며, timeframe은 'short', 'mid', 'long' 중 하나여야 한다.

    **Validates: Requirements 5.3, 5.4**
    """
    rng = np.random.RandomState(seed)
    n_samples, n_features = 100, 5
    X = rng.rand(n_samples, n_features) * 10
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.randn(n_samples) * 0.1
    feature_names = [f"feat_{i}" for i in range(n_features)]

    if model_type == "xgboost":
        model = XGBoostModel(feature_names=feature_names, n_estimators=10)
    else:
        model = RandomForestModel(feature_names=feature_names, n_estimators=10)

    model.train(X, y)
    result = model.predict(X, timeframe)

    # 상승 확률 + 하락 확률 == 100
    assert abs(result.up_probability + result.down_probability - 100.0) < 0.01, (
        f"up_prob({result.up_probability}) + down_prob({result.down_probability}) != 100"
    )
    # 신뢰도 0~100
    assert 0.0 <= result.confidence <= 100.0, (
        f"confidence({result.confidence}) out of range [0, 100]"
    )
    # timeframe 유효성
    assert result.timeframe in ("short", "mid", "long"), (
        f"timeframe({result.timeframe}) not in valid set"
    )


# Feature: xrp-price-prediction-dashboard, Property 18: 피처 기여도 유효성
@settings(max_examples=100)
@given(
    seed=st.integers(min_value=0, max_value=10000),
    n_features=st.integers(min_value=2, max_value=20),
    model_type=st.sampled_from(["xgboost", "rf"]),
)
def test_property_18_feature_importance_validity(seed, n_features, model_type):
    """Property 18: 학습된 Individual_Model에 대해, get_feature_importance가 반환하는
    모든 값은 0 이상이고, 전체 값의 합은 1.0(허용 오차 ±0.01)이어야 한다.

    **Validates: Requirements 5.10**
    """
    rng = np.random.RandomState(seed)
    n_samples = 100
    X = rng.rand(n_samples, n_features) * 10
    y = X[:, 0] * 2 + rng.randn(n_samples) * 0.1
    feature_names = [f"feat_{i}" for i in range(n_features)]

    if model_type == "xgboost":
        model = XGBoostModel(feature_names=feature_names, n_estimators=10)
    else:
        model = RandomForestModel(feature_names=feature_names, n_estimators=10)

    model.train(X, y)
    importance = model.get_feature_importance()

    # 모든 값 >= 0
    for name, value in importance.items():
        assert value >= 0.0, f"feature '{name}' importance({value}) < 0"

    # 합 ≈ 1.0
    total = sum(importance.values())
    assert abs(total - 1.0) < 0.01, f"feature importance sum({total}) != 1.0 (±0.01)"
