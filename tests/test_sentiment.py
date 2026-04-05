"""심리 데이터 정규화 범위 속성 테스트.

Property 12: 모든 입력값과 소스에 대해, normalize 함수의 반환값은
항상 0 이상 100 이하여야 한다.

**Validates: Requirements 3.5**
"""

import pytest
from hypothesis import given, settings, strategies as st

from collectors.sentiment_collector import SentimentModule


@pytest.fixture
def module():
    return SentimentModule()


# ── Property 12: 심리 데이터 정규화 범위 ─────────────────────


class TestNormalizeRangeProperty:
    """Feature: xrp-price-prediction-dashboard, Property 12: 심리 데이터 정규화 범위"""

    @settings(max_examples=100)
    @given(
        value=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        source=st.sampled_from(
            ["google_trends", "twitter_mention", "twitter_sentiment", "fear_greed"]
        ),
    )
    def test_normalize_known_sources_in_range(self, value, source):
        """알려진 소스에 대해 정규화 결과는 0~100 범위여야 한다.

        **Validates: Requirements 3.5**
        """
        module = SentimentModule()
        result = module.normalize(value, source)

        assert 0.0 <= result <= 100.0, (
            f"정규화 결과가 범위를 벗어남: {result} (value={value}, source={source})"
        )

    @settings(max_examples=100)
    @given(
        value=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        source=st.text(min_size=1, max_size=20),
    )
    def test_normalize_any_source_in_range(self, value, source):
        """알 수 없는 소스를 포함한 모든 소스에 대해 정규화 결과는 0~100 범위여야 한다.

        **Validates: Requirements 3.5**
        """
        module = SentimentModule()
        result = module.normalize(value, source)

        assert 0.0 <= result <= 100.0, (
            f"정규화 결과가 범위를 벗어남: {result} (value={value}, source={source})"
        )


# ── 단위 테스트 ──────────────────────────────────────────────


class TestNormalizeEdgeCases:
    def test_google_trends_midpoint(self, module):
        """Google Trends 50은 정규화 후 50이어야 한다."""
        assert module.normalize(50.0, "google_trends") == pytest.approx(50.0)

    def test_google_trends_zero(self, module):
        """Google Trends 0은 정규화 후 0이어야 한다."""
        assert module.normalize(0.0, "google_trends") == pytest.approx(0.0)

    def test_google_trends_max(self, module):
        """Google Trends 100은 정규화 후 100이어야 한다."""
        assert module.normalize(100.0, "google_trends") == pytest.approx(100.0)

    def test_fear_greed_value(self, module):
        """공포탐욕지수 72는 정규화 후 72여야 한다."""
        assert module.normalize(72.0, "fear_greed") == pytest.approx(72.0)

    def test_twitter_sentiment_negative(self, module):
        """Twitter 감성 -1.0은 정규화 후 0이어야 한다."""
        assert module.normalize(-1.0, "twitter_sentiment") == pytest.approx(0.0)

    def test_twitter_sentiment_positive(self, module):
        """Twitter 감성 1.0은 정규화 후 100이어야 한다."""
        assert module.normalize(1.0, "twitter_sentiment") == pytest.approx(100.0)

    def test_twitter_sentiment_neutral(self, module):
        """Twitter 감성 0.0은 정규화 후 50이어야 한다."""
        assert module.normalize(0.0, "twitter_sentiment") == pytest.approx(50.0)

    def test_twitter_mention_500(self, module):
        """Twitter 언급량 500은 정규화 후 50이어야 한다."""
        assert module.normalize(500.0, "twitter_mention") == pytest.approx(50.0)

    def test_value_below_range_clamped(self, module):
        """범위 아래 값은 0으로 클램핑된다."""
        result = module.normalize(-100.0, "google_trends")
        assert result == 0.0

    def test_value_above_range_clamped(self, module):
        """범위 위 값은 100으로 클램핑된다."""
        result = module.normalize(200.0, "google_trends")
        assert result == 100.0

    def test_unknown_source_clamped(self, module):
        """알 수 없는 소스의 값은 0~100으로 클램핑된다."""
        assert module.normalize(50.0, "unknown_source") == 50.0
        assert module.normalize(-10.0, "unknown_source") == 0.0
        assert module.normalize(150.0, "unknown_source") == 100.0
