"""상관계수 범위 유효성 속성 테스트.

Property 11: 두 시계열 데이터에 대해, calculate_correlation이 반환하는
상관계수는 -1 이상 1 이하여야 하며, 동일한 시계열에 대한 상관계수는 1.0이어야 한다.

**Validates: Requirements 2.5**
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from collectors.pair_collector import PairDataModule


@pytest.fixture
def module():
    return PairDataModule()


def _make_close_df(values: list[float]) -> pd.DataFrame:
    """close 컬럼을 가진 DataFrame을 생성한다."""
    return pd.DataFrame({"close": values})


# ── Property 11: 상관계수 범위 유효성 ────────────────────────


class TestCorrelationRangeProperty:
    """Feature: xrp-price-prediction-dashboard, Property 11: 상관계수 범위 유효성"""

    @settings(max_examples=100)
    @given(
        xrp_prices=st.lists(
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=200,
        ),
        pair_prices=st.lists(
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=200,
        ),
    )
    def test_correlation_within_bounds(self, xrp_prices, pair_prices):
        """두 시계열의 상관계수는 -1 이상 1 이하여야 한다.

        **Validates: Requirements 2.5**
        """
        module = PairDataModule()
        xrp_df = _make_close_df(xrp_prices)
        pair_df = _make_close_df(pair_prices)

        corr = module.calculate_correlation(xrp_df, pair_df)

        assert -1.0 <= corr <= 1.0, f"상관계수가 범위를 벗어남: {corr}"

    @settings(max_examples=100)
    @given(
        prices=st.lists(
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=200,
        ),
    )
    def test_self_correlation_is_one(self, prices):
        """동일한 시계열에 대한 상관계수는 1.0이어야 한다.

        **Validates: Requirements 2.5**
        """
        module = PairDataModule()
        df = _make_close_df(prices)

        corr = module.calculate_correlation(df, df)

        # 모든 값이 동일하면 std=0이므로 특수 처리됨
        if len(set(prices)) == 1:
            assert corr == 1.0
        else:
            assert corr == pytest.approx(1.0, abs=1e-10), (
                f"동일 시계열 상관계수가 1.0이 아님: {corr}"
            )


# ── 단위 테스트 ──────────────────────────────────────────────


class TestCorrelationEdgeCases:
    def test_empty_dataframes(self, module):
        """빈 DataFrame에 대해 0.0을 반환한다."""
        empty_df = pd.DataFrame({"close": []})
        assert module.calculate_correlation(empty_df, empty_df) == 0.0

    def test_single_element(self, module):
        """단일 요소 DataFrame에 대해 0.0을 반환한다."""
        df = _make_close_df([1.0])
        assert module.calculate_correlation(df, df) == 0.0

    def test_perfectly_correlated(self, module):
        """완벽한 양의 상관관계."""
        xrp_df = _make_close_df([1.0, 2.0, 3.0, 4.0, 5.0])
        pair_df = _make_close_df([10.0, 20.0, 30.0, 40.0, 50.0])
        corr = module.calculate_correlation(xrp_df, pair_df)
        assert corr == pytest.approx(1.0, abs=1e-10)

    def test_perfectly_negatively_correlated(self, module):
        """완벽한 음의 상관관계."""
        xrp_df = _make_close_df([1.0, 2.0, 3.0, 4.0, 5.0])
        pair_df = _make_close_df([50.0, 40.0, 30.0, 20.0, 10.0])
        corr = module.calculate_correlation(xrp_df, pair_df)
        assert corr == pytest.approx(-1.0, abs=1e-10)

    def test_different_lengths(self, module):
        """길이가 다른 시계열은 짧은 쪽에 맞춘다."""
        xrp_df = _make_close_df([1.0, 2.0, 3.0, 4.0, 5.0])
        pair_df = _make_close_df([10.0, 20.0, 30.0])
        corr = module.calculate_correlation(xrp_df, pair_df)
        assert -1.0 <= corr <= 1.0

    def test_constant_series(self, module):
        """상수 시계열에 대해 동일하면 1.0을 반환한다."""
        df = _make_close_df([5.0, 5.0, 5.0, 5.0, 5.0])
        corr = module.calculate_correlation(df, df)
        assert corr == 1.0
