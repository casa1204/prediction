"""피처 엔지니어링 모듈 — 모든 데이터 소스를 통합하여 학습 데이터셋을 구성한다.

기술 지표, 엘리엇 파동, 와이코프, 페어, 심리, 온체인 데이터를 DB에서 조회하여
통합 피처 매트릭스(X)와 타겟(y)을 생성한다.

Requirements: 5.1, 5.9
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.models import (
    DailyPriceData,
    ElliottWaveData,
    EtfData,
    OnchainData,
    PairData,
    PriceData,
    SentimentData,
    TechnicalIndicator,
    WyckoffData,
)

logger = logging.getLogger(__name__)

# 엘리엇 파동 번호 → 정수 인코딩
_WAVE_NUMBER_ENCODING: dict[str, int] = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "A": 6, "B": 7, "C": 8,
}

# 와이코프 시장 단계 → 정수 인코딩
_MARKET_PHASE_ENCODING: dict[str, int] = {
    "Accumulation": 1,
    "Markup": 2,
    "Distribution": 3,
    "Markdown": 4,
}

# 와이코프 Phase → 정수 인코딩
_WYCKOFF_PHASE_ENCODING: dict[str, int] = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5,
}

# 최소 데이터 일수 기준
_MIN_DAYS = 90


class FeatureEngineering:
    """모든 데이터 소스를 통합하여 ML 학습 데이터셋을 구성한다.

    build_dataset()은 DB에서 기술 지표, 엘리엇 파동, 와이코프, 페어,
    심리, 온체인 데이터를 조회하고 날짜 기준으로 조인하여 통합 피처
    매트릭스(X)와 타겟(y: 다음 날 종가)을 생성한다.
    """

    # ── 기술 지표 피처 이름 ──────────────────────────────────
    TECHNICAL_FEATURES: list[str] = [
        "rsi_14", "macd", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "obv", "sma_100", "disparity_50", "disparity_100", "disparity_200",
    ]

    # ── 엘리엇 파동 피처 이름 ────────────────────────────────
    ELLIOTT_FEATURES: list[str] = [
        "current_wave_number", "next_direction",
    ]

    # ── 와이코프 피처 이름 ───────────────────────────────────
    WYCKOFF_FEATURES: list[str] = [
        "market_phase", "wyckoff_phase", "confidence_score",
    ]

    # ── 심리 피처 이름 ───────────────────────────────────────
    SENTIMENT_FEATURES: list[str] = [
        "google_trend_score", "sns_mention_score",
        "sns_sentiment_score", "fear_greed_index",
        "trend_macro_score", "trend_etf_regulatory_score",
        "trend_sentiment_score",
        "attention_ratio", "fomo_spread", "macro_aggregate",
    ]

    # ── 온체인 피처 이름 ─────────────────────────────────────
    ONCHAIN_FEATURES: list[str] = [
        "active_wallets", "new_wallets", "transaction_count",
        "total_volume_xrp", "whale_tx_count", "whale_tx_volume",
    ]

    # ── ETF 피처 이름 (동적으로 생성되므로 기본 목록만 정의) ──
    ETF_STATIC_FEATURES: list[str] = [
        "etf_total_volume",
    ]

    def build_dataset(
        self, session: Session, timeframe: str = "short",
    ) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
        """DB에서 모든 데이터 소스를 조회하여 통합 피처 매트릭스와 타겟을 생성한다.

        Parameters
        ----------
        session : Session
            SQLAlchemy 세션.
        timeframe : str
            예측 타임프레임. "short"=1일 후, "mid"=7일 후, "long"=30일 후.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str], bool]
            - X: 피처 매트릭스 (n_samples, n_features) — close 포함
            - y: 타겟 벡터 (타임프레임별 미래 종가)
            - feature_names: 피처 이름 리스트
            - insufficient_data: 90일 미만 데이터 경고 플래그
        """
        # 타임프레임별 shift 값
        _SHIFT_MAP = {"short": 1, "mid": 7, "long": 30}
        shift_days = _SHIFT_MAP.get(timeframe, 1)

        # 1) 가격 데이터 (기준 날짜 축)
        price_df = self._load_price_data(session)
        if price_df.empty:
            logger.warning("가격 데이터가 없습니다.")
            return np.array([]), np.array([]), [], True

        # 2) 각 데이터 소스 로드
        tech_df = self._load_technical_indicators(session)
        elliott_df = self._load_elliott_wave(session)
        wyckoff_df = self._load_wyckoff(session)
        pair_df = self._load_pair_data(session)
        sentiment_df = self._load_sentiment(session)
        onchain_df = self._load_onchain(session)
        etf_df = self._load_etf_data(session)

        # 3) 날짜 기준으로 병합
        merged = price_df[["date", "close"]].copy()

        merged = self._merge_on_date(merged, tech_df, self.TECHNICAL_FEATURES)
        merged = self._merge_on_date(merged, elliott_df, self.ELLIOTT_FEATURES)
        merged = self._merge_on_date(merged, wyckoff_df, self.WYCKOFF_FEATURES)
        merged = self._merge_on_date(merged, sentiment_df, self.SENTIMENT_FEATURES)
        merged = self._merge_on_date(merged, onchain_df, self.ONCHAIN_FEATURES)

        # 페어 데이터는 자산별로 피벗되어 있으므로 별도 처리
        pair_feature_names: list[str] = []
        if not pair_df.empty:
            merged, pair_feature_names = self._merge_pair_data(merged, pair_df)

        # ETF 데이터는 티커별로 피벗되어 있으므로 별도 처리
        etf_feature_names: list[str] = []
        if not etf_df.empty:
            merged, etf_feature_names = self._merge_etf_data(merged, etf_df)

        # 4) 피처 이름 목록 구성 — close를 첫 번째 피처로 포함
        feature_names = (
            ["close"]
            + self.TECHNICAL_FEATURES
            + self.ELLIOTT_FEATURES
            + self.WYCKOFF_FEATURES
            + pair_feature_names
            + self.SENTIMENT_FEATURES
            + self.ONCHAIN_FEATURES
            + etf_feature_names
        )

        # 5) 누락 피처 처리: ffill → 0 대체
        for col in feature_names:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].ffill().fillna(0.0)

        # 5.5) 전부 0인 피처 제거 (데이터 미수집 피처는 노이즈만 추가)
        zero_cols = [
            col for col in feature_names
            if col != "close" and (merged[col] == 0.0).all()
        ]
        if zero_cols:
            feature_names = [col for col in feature_names if col not in zero_cols]
            logger.info(
                "전부 0인 피처 %d개 제외: %s",
                len(zero_cols), ", ".join(zero_cols),
            )        # 6) 타겟 생성: 타임프레임별 미래 종가
        merged["target"] = merged["close"].shift(-shift_days)
        merged = merged.dropna(subset=["target"])

        if merged.empty:
            logger.warning("타겟 생성 후 유효한 데이터가 없습니다.")
            return np.array([]), np.array([]), feature_names, True

        # 7) 90일 미만 데이터 경고
        n_days = len(merged)
        insufficient_data = n_days < _MIN_DAYS
        if insufficient_data:
            logger.warning(
                "학습 데이터가 %d일로 최소 기준(%d일) 미만입니다. "
                "예측 결과에 '데이터 부족' 경고가 포함됩니다.",
                n_days, _MIN_DAYS,
            )

        # 8) numpy 배열 변환
        X = merged[feature_names].to_numpy(dtype=np.float64)
        y = merged["target"].to_numpy(dtype=np.float64)

        logger.info(
            "데이터셋 구성 완료: %d 샘플, %d 피처, timeframe=%s, 데이터 부족=%s",
            X.shape[0], X.shape[1], timeframe, insufficient_data,
        )

        return X, y, feature_names, insufficient_data

    # ── 데이터 로드 메서드 ───────────────────────────────────

    def _load_price_data(self, session: Session) -> pd.DataFrame:
        """DailyPriceData 테이블에서 일별 종가를 로드한다.

        일봉 데이터(약 1000일치)를 사용하여 학습 데이터 양을 극대화한다.
        일봉이 없으면 시간봉(PriceData)에서 일별 종가를 추출한다.
        """
        # 우선 일봉 데이터 시도
        rows = (
            session.query(DailyPriceData)
            .order_by(DailyPriceData.timestamp.asc())
            .all()
        )
        if rows:
            data = [{"date": r.timestamp.date(), "close": r.close} for r in rows]
            df = pd.DataFrame(data)
            df = df.groupby("date", as_index=False).last()
            df = df.sort_values("date").reset_index(drop=True)
            logger.info("일봉 데이터 %d일 로드", len(df))
            return df

        # 일봉이 없으면 시간봉에서 추출 (폴백)
        rows = (
            session.query(PriceData)
            .order_by(PriceData.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = [{"date": r.timestamp.date(), "close": r.close} for r in rows]
        df = pd.DataFrame(data)
        df = df.groupby("date", as_index=False).last()
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("시간봉에서 일별 종가 %d일 추출 (폴백)", len(df))
        return df

    def _load_technical_indicators(self, session: Session) -> pd.DataFrame:
        """TechnicalIndicator 테이블에서 기술 지표를 로드한다."""
        rows = (
            session.query(TechnicalIndicator)
            .order_by(TechnicalIndicator.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = []
        for r in rows:
            record: dict = {"date": r.timestamp.date()}
            for feat in self.TECHNICAL_FEATURES:
                record[feat] = getattr(r, feat, None)
            data.append(record)

        df = pd.DataFrame(data)
        df = df.groupby("date", as_index=False).last()
        return df

    def _load_elliott_wave(self, session: Session) -> pd.DataFrame:
        """ElliottWaveData 테이블에서 최신 파동 정보를 날짜별로 로드한다."""
        rows = (
            session.query(ElliottWaveData)
            .order_by(ElliottWaveData.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = []
        for r in rows:
            wave_num = _WAVE_NUMBER_ENCODING.get(r.wave_number, 0)
            # next_direction: impulse 파동이면 1(up), corrective이면 0(down)
            next_dir = 1 if r.wave_type == "impulse" else 0
            data.append({
                "date": r.timestamp.date(),
                "current_wave_number": wave_num,
                "next_direction": next_dir,
            })

        df = pd.DataFrame(data)
        df = df.groupby("date", as_index=False).last()
        return df

    def _load_wyckoff(self, session: Session) -> pd.DataFrame:
        """WyckoffData 테이블에서 와이코프 분석 결과를 날짜별로 로드한다."""
        rows = (
            session.query(WyckoffData)
            .order_by(WyckoffData.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = []
        for r in rows:
            data.append({
                "date": r.timestamp.date(),
                "market_phase": _MARKET_PHASE_ENCODING.get(r.market_phase, 0),
                "wyckoff_phase": _WYCKOFF_PHASE_ENCODING.get(r.wyckoff_phase, 0),
                "confidence_score": r.confidence_score if r.confidence_score is not None else 0.0,
            })

        df = pd.DataFrame(data)
        df = df.groupby("date", as_index=False).last()
        return df

    def _load_pair_data(self, session: Session) -> pd.DataFrame:
        """PairData 테이블에서 페어 데이터를 로드한다."""
        rows = (
            session.query(PairData)
            .order_by(PairData.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = []
        for r in rows:
            data.append({
                "date": r.timestamp.date(),
                "asset_name": r.asset_name,
                "price": r.price,
                "correlation_with_xrp": r.correlation_with_xrp if r.correlation_with_xrp is not None else 0.0,
            })

        return pd.DataFrame(data)

    def _load_sentiment(self, session: Session) -> pd.DataFrame:
        """SentimentData 테이블에서 심리 데이터를 로드한다."""
        rows = (
            session.query(SentimentData)
            .order_by(SentimentData.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = []
        for r in rows:
            data.append({
                "date": r.timestamp.date(),
                "google_trend_score": r.google_trend_score,
                "sns_mention_score": r.sns_mention_score,
                "sns_sentiment_score": r.sns_sentiment_score,
                "fear_greed_index": r.fear_greed_index,
                "trend_macro_score": r.trend_macro_score,
                "trend_etf_regulatory_score": r.trend_etf_regulatory_score,
                "trend_sentiment_score": r.trend_sentiment_score,
                "attention_ratio": r.attention_ratio,
                "fomo_spread": r.fomo_spread,
                "macro_aggregate": r.macro_aggregate,
            })

        df = pd.DataFrame(data)
        df = df.groupby("date", as_index=False).last()
        return df

    def _load_onchain(self, session: Session) -> pd.DataFrame:
        """OnchainData 테이블에서 온체인 데이터를 로드한다."""
        rows = (
            session.query(OnchainData)
            .order_by(OnchainData.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = []
        for r in rows:
            data.append({
                "date": r.timestamp.date(),
                "active_wallets": r.active_wallets,
                "new_wallets": r.new_wallets,
                "transaction_count": r.transaction_count,
                "total_volume_xrp": r.total_volume_xrp,
                "whale_tx_count": r.whale_tx_count,
                "whale_tx_volume": r.whale_tx_volume,
            })

        df = pd.DataFrame(data)
        df = df.groupby("date", as_index=False).last()
        return df

    def _load_etf_data(self, session: Session) -> pd.DataFrame:
        """EtfData 테이블에서 ETF 데이터를 로드한다."""
        rows = (
            session.query(EtfData)
            .order_by(EtfData.timestamp.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()

        data = []
        for r in rows:
            data.append({
                "date": r.timestamp.date(),
                "ticker": r.ticker,
                "premium_discount": r.premium_discount if r.premium_discount is not None else 0.0,
                "volume": r.volume if r.volume is not None else 0,
            })

        return pd.DataFrame(data)

    # ── 병합 헬퍼 ───────────────────────────────────────────

    @staticmethod
    def _merge_on_date(
        base: pd.DataFrame,
        source: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """날짜 기준으로 left join 병합한다. source가 비어있으면 0으로 채운다."""
        if source.empty:
            for col in feature_cols:
                base[col] = 0.0
            return base

        return base.merge(source[["date"] + feature_cols], on="date", how="left")

    def _merge_pair_data(
        self, base: pd.DataFrame, pair_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """페어 데이터를 자산별로 피벗하여 병합한다.

        각 자산에 대해 '{asset}_price'와 '{asset}_corr' 피처를 생성한다.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            병합된 DataFrame과 생성된 피처 이름 리스트.
        """
        feature_names: list[str] = []
        assets = pair_df["asset_name"].unique()

        for asset in sorted(assets):
            asset_data = pair_df[pair_df["asset_name"] == asset].copy()
            # 같은 날짜에 여러 레코드가 있으면 마지막 값 사용
            asset_data = asset_data.groupby("date", as_index=False).last()

            safe_name = asset.replace("/", "_").replace("&", "n")
            price_col = f"{safe_name}_price"
            corr_col = f"{safe_name}_corr"

            asset_data = asset_data.rename(columns={
                "price": price_col,
                "correlation_with_xrp": corr_col,
            })

            base = base.merge(
                asset_data[["date", price_col, corr_col]],
                on="date",
                how="left",
            )

            feature_names.extend([price_col, corr_col])

        return base, feature_names

    def _merge_etf_data(
        self, base: pd.DataFrame, etf_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """ETF 데이터를 티커별 프리미엄/디스카운트 + 총 거래량으로 병합한다.

        각 티커에 대해 'etf_{ticker}_premium' 피처를 생성하고,
        전체 ETF 거래량 합계 'etf_total_volume' 피처를 추가한다.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            병합된 DataFrame과 생성된 피처 이름 리스트.
        """
        feature_names: list[str] = []
        tickers = sorted(etf_df["ticker"].unique())

        # 날짜별 총 거래량 계산
        vol_by_date = etf_df.groupby("date", as_index=False)["volume"].sum()
        vol_by_date = vol_by_date.rename(columns={"volume": "etf_total_volume"})
        base = base.merge(vol_by_date[["date", "etf_total_volume"]], on="date", how="left")
        feature_names.append("etf_total_volume")

        # 티커별 프리미엄/디스카운트 피처
        for ticker in tickers:
            ticker_data = etf_df[etf_df["ticker"] == ticker].copy()
            ticker_data = ticker_data.groupby("date", as_index=False).last()

            col_name = f"etf_{ticker}_premium"
            ticker_data = ticker_data.rename(columns={"premium_discount": col_name})

            base = base.merge(
                ticker_data[["date", col_name]],
                on="date",
                how="left",
            )
            feature_names.append(col_name)

        return base, feature_names
