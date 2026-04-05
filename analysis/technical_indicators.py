"""기술 지표 계산 모듈 — ta 라이브러리를 활용한 RSI, MACD, BB, SMA, EMA 계산."""

import logging
from datetime import datetime

import pandas as pd
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
from sqlalchemy.orm import Session

from db.models import TechnicalIndicator

logger = logging.getLogger(__name__)


class TechnicalIndicatorModule:
    """XRP 가격 데이터로부터 기술 지표를 계산하고 DB에 저장한다."""

    def interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측값 보간 처리 및 로그 기록.

        선형 보간법(linear interpolation)으로 NaN을 채우고,
        보간된 컬럼과 건수를 로깅한다.
        """
        if df.empty:
            return df

        result = df.copy()
        for col in result.columns:
            missing_count = result[col].isna().sum()
            if missing_count > 0:
                logger.info(
                    "컬럼 '%s'에서 %d개의 결측값을 선형 보간 처리합니다.", col, missing_count
                )
                result[col] = result[col].interpolate(method="linear")
                # 시작 부분의 NaN은 forward fill 불가 → backward fill 처리
                result[col] = result[col].bfill()
                remaining = result[col].isna().sum()
                if remaining > 0:
                    result[col] = result[col].ffill()
                    logger.warning(
                        "컬럼 '%s'에서 보간 후에도 %d개의 결측값이 남아 ffill 처리했습니다.",
                        col,
                        remaining,
                    )
        return result

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI(14), MACD(12,26,9), BB(20), SMA(5,10,20,50,200), EMA(12,26) 계산.

        Parameters
        ----------
        df : pd.DataFrame
            timestamp, open, high, low, close, volume 컬럼을 포함하는 OHLCV 데이터.

        Returns
        -------
        pd.DataFrame
            원본 타임스탬프와 계산된 모든 기술 지표 컬럼을 포함하는 DataFrame.
        """
        if df.empty:
            return df

        result = pd.DataFrame()
        result["timestamp"] = df["timestamp"]

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI (14)
        rsi_indicator = ta.momentum.RSIIndicator(close=close, window=14)
        result["rsi_14"] = rsi_indicator.rsi()

        # MACD (12, 26, 9)
        macd_indicator = ta.trend.MACD(
            close=close, window_slow=26, window_fast=12, window_sign=9
        )
        result["macd"] = macd_indicator.macd()
        result["macd_signal"] = macd_indicator.macd_signal()
        result["macd_histogram"] = macd_indicator.macd_diff()

        # Bollinger Bands (20)
        bb_indicator = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        result["bb_upper"] = bb_indicator.bollinger_hband()
        result["bb_middle"] = bb_indicator.bollinger_mavg()
        result["bb_lower"] = bb_indicator.bollinger_lband()

        # SMA (5, 10, 20, 50, 200)
        for period in [5, 10, 20, 50, 200]:
            sma = ta.trend.SMAIndicator(close=close, window=period)
            result[f"sma_{period}"] = sma.sma_indicator()

        # EMA (12, 26)
        for period in [12, 26]:
            ema = ta.trend.EMAIndicator(close=close, window=period)
            result[f"ema_{period}"] = ema.ema_indicator()

        # OBV
        obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
        result["obv"] = obv_indicator.on_balance_volume()

        # SMA 100
        sma_100 = ta.trend.SMAIndicator(close=close, window=100)
        result["sma_100"] = sma_100.sma_indicator()

        # 이격도 (Disparity): (close - sma) / sma * 100
        result["disparity_50"] = (close - result["sma_50"]) / result["sma_50"] * 100
        result["disparity_100"] = (close - result["sma_100"]) / result["sma_100"] * 100
        result["disparity_200"] = (close - result["sma_200"]) / result["sma_200"] * 100

        # 결측값 보간 (timestamp 제외)
        indicator_cols = [c for c in result.columns if c != "timestamp"]
        result[indicator_cols] = self.interpolate_missing(result[indicator_cols])

        return result

    def save_to_db(self, df: pd.DataFrame, session: Session) -> int:
        """계산된 지표를 TechnicalIndicator 테이블에 저장한다.

        이미 존재하는 타임스탬프는 건너뛴다.

        Parameters
        ----------
        df : pd.DataFrame
            calculate_all()이 반환한 DataFrame.
        session : Session
            SQLAlchemy 세션.

        Returns
        -------
        int
            새로 저장된 레코드 수.
        """
        if df.empty:
            return 0

        # 기존 타임스탬프 조회
        existing_timestamps = {
            row.timestamp
            for row in session.query(TechnicalIndicator.timestamp).all()
        }

        saved_count = 0
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts in existing_timestamps:
                logger.debug("타임스탬프 %s는 이미 존재하므로 건너뜁니다.", ts)
                continue

            record = TechnicalIndicator(
                timestamp=ts,
                rsi_14=_safe_float(row.get("rsi_14")),
                macd=_safe_float(row.get("macd")),
                macd_signal=_safe_float(row.get("macd_signal")),
                macd_histogram=_safe_float(row.get("macd_histogram")),
                bb_upper=_safe_float(row.get("bb_upper")),
                bb_middle=_safe_float(row.get("bb_middle")),
                bb_lower=_safe_float(row.get("bb_lower")),
                sma_5=_safe_float(row.get("sma_5")),
                sma_10=_safe_float(row.get("sma_10")),
                sma_20=_safe_float(row.get("sma_20")),
                sma_50=_safe_float(row.get("sma_50")),
                sma_200=_safe_float(row.get("sma_200")),
                ema_12=_safe_float(row.get("ema_12")),
                ema_26=_safe_float(row.get("ema_26")),
                obv=_safe_float(row.get("obv")),
                sma_100=_safe_float(row.get("sma_100")),
                disparity_50=_safe_float(row.get("disparity_50")),
                disparity_100=_safe_float(row.get("disparity_100")),
                disparity_200=_safe_float(row.get("disparity_200")),
            )
            session.add(record)
            saved_count += 1

        if saved_count > 0:
            session.commit()
            logger.info("%d개의 기술 지표 레코드를 저장했습니다.", saved_count)

        return saved_count


def _safe_float(value) -> float | None:
    """NaN/None을 None으로, 그 외는 float로 변환."""
    if value is None:
        return None
    try:
        import math
        f = float(value)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None
