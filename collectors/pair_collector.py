"""PairCollector / PairDataModule — 상관 자산 가격 데이터를 수집하고 상관계수를 계산한다.

BaseCollector를 상속하며, 암호화폐 페어(바이낸스 API)와
전통 금융 지수(Yahoo Finance)를 수집하여 PairData ORM 모델로 DB에 저장한다.
모든 타임스탬프는 미국 동부시간(America/New_York)으로 변환된다.

PairDataModule은 XRP와 각 자산 간 일별 상관계수를 계산한다.

Requirements: 2.1, 2.2, 2.3, 2.5
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

from collectors.base_collector import BaseCollector
from config import BINANCE_BASE_URL, BINANCE_US_BASE_URL, CRYPTO_PAIRS, INDICES, TIMEZONE
from db.database import get_session
from db.models import PairData

logger = logging.getLogger(__name__)

# 동부시간 타임존
_ET = ZoneInfo(TIMEZONE)

# 암호화폐 페어 → 바이낸스 심볼 매핑
_PAIR_TO_BINANCE_SYMBOL: dict[str, str] = {
    "XRP/BTC": "XRPBTC",
    "XRP/ETH": "XRPETH",
    "XRP/USD": "XRPUSDT",
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
}

# Yahoo Finance 지수 → ticker 매핑
_INDEX_TICKER_MAP: dict[str, str] = {
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}


class PairCollector(BaseCollector):
    """상관 자산 가격 데이터를 수집하는 수집기.

    암호화폐 페어: 바이낸스 klines API (1시간 간격)
    전통 금융 지수: Yahoo Finance (일별 종가)
    """

    source_name = "pair"

    def __init__(
        self,
        base_url: str = BINANCE_BASE_URL,
        pair_to_symbol: dict[str, str] | None = None,
        index_ticker_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self._base_url = base_url
        self._pair_to_symbol = pair_to_symbol or _PAIR_TO_BINANCE_SYMBOL
        self._index_ticker_map = index_ticker_map or _INDEX_TICKER_MAP
        self._initial_collect_done = False

    async def collect(self) -> dict:
        """암호화폐 페어 및 전통 금융 지수 데이터를 수집하고 DB에 저장한다."""
        records: list[dict] = []

        # 1) 암호화폐 페어 수집 (바이낸스)
        crypto_records = await self._collect_crypto_pairs()
        records.extend(crypto_records)

        # 2) 전통 금융 지수 수집 (Yahoo Finance)
        index_records = self._collect_indices()
        records.extend(index_records)

        # 3) DB에 저장
        if records:
            _save_records(records)

        self._initial_collect_done = True
        logger.info("[pair] %d건의 상관 자산 데이터 수집 완료", len(records))

        return {
            "source": self.source_name,
            "count": len(records),
            "records": [
                {
                    "timestamp": r["timestamp"].isoformat(),
                    "asset_name": r["asset_name"],
                    "price": r["price"],
                }
                for r in records
            ],
        }

    async def _collect_crypto_pairs(self) -> list[dict]:
        """바이낸스 klines API에서 암호화폐 페어 가격을 수집한다.

        첫 수집: 1000개 일봉 (약 2.7년치 백필)
        이후: 24개 시간봉 (1일치)
        글로벌 API 451 차단 시 US API로 폴백.
        """
        records: list[dict] = []

        if not self._initial_collect_done:
            interval = "1d"
            limit = 1000
        else:
            interval = "1h"
            limit = 24

        async with httpx.AsyncClient(timeout=60.0) as client:
            for pair_name in CRYPTO_PAIRS:
                symbol = self._pair_to_symbol.get(pair_name)
                if not symbol:
                    logger.warning("[pair] 바이낸스 심볼 매핑 없음: %s", pair_name)
                    continue

                try:
                    klines = await self._fetch_klines(client, symbol, interval, limit)

                    for entry in klines:
                        if not isinstance(entry, list) or len(entry) < 5:
                            continue
                        ts_ms = int(entry[0])
                        close_price = float(entry[4])
                        et_dt = _ms_to_et_datetime(ts_ms)
                        records.append({
                            "timestamp": et_dt,
                            "asset_name": pair_name,
                            "price": close_price,
                        })

                    logger.info("[pair] %s: %d건 수집 (%s, limit=%d)", pair_name, len(klines), interval, limit)

                except Exception:
                    logger.exception("[pair] 암호화폐 페어 수집 실패: %s (%s)", pair_name, symbol)
                    raise

        return records

    async def _fetch_klines(
        self, client: httpx.AsyncClient, symbol: str, interval: str, limit: int,
    ) -> list:
        """바이낸스 글로벌 → US 폴백으로 klines를 가져온다."""
        for base_url in [self._base_url, BINANCE_US_BASE_URL]:
            try:
                resp = await client.get(
                    f"{base_url}/api/v3/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 451:
                    logger.warning("[pair] %s 451 차단, US 폴백 시도", base_url)
                    continue
                raise
        raise RuntimeError(f"바이낸스 글로벌/US 모두 {symbol} 수집 실패")

    def _collect_indices(self) -> list[dict]:
        """Yahoo Finance에서 전통 금융 지수 일별 종가를 수집한다.

        첫 수집: 3년치 (period="3y")
        이후: 5일치 (period="5d")
        """
        records: list[dict] = []
        period = "3y" if not self._initial_collect_done else "5d"

        for index_name in INDICES:
            ticker_symbol = self._index_ticker_map.get(index_name)
            if not ticker_symbol:
                logger.warning("[pair] 티커 매핑 없음: %s", index_name)
                continue

            try:
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period=period)

                if hist.empty:
                    logger.warning("[pair] 데이터 없음: %s (%s)", index_name, ticker_symbol)
                    continue

                for idx_ts, row in hist.iterrows():
                    et_dt = _pandas_ts_to_et_datetime(idx_ts)
                    records.append({
                        "timestamp": et_dt,
                        "asset_name": index_name,
                        "price": float(row["Close"]),
                    })

            except Exception:
                logger.exception("[pair] 지수 수집 실패: %s", index_name)
                raise

        return records


# ── 내부 헬퍼 함수 ───────────────────────────────────────────


def _ms_to_et_datetime(timestamp_ms: int) -> datetime:
    """밀리초 타임스탬프를 미국 동부시간 datetime으로 변환한다."""
    utc_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return utc_dt.astimezone(_ET).replace(tzinfo=None)


def _pandas_ts_to_et_datetime(ts) -> datetime:
    """pandas Timestamp를 미국 동부시간 naive datetime으로 변환한다."""
    if ts.tzinfo is not None:
        et_dt = ts.astimezone(_ET)
    else:
        et_dt = ts.tz_localize("UTC").astimezone(_ET)
    return et_dt.to_pydatetime().replace(tzinfo=None)


def _save_records(records: list[dict]) -> None:
    """PairData 레코드를 DB에 저장한다. 중복 timestamp+asset_name은 건너뛴다."""
    session = get_session()
    try:
        for rec in records:
            existing = (
                session.query(PairData)
                .filter(
                    PairData.timestamp == rec["timestamp"],
                    PairData.asset_name == rec["asset_name"],
                )
                .first()
            )
            if existing:
                continue

            pair_data = PairData(
                timestamp=rec["timestamp"],
                asset_name=rec["asset_name"],
                price=rec["price"],
            )
            session.add(pair_data)

        session.commit()
    except Exception:
        session.rollback()
        logger.exception("[pair] DB 저장 실패")
        raise
    finally:
        session.close()


class PairDataModule:
    """XRP와 각 자산 간 일별 상관계수를 계산하는 모듈."""

    def calculate_correlation(
        self, xrp_df: pd.DataFrame, pair_df: pd.DataFrame
    ) -> float:
        """XRP와 특정 자산 간 일별 상관계수를 계산한다."""
        if xrp_df.empty or pair_df.empty:
            return 0.0

        xrp_close = xrp_df["close"].values.astype(float)
        pair_close = pair_df["close"].values.astype(float)

        min_len = min(len(xrp_close), len(pair_close))
        if min_len < 2:
            return 0.0

        xrp_close = xrp_close[:min_len]
        pair_close = pair_close[:min_len]

        if np.std(xrp_close) == 0 or np.std(pair_close) == 0:
            if np.array_equal(xrp_close, pair_close):
                return 1.0
            return 0.0

        corr_matrix = np.corrcoef(xrp_close, pair_close)
        corr = float(corr_matrix[0, 1])

        if np.isnan(corr):
            return 0.0

        return max(-1.0, min(1.0, corr))
