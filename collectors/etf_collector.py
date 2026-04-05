"""EtfCollector — XRP 현물 ETF 가격/거래량 수집 및 프리미엄/디스카운트 계산.

BaseCollector를 상속하며, yfinance로 각 ETF의 최근 5일 데이터를 수집하고
XRP 현물 가격과 비교하여 프리미엄/디스카운트를 계산한다.

2x 레버리지 ETF(XXRP, XRPT)는 가격/2로 비교한다.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import yfinance as yf

from collectors.base_collector import BaseCollector
from config import TIMEZONE, XRP_ETF_LEVERAGED, XRP_ETF_TICKERS
from db.database import get_session
from db.models import EtfData, PriceData

logger = logging.getLogger(__name__)

_ET = ZoneInfo(TIMEZONE)


class EtfCollector(BaseCollector):
    """XRP 현물 ETF 데이터를 수집하는 수집기."""

    source_name = "etf"

    async def collect(self) -> dict:
        """각 ETF의 최근 5일 가격/거래량을 수집하고 DB에 저장한다."""
        # XRP 현물 가격 조회 (프리미엄 계산용)
        xrp_spot_price = self._get_xrp_spot_price()

        records: list[dict] = []

        for ticker, name in XRP_ETF_TICKERS.items():
            try:
                etf = yf.Ticker(ticker)
                hist = etf.history(period="5d")

                if hist.empty:
                    logger.warning("[etf] 데이터 없음: %s", ticker)
                    continue

                for idx_ts, row in hist.iterrows():
                    et_dt = _pandas_ts_to_et_datetime(idx_ts)
                    price = float(row["Close"])
                    volume = int(row["Volume"]) if row["Volume"] else 0

                    # 프리미엄/디스카운트 계산
                    premium_discount = None
                    if xrp_spot_price and xrp_spot_price > 0:
                        compare_price = price / 2 if ticker in XRP_ETF_LEVERAGED else price
                        premium_discount = round(
                            ((compare_price / xrp_spot_price) - 1) * 100, 4
                        )

                    records.append({
                        "timestamp": et_dt,
                        "ticker": ticker,
                        "name": name,
                        "price": price,
                        "volume": volume,
                        "premium_discount": premium_discount,
                    })

            except Exception:
                logger.exception("[etf] ETF 수집 실패: %s", ticker)
                raise

        if records:
            _save_records(records)

        logger.info("[etf] %d건의 ETF 데이터 수집 완료", len(records))

        return {
            "source": self.source_name,
            "count": len(records),
            "records": [
                {
                    "timestamp": r["timestamp"].isoformat(),
                    "ticker": r["ticker"],
                    "price": r["price"],
                    "volume": r["volume"],
                    "premium_discount": r["premium_discount"],
                }
                for r in records
            ],
        }

    def _get_xrp_spot_price(self) -> float | None:
        """DB에서 XRP 최신 종가를 조회한다."""
        session = get_session()
        try:
            from sqlalchemy import desc

            latest = (
                session.query(PriceData)
                .order_by(desc(PriceData.timestamp))
                .first()
            )
            return latest.close if latest else None
        finally:
            session.close()


def _pandas_ts_to_et_datetime(ts) -> datetime:
    """pandas Timestamp를 미국 동부시간 naive datetime으로 변환한다."""
    if ts.tzinfo is not None:
        et_dt = ts.astimezone(_ET)
    else:
        et_dt = ts.tz_localize("UTC").astimezone(_ET)
    return et_dt.to_pydatetime().replace(tzinfo=None)


def _save_records(records: list[dict]) -> None:
    """EtfData 레코드를 DB에 저장한다. 중복 timestamp+ticker는 건너뛴다."""
    session = get_session()
    try:
        for rec in records:
            existing = (
                session.query(EtfData)
                .filter(
                    EtfData.timestamp == rec["timestamp"],
                    EtfData.ticker == rec["ticker"],
                )
                .first()
            )
            if existing:
                continue

            etf_data = EtfData(
                timestamp=rec["timestamp"],
                ticker=rec["ticker"],
                name=rec["name"],
                price=rec["price"],
                volume=rec["volume"],
                premium_discount=rec["premium_discount"],
            )
            session.add(etf_data)

        session.commit()
    except Exception:
        session.rollback()
        logger.exception("[etf] DB 저장 실패")
        raise
    finally:
        session.close()
