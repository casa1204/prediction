"""PriceCollector — 바이낸스 API에서 XRP OHLCV 시간봉 데이터를 수집한다.

BaseCollector를 상속하며, 수집된 데이터를 PriceData ORM 모델로 변환하여 DB에 저장한다.
모든 타임스탬프는 미국 동부시간(America/New_York)으로 변환된다.

첫 수집 시 1000개 시간봉(약 42일), 이후 반복 수집은 최근 24개(1일치)만 가져온다.

Requirements: 1.1, 1.2
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from collectors.base_collector import BaseCollector
from config import BINANCE_BASE_URL, BINANCE_US_BASE_URL, TIMEZONE
from db.database import get_session
from db.models import DailyPriceData, PriceData

logger = logging.getLogger(__name__)

# 동부시간 타임존
_ET = ZoneInfo(TIMEZONE)


class PriceCollector(BaseCollector):
    """바이낸스 API에서 XRP/USDT 시간봉 OHLCV 데이터를 수집하는 수집기.

    바이낸스 ``/api/v3/klines`` 엔드포인트를 사용한다.
    첫 수집 시 1000개 시간봉(약 42일), 이후 24개(1일치)만 가져온다.
    """

    source_name = "price"

    def __init__(self, base_url: str = BINANCE_BASE_URL):
        super().__init__()
        self._base_url = base_url
        self._initial_collect_done = False

    async def collect(self) -> dict:
        """바이낸스 API에서 시간봉 + 일봉 데이터를 수집하고 DB에 저장한다."""
        # 시간봉: 첫 수집 1000개, 이후 24개
        h_limit = 1000 if not self._initial_collect_done else 24
        # 일봉: 첫 수집 1000개(약 2.7년), 이후 7개
        d_limit = 1000 if not self._initial_collect_done else 7

        async with httpx.AsyncClient(timeout=60.0) as client:
            # 시간봉 수집 (글로벌 → US 폴백)
            h_resp = await self._fetch_klines(client, "XRPUSDT", "1h", h_limit)
            # 일봉 수집
            d_resp = await self._fetch_klines(client, "XRPUSDT", "1d", d_limit)

        h_records = _parse_klines(h_resp)
        d_records = _parse_klines(d_resp)

        if h_records:
            _save_records(h_records, PriceData)
        if d_records:
            _save_records(d_records, DailyPriceData)

        self._initial_collect_done = True
        logger.info(
            "[price] 시간봉 %d건 + 일봉 %d건 수집 완료",
            len(h_records), len(d_records),
        )

        return {
            "source": self.source_name,
            "hourly_count": len(h_records),
            "daily_count": len(d_records),
        }

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
                logger.info("[price] %s에서 %s %s %d건 수집", base_url, symbol, interval, limit)
                return resp.json()
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code == 451:
                    logger.warning("[price] %s 451 차단, US 폴백 시도", base_url)
                    continue
                if code == 400:
                    logger.warning("[price] %s에서 %s 없음 (400), 건너뜀", base_url, symbol)
                    return []
                raise
        logger.warning("[price] 글로벌/US 모두 %s 수집 실패", symbol)
        return []


# ── 내부 헬퍼 함수 ───────────────────────────────────────────


def _ms_to_et_datetime(timestamp_ms: int) -> datetime:
    """밀리초 타임스탬프를 미국 동부시간 datetime으로 변환한다."""
    utc_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return utc_dt.astimezone(_ET).replace(tzinfo=None)


def _parse_klines(klines: list) -> list[dict]:
    """바이낸스 klines 응답을 파싱하여 레코드 리스트로 변환한다.

    바이낸스 klines 형식:
    [[open_time, open, high, low, close, volume, close_time, ...], ...]
    """
    records = []
    for entry in klines:
        if not isinstance(entry, list) or len(entry) < 6:
            continue

        ts_ms = int(entry[0])
        records.append({
            "timestamp": _ms_to_et_datetime(ts_ms),
            "open": float(entry[1]),
            "high": float(entry[2]),
            "low": float(entry[3]),
            "close": float(entry[4]),
            "volume": float(entry[5]),
        })

    return records


def _save_records(records: list[dict], model_class=PriceData) -> None:
    """OHLCV 레코드를 DB에 저장한다. 중복 타임스탬프는 건너뛴다."""
    session = get_session()
    try:
        for rec in records:
            existing = (
                session.query(model_class)
                .filter(model_class.timestamp == rec["timestamp"])
                .first()
            )
            if existing:
                continue

            row = model_class(
                timestamp=rec["timestamp"],
                open=rec["open"],
                high=rec["high"],
                low=rec["low"],
                close=rec["close"],
                volume=rec["volume"],
            )
            session.add(row)

        session.commit()
    except Exception:
        session.rollback()
        logger.exception("[price] DB 저장 실패")
        raise
    finally:
        session.close()
