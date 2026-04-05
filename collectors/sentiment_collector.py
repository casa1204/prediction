"""SentimentCollector / SentimentModule — 시장 심리 데이터를 수집하고 정규화한다.

BaseCollector를 상속하며, Google Trends(pytrends), CryptoCompare Social Stats,
공포탐욕지수(alternative.me API)를 수집하여 SentimentData ORM 모델로 DB에 저장한다.
모든 타임스탬프는 미국 동부시간(America/New_York)으로 변환된다.

Google Trends 키워드 3그룹:
- 그룹1 (기본): "XRP", "리플", "Ripple"
- 그룹2 (거시경제): "Fed Interest Rate", "Inflation", "CPI", "Recession", "Rate Cut"
- 그룹3 (ETF/규제): "XRP ETF", "Spot ETF", "RLUSD", "SEC", "Crypto Regulation"
- 그룹4 (심리): "Buy Crypto", "Sell Crypto", "Bitcoin", "Crypto Scam"

파생 피처:
- attention_ratio: XRP trend / Bitcoin trend
- fomo_spread: "Buy Crypto" - "Sell Crypto"
- macro_aggregate: 거시경제 키워드 가중 평균

SentimentModule은 수집된 심리 데이터를 0~100 범위로 정규화한다.

Requirements: 3.1, 3.2, 3.3, 3.5
"""

import logging
import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
from pytrends.request import TrendReq

from collectors.base_collector import BaseCollector
from config import (
    TIMEZONE,
    LUNARCRUSH_API_KEY,
)
from db.database import get_session
from db.models import SentimentData

logger = logging.getLogger(__name__)

# 동부시간 타임존
_ET = ZoneInfo(TIMEZONE)

# LunarCrush API URL
_LUNARCRUSH_BASE_URL = "https://lunarcrush.com/api4/public/coins/xrp/v1"

# 공포탐욕지수 API URL
_FEAR_GREED_API_URL = "https://api.alternative.me/fng/"

# ── Google Trends 키워드 그룹 ────────────────────────────────
_TREND_GROUP_BASE = ["XRP", "리플", "Ripple"]
_TREND_GROUP_MACRO = ["Fed Interest Rate", "Inflation", "CPI", "Recession", "Rate Cut"]
_TREND_GROUP_ETF_REG = ["XRP ETF", "Spot ETF", "RLUSD", "SEC", "Crypto Regulation"]
_TREND_GROUP_SENTIMENT = ["Buy Crypto", "Sell Crypto", "Bitcoin", "Crypto Scam"]

# 그룹 간 호출 딜레이 (초)
_TREND_GROUP_DELAY = 5


class SentimentCollector(BaseCollector):
    """시장 심리 데이터를 수집하는 수집기.

    Google Trends: pytrends 라이브러리로 4개 키워드 그룹 트렌드 수집
    LunarCrush: XRP 소셜 감성 데이터 수집
    공포탐욕지수: alternative.me API에서 수집
    """

    source_name = "sentiment"

    def __init__(
        self,
        fear_greed_url: str = _FEAR_GREED_API_URL,
        trend_keywords: list[str] | None = None,
    ):
        super().__init__()
        self._fear_greed_url = fear_greed_url
        self._trend_keywords = trend_keywords or _TREND_GROUP_BASE

    async def collect(self) -> dict:
        """Google Trends 4그룹, LunarCrush, 공포탐욕지수 데이터를 수집하고 DB에 저장한다.

        Returns:
            수집된 심리 데이터 정보를 포함하는 dict.
        """
        now_et = datetime.now(timezone.utc).astimezone(_ET).replace(tzinfo=None)
        # 날짜 단위로 정규화 (일별 수집)
        timestamp = now_et.replace(hour=0, minute=0, second=0, microsecond=0)

        # 1) Google Trends 4그룹 순차 수집 (그룹 간 5초 딜레이)
        google_trend_score, trend_details = await self._collect_all_trend_groups()

        # 2) CryptoCompare Social Stats 수집
        sns_mention_score, sns_sentiment_score = await self._collect_social_stats()

        # 3) 공포탐욕지수 수집
        fear_greed_index = await self._collect_fear_greed()

        # 4) 파생 피처 계산
        attention_ratio = self._calc_attention_ratio(trend_details)
        fomo_spread = self._calc_fomo_spread(trend_details)
        macro_aggregate = trend_details.get("macro_score")

        record = {
            "timestamp": timestamp,
            "google_trend_score": google_trend_score,
            "sns_mention_score": sns_mention_score,
            "sns_sentiment_score": sns_sentiment_score,
            "fear_greed_index": fear_greed_index,
            "trend_macro_score": trend_details.get("macro_score"),
            "trend_etf_regulatory_score": trend_details.get("etf_reg_score"),
            "trend_sentiment_score": trend_details.get("sentiment_score"),
            "attention_ratio": attention_ratio,
            "fomo_spread": fomo_spread,
            "macro_aggregate": macro_aggregate,
        }

        # DB에 저장
        _save_record(record)

        logger.info(
            "[sentiment] 심리 데이터 수집 완료 — "
            "google=%.1f, macro=%.1f, etf_reg=%.1f, sent=%.1f, "
            "attn_ratio=%.2f, fomo=%.1f, fgi=%.1f",
            google_trend_score or 0,
            trend_details.get("macro_score") or 0,
            trend_details.get("etf_reg_score") or 0,
            trend_details.get("sentiment_score") or 0,
            attention_ratio or 0,
            fomo_spread or 0,
            fear_greed_index or 0,
        )

        return {
            "source": self.source_name,
            "timestamp": timestamp.isoformat(),
            "google_trend_score": google_trend_score,
            "sns_mention_score": sns_mention_score,
            "sns_sentiment_score": sns_sentiment_score,
            "fear_greed_index": fear_greed_index,
            "trend_macro_score": trend_details.get("macro_score"),
            "trend_etf_regulatory_score": trend_details.get("etf_reg_score"),
            "trend_sentiment_score": trend_details.get("sentiment_score"),
            "attention_ratio": attention_ratio,
            "fomo_spread": fomo_spread,
            "macro_aggregate": macro_aggregate,
        }

    # ── Google Trends ────────────────────────────────────────

    async def _collect_all_trend_groups(self) -> tuple[float | None, dict]:
        """4개 키워드 그룹을 순차 호출하여 트렌드 점수를 수집한다.

        그룹 간 _TREND_GROUP_DELAY초 딜레이를 두어 429 방지.
        asyncio.sleep을 사용하여 이벤트 루프를 블로킹하지 않는다.

        Returns:
            (기본 XRP 트렌드 점수, 상세 점수 dict)
        """
        details: dict = {
            "macro_score": None,
            "etf_reg_score": None,
            "sentiment_score": None,
            "keyword_values": {},  # 개별 키워드 값 (파생 피처 계산용)
        }

        # 그룹1: 기본 XRP 키워드
        base_score, base_vals = self._collect_trend_group(self._trend_keywords, "base")

        # 그룹2: 거시경제
        await asyncio.sleep(_TREND_GROUP_DELAY)
        macro_score, macro_vals = self._collect_trend_group(_TREND_GROUP_MACRO, "macro")
        details["macro_score"] = macro_score

        # 그룹3: ETF/규제
        await asyncio.sleep(_TREND_GROUP_DELAY)
        etf_score, etf_vals = self._collect_trend_group(_TREND_GROUP_ETF_REG, "etf_reg")
        details["etf_reg_score"] = etf_score

        # 그룹4: 심리
        await asyncio.sleep(_TREND_GROUP_DELAY)
        sent_score, sent_vals = self._collect_trend_group(_TREND_GROUP_SENTIMENT, "sentiment")
        details["sentiment_score"] = sent_score

        # 개별 키워드 값 병합 (파생 피처 계산용)
        for vals in [base_vals, macro_vals, etf_vals, sent_vals]:
            details["keyword_values"].update(vals)

        return base_score, details

    def _collect_trend_group(
        self, keywords: list[str], group_name: str,
    ) -> tuple[float | None, dict[str, float]]:
        """단일 키워드 그룹의 Google Trends 점수를 수집한다.

        Args:
            keywords: 최대 5개 키워드 리스트.
            group_name: 로깅용 그룹 이름.

        Returns:
            (그룹 평균 점수 0~100, 개별 키워드별 최신 값 dict)
        """
        keyword_values: dict[str, float] = {}
        try:
            pytrends = TrendReq(hl="en-US", tz=300)
            pytrends.build_payload(
                keywords,
                cat=0,
                timeframe="now 7-d",
                geo="",
            )
            interest_df = pytrends.interest_over_time()

            if interest_df.empty:
                logger.warning("[sentiment] Google Trends '%s' 데이터 없음", group_name)
                return None, keyword_values

            keyword_cols = [c for c in interest_df.columns if c != "isPartial"]
            if not keyword_cols:
                return None, keyword_values

            latest_row = interest_df[keyword_cols].iloc[-1]
            for kw in keyword_cols:
                keyword_values[kw] = float(latest_row[kw])

            score = float(latest_row.mean())
            score = max(0.0, min(100.0, score))

            logger.info(
                "[sentiment] Google Trends '%s': %.1f (%s)",
                group_name, score,
                ", ".join(f"{k}={v:.0f}" for k, v in keyword_values.items()),
            )
            return score, keyword_values

        except Exception:
            logger.exception("[sentiment] Google Trends '%s' 수집 실패", group_name)
            return None, keyword_values

    def _collect_google_trends(self) -> float | None:
        """pytrends로 기본 Google Trends 검색 트렌드를 수집한다 (하위 호환용)."""
        score, _ = self._collect_trend_group(self._trend_keywords, "base")
        return score

    # ── 파생 피처 계산 ───────────────────────────────────────

    @staticmethod
    def _calc_attention_ratio(details: dict) -> float | None:
        """Attention Ratio = XRP trend / Bitcoin trend.

        XRP에 대한 관심이 Bitcoin 대비 얼마나 높은지 비율로 나타낸다.
        """
        kv = details.get("keyword_values", {})
        xrp_val = kv.get("XRP")
        btc_val = kv.get("Bitcoin")
        if xrp_val is None or btc_val is None:
            return None
        if btc_val == 0:
            return 100.0 if xrp_val > 0 else 0.0
        return round(xrp_val / btc_val, 4)

    @staticmethod
    def _calc_fomo_spread(details: dict) -> float | None:
        """FOMO Spread = "Buy Crypto" trend - "Sell Crypto" trend.

        양수면 매수 심리 우세, 음수면 매도 심리 우세.
        """
        kv = details.get("keyword_values", {})
        buy_val = kv.get("Buy Crypto")
        sell_val = kv.get("Sell Crypto")
        if buy_val is None or sell_val is None:
            return None
        return round(buy_val - sell_val, 2)

    # ── LunarCrush Social Stats ──────────────────────────────

    async def _collect_social_stats(self) -> tuple[float | None, float | None]:
        """LunarCrush API에서 XRP 소셜 감성 데이터를 수집한다.

        Returns:
            (sns_mention_score, sns_sentiment_score) 튜플.
            각각 0~100 범위로 정규화된 값.
        """
        if not LUNARCRUSH_API_KEY:
            logger.warning("[sentiment] LunarCrush API 키 미설정, 건너뜀")
            return None, None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    _LUNARCRUSH_BASE_URL,
                    headers={"Authorization": f"Bearer {LUNARCRUSH_API_KEY}"},
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})

            # LunarCrush 주요 지표
            galaxy_score = data.get("galaxy_score", 0)        # 0~100 종합 소셜 점수
            alt_rank = data.get("alt_rank", 0)                # 순위 (낮을수록 좋음)
            social_volume = data.get("social_volume", 0)      # 소셜 언급량
            social_score = data.get("social_score", 0)        # 소셜 점수
            sentiment = data.get("sentiment", 0)              # 감성 점수 (0~100)

            # 언급량 점수: galaxy_score 사용 (이미 0~100)
            mention_score = min(100.0, max(0.0, float(galaxy_score)))

            # 감성 점수: sentiment 사용 (이미 0~100 범위)
            sentiment_score = min(100.0, max(0.0, float(sentiment) if sentiment else 50.0))

            logger.info(
                "[sentiment] LunarCrush: galaxy=%s, sentiment=%s, social_vol=%s, alt_rank=%s",
                galaxy_score, sentiment, social_volume, alt_rank,
            )

            return mention_score, sentiment_score

        except Exception:
            logger.exception("[sentiment] LunarCrush 수집 실패")
            return None, None

    # ── 공포탐욕지수 ─────────────────────────────────────────

    async def _collect_fear_greed(self) -> float | None:
        """alternative.me API에서 암호화폐 공포탐욕지수를 수집한다.

        Returns:
            0~100 범위의 공포탐욕지수 값.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    self._fear_greed_url,
                    params={"limit": "1", "format": "json"},
                )
                resp.raise_for_status()
                data = resp.json()

            fng_data = data.get("data", [])
            if not fng_data:
                logger.warning("[sentiment] 공포탐욕지수 데이터 없음")
                return None

            value = float(fng_data[0].get("value", 0))
            return max(0.0, min(100.0, value))

        except Exception:
            logger.exception("[sentiment] 공포탐욕지수 수집 실패")
            return None


# ── 내부 헬퍼 함수 ───────────────────────────────────────────


def _save_record(record: dict) -> None:
    """SentimentData 레코드를 DB에 저장한다. 중복 타임스탬프는 업데이트한다."""
    session = get_session()
    try:
        existing = (
            session.query(SentimentData)
            .filter(SentimentData.timestamp == record["timestamp"])
            .first()
        )
        if existing:
            # 기존 레코드 업데이트 (확장 필드 추가 반영)
            for key, value in record.items():
                if key != "timestamp" and value is not None:
                    setattr(existing, key, value)
            session.commit()
            logger.info("[sentiment] 기존 레코드 업데이트: %s", record["timestamp"])
            return

        sentiment_data = SentimentData(
            timestamp=record["timestamp"],
            google_trend_score=record["google_trend_score"],
            sns_mention_score=record["sns_mention_score"],
            sns_sentiment_score=record["sns_sentiment_score"],
            fear_greed_index=record["fear_greed_index"],
            trend_macro_score=record.get("trend_macro_score"),
            trend_etf_regulatory_score=record.get("trend_etf_regulatory_score"),
            trend_sentiment_score=record.get("trend_sentiment_score"),
            attention_ratio=record.get("attention_ratio"),
            fomo_spread=record.get("fomo_spread"),
            macro_aggregate=record.get("macro_aggregate"),
        )
        session.add(sentiment_data)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("[sentiment] DB 저장 실패")
        raise
    finally:
        session.close()


# ── 소스별 정규화 범위 설정 ──────────────────────────────────
# 각 소스의 원본 데이터 범위를 정의하여 0~100으로 매핑한다.
_SOURCE_RANGES: dict[str, tuple[float, float]] = {
    "google_trends": (0.0, 100.0),      # Google Trends: 이미 0~100
    "twitter_mention": (0.0, 1000.0),    # 트윗 수 기준 (최대 1000건)
    "twitter_sentiment": (-1.0, 1.0),    # 감성 점수: -1(부정) ~ 1(긍정)
    "fear_greed": (0.0, 100.0),          # 공포탐욕지수: 이미 0~100
}


class SentimentModule:
    """시장 심리 데이터를 0~100 범위로 정규화하는 모듈.

    Requirements: 3.5
    """

    def __init__(
        self,
        source_ranges: dict[str, tuple[float, float]] | None = None,
    ):
        self._source_ranges = source_ranges or _SOURCE_RANGES

    def normalize(self, value: float, source: str) -> float:
        """수집된 심리 데이터를 0~100 범위로 정규화한다.

        소스별로 정의된 원본 범위를 기준으로 min-max 정규화를 수행한다.
        알 수 없는 소스의 경우 값을 0~100으로 클램핑한다.

        Args:
            value: 원본 데이터 값.
            source: 데이터 소스 이름 ("google_trends", "twitter_mention",
                    "twitter_sentiment", "fear_greed" 등).

        Returns:
            0.0 ~ 100.0 범위로 정규화된 값.
        """
        if source in self._source_ranges:
            src_min, src_max = self._source_ranges[source]
            if src_max == src_min:
                # 범위가 0이면 중간값 반환
                return 50.0
            # min-max 정규화: (value - min) / (max - min) * 100
            normalized = (value - src_min) / (src_max - src_min) * 100.0
        else:
            # 알 수 없는 소스: 값을 그대로 사용
            normalized = value

        # 항상 0~100 범위로 클램핑
        return max(0.0, min(100.0, normalized))
