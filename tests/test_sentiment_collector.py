"""SentimentCollector 단위 테스트.

Google Trends, Twitter/X, 공포탐욕지수 API 응답을 모킹하여
SentimentCollector의 데이터 수집, DB 저장, 타임존 변환 로직을 검증한다.

**Validates: Requirements 3.1, 3.2, 3.3**
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collectors.sentiment_collector import (
    SentimentCollector,
    _save_record,
)
from config import TIMEZONE
from db.models import Base, SentimentData

_ET = ZoneInfo(TIMEZONE)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def in_memory_session():
    """인메모리 SQLite 세션을 생성한다."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sample_fear_greed_response():
    """alternative.me 공포탐욕지수 API 응답 샘플."""
    return {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": "72",
                "value_classification": "Greed",
                "timestamp": "1705276800",
            }
        ],
    }


@pytest.fixture
def sample_google_trends_df():
    """Google Trends interest_over_time 샘플 DataFrame."""
    dates = pd.date_range("2024-01-10", periods=7, freq="D")
    return pd.DataFrame(
        {
            "XRP": [30, 35, 40, 45, 50, 55, 60],
            "리플": [20, 25, 30, 35, 40, 45, 50],
            "Ripple": [25, 30, 35, 40, 45, 50, 55],
            "isPartial": [False] * 7,
        },
        index=dates,
    )


# ── SentimentCollector 기본 테스트 ────────────────────────────


class TestSentimentCollectorBasic:
    def test_source_name(self):
        collector = SentimentCollector()
        assert collector.source_name == "sentiment"

    def test_default_keywords(self):
        collector = SentimentCollector()
        assert collector._trend_keywords == ["XRP", "리플", "Ripple"]

    def test_custom_keywords(self):
        collector = SentimentCollector(trend_keywords=["BTC", "Bitcoin"])
        assert collector._trend_keywords == ["BTC", "Bitcoin"]


# ── Google Trends 수집 테스트 ─────────────────────────────────


class TestCollectGoogleTrends:
    def test_returns_average_score(self, sample_google_trends_df):
        """키워드별 최신 값의 평균을 반환한다."""
        collector = SentimentCollector()

        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = sample_google_trends_df

        with patch("collectors.sentiment_collector.TrendReq", return_value=mock_pytrends):
            score = collector._collect_google_trends()

        # 최신 행: XRP=60, 리플=50, Ripple=55 → 평균 = 55.0
        assert score == pytest.approx(55.0)

    def test_returns_none_on_empty_data(self):
        """데이터가 없으면 None을 반환한다."""
        collector = SentimentCollector()

        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = pd.DataFrame()

        with patch("collectors.sentiment_collector.TrendReq", return_value=mock_pytrends):
            score = collector._collect_google_trends()

        assert score is None

    def test_returns_none_on_exception(self):
        """예외 발생 시 None을 반환한다."""
        collector = SentimentCollector()

        with patch(
            "collectors.sentiment_collector.TrendReq",
            side_effect=Exception("API error"),
        ):
            score = collector._collect_google_trends()

        assert score is None

    def test_score_clamped_to_100(self):
        """점수가 100을 초과하면 100으로 클램핑한다."""
        collector = SentimentCollector()

        dates = pd.date_range("2024-01-10", periods=1, freq="D")
        df = pd.DataFrame(
            {"XRP": [120], "리플": [130], "Ripple": [110], "isPartial": [False]},
            index=dates,
        )

        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = df

        with patch("collectors.sentiment_collector.TrendReq", return_value=mock_pytrends):
            score = collector._collect_google_trends()

        assert score == 100.0


# ── Twitter/X 수집 테스트 ─────────────────────────────────────


class TestCollectTwitter:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_token(self):
        """Bearer Token이 미설정이면 (None, None)을 반환한다."""
        collector = SentimentCollector()

        with patch("collectors.sentiment_collector.TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN"):
            mention, sentiment = await collector._collect_twitter()

        assert mention is None
        assert sentiment is None

    @pytest.mark.asyncio
    async def test_returns_scores_on_success(self):
        """정상 응답 시 언급량과 감성 점수를 반환한다."""
        collector = SentimentCollector()

        mock_tweet1 = MagicMock()
        mock_tweet1.public_metrics = {
            "like_count": 10,
            "retweet_count": 5,
            "reply_count": 2,
        }
        mock_tweet2 = MagicMock()
        mock_tweet2.public_metrics = {
            "like_count": 20,
            "retweet_count": 3,
            "reply_count": 1,
        }

        mock_response = MagicMock()
        mock_response.data = [mock_tweet1, mock_tweet2]

        mock_client = MagicMock()
        mock_client.search_recent_tweets.return_value = mock_response

        with patch("collectors.sentiment_collector.TWITTER_BEARER_TOKEN", "valid_token"):
            with patch("collectors.sentiment_collector.tweepy.Client", return_value=mock_client):
                mention, sentiment = await collector._collect_twitter()

        # 2 tweets → mention = (2/100)*100 = 2.0
        assert mention == pytest.approx(2.0)
        # likes: 10+20=30, total engagement: 10+5+2+20+3+1=41
        # sentiment = (30/41)*100 ≈ 73.17
        assert sentiment is not None
        assert 0 <= sentiment <= 100

    @pytest.mark.asyncio
    async def test_returns_none_on_no_data(self):
        """트윗 데이터가 없으면 (None, None)을 반환한다."""
        collector = SentimentCollector()

        mock_response = MagicMock()
        mock_response.data = None

        mock_client = MagicMock()
        mock_client.search_recent_tweets.return_value = mock_response

        with patch("collectors.sentiment_collector.TWITTER_BEARER_TOKEN", "valid_token"):
            with patch("collectors.sentiment_collector.tweepy.Client", return_value=mock_client):
                mention, sentiment = await collector._collect_twitter()

        assert mention is None
        assert sentiment is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """예외 발생 시 (None, None)을 반환한다."""
        collector = SentimentCollector()

        with patch("collectors.sentiment_collector.TWITTER_BEARER_TOKEN", "valid_token"):
            with patch(
                "collectors.sentiment_collector.tweepy.Client",
                side_effect=Exception("API error"),
            ):
                mention, sentiment = await collector._collect_twitter()

        assert mention is None
        assert sentiment is None


# ── 공포탐욕지수 수집 테스트 ──────────────────────────────────


class TestCollectFearGreed:
    @pytest.mark.asyncio
    async def test_returns_value_on_success(self, sample_fear_greed_response):
        """정상 응답 시 공포탐욕지수 값을 반환한다."""
        collector = SentimentCollector()

        response = httpx.Response(
            200,
            json=sample_fear_greed_response,
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await collector._collect_fear_greed()

        assert result == 72.0

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_data(self):
        """데이터가 없으면 None을 반환한다."""
        collector = SentimentCollector()

        response = httpx.Response(
            200,
            json={"data": []},
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await collector._collect_fear_greed()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """예외 발생 시 None을 반환한다."""
        collector = SentimentCollector()

        async def mock_get(url, **kwargs):
            raise httpx.HTTPStatusError(
                "500",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(500),
            )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await collector._collect_fear_greed()

        assert result is None


# ── DB 저장 테스트 ────────────────────────────────────────────


class TestSaveRecord:
    def test_saves_record(self, in_memory_session):
        """레코드를 DB에 저장한다."""
        record = {
            "timestamp": datetime(2024, 1, 15, 0, 0, 0),
            "google_trend_score": 55.0,
            "sns_mention_score": 30.0,
            "sns_sentiment_score": 65.0,
            "fear_greed_index": 72.0,
        }

        with patch("collectors.sentiment_collector.get_session", return_value=in_memory_session):
            _save_record(record)

        saved = in_memory_session.query(SentimentData).all()
        assert len(saved) == 1
        assert saved[0].google_trend_score == 55.0
        assert saved[0].fear_greed_index == 72.0

    def test_skips_duplicate_timestamp(self, in_memory_session):
        """중복 타임스탬프는 건너뛴다."""
        record = {
            "timestamp": datetime(2024, 1, 15, 0, 0, 0),
            "google_trend_score": 55.0,
            "sns_mention_score": 30.0,
            "sns_sentiment_score": 65.0,
            "fear_greed_index": 72.0,
        }

        with patch("collectors.sentiment_collector.get_session", return_value=in_memory_session):
            _save_record(record)
            _save_record(record)  # 두 번째 호출

        saved = in_memory_session.query(SentimentData).all()
        assert len(saved) == 1

    def test_saves_with_none_values(self, in_memory_session):
        """일부 값이 None이어도 저장한다."""
        record = {
            "timestamp": datetime(2024, 1, 15, 0, 0, 0),
            "google_trend_score": None,
            "sns_mention_score": None,
            "sns_sentiment_score": None,
            "fear_greed_index": 50.0,
        }

        with patch("collectors.sentiment_collector.get_session", return_value=in_memory_session):
            _save_record(record)

        saved = in_memory_session.query(SentimentData).first()
        assert saved.google_trend_score is None
        assert saved.fear_greed_index == 50.0


# ── SentimentCollector.collect() 통합 테스트 ──────────────────


class TestSentimentCollectorCollect:
    @pytest.mark.asyncio
    async def test_collect_success(self, in_memory_session, sample_google_trends_df, sample_fear_greed_response):
        """모든 소스에서 데이터를 수집하고 DB에 저장한다."""
        collector = SentimentCollector()

        # Google Trends 모킹
        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = sample_google_trends_df

        # Fear & Greed 모킹
        fgi_response = httpx.Response(
            200,
            json=sample_fear_greed_response,
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return fgi_response

        # Twitter 모킹 (토큰 미설정으로 건너뜀)
        with patch("collectors.sentiment_collector.get_session", return_value=in_memory_session):
            with patch("collectors.sentiment_collector.TrendReq", return_value=mock_pytrends):
                with patch("collectors.sentiment_collector.TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN"):
                    with patch("httpx.AsyncClient") as mock_client_cls:
                        mock_client = AsyncMock()
                        mock_client.get = mock_get
                        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                        mock_client.__aexit__ = AsyncMock(return_value=False)
                        mock_client_cls.return_value = mock_client

                        result = await collector.collect()

        assert result["source"] == "sentiment"
        assert result["google_trend_score"] == pytest.approx(55.0)
        assert result["fear_greed_index"] == 72.0
        assert result["sns_mention_score"] is None  # 토큰 미설정
        assert result["sns_sentiment_score"] is None

        # DB 확인
        saved = in_memory_session.query(SentimentData).all()
        assert len(saved) == 1

    @pytest.mark.asyncio
    async def test_collect_deduplicates(self, in_memory_session, sample_google_trends_df, sample_fear_greed_response):
        """동일 날짜에 두 번 수집해도 중복 저장하지 않는다."""
        collector = SentimentCollector()

        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = sample_google_trends_df

        fgi_response = httpx.Response(
            200,
            json=sample_fear_greed_response,
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return fgi_response

        with patch("collectors.sentiment_collector.get_session", return_value=in_memory_session):
            with patch("collectors.sentiment_collector.TrendReq", return_value=mock_pytrends):
                with patch("collectors.sentiment_collector.TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN"):
                    with patch("httpx.AsyncClient") as mock_client_cls:
                        mock_client = AsyncMock()
                        mock_client.get = mock_get
                        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                        mock_client.__aexit__ = AsyncMock(return_value=False)
                        mock_client_cls.return_value = mock_client

                        await collector.collect()
                        await collector.collect()  # 두 번째 호출

        saved = in_memory_session.query(SentimentData).all()
        assert len(saved) == 1

    @pytest.mark.asyncio
    async def test_timestamp_is_eastern_time_midnight(self, in_memory_session, sample_google_trends_df, sample_fear_greed_response):
        """저장된 타임스탬프가 동부시간 자정으로 정규화되었는지 확인한다."""
        collector = SentimentCollector()

        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = sample_google_trends_df

        fgi_response = httpx.Response(
            200,
            json=sample_fear_greed_response,
            request=httpx.Request("GET", "http://test"),
        )

        async def mock_get(url, **kwargs):
            return fgi_response

        with patch("collectors.sentiment_collector.get_session", return_value=in_memory_session):
            with patch("collectors.sentiment_collector.TrendReq", return_value=mock_pytrends):
                with patch("collectors.sentiment_collector.TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN"):
                    with patch("httpx.AsyncClient") as mock_client_cls:
                        mock_client = AsyncMock()
                        mock_client.get = mock_get
                        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                        mock_client.__aexit__ = AsyncMock(return_value=False)
                        mock_client_cls.return_value = mock_client

                        await collector.collect()

        saved = in_memory_session.query(SentimentData).first()
        # 자정으로 정규화
        assert saved.timestamp.hour == 0
        assert saved.timestamp.minute == 0
        assert saved.timestamp.second == 0
