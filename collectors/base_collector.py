"""수집기 기본 클래스 — 재시도, 로깅, 연속 실패 알림 공통 동작 정의."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from config import CONSECUTIVE_FAILURE_ALERT_THRESHOLD, MAX_RETRIES
from db.database import get_session
from db.models import CollectionLog

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """모든 데이터 수집기의 공통 기본 클래스.

    하위 클래스는 ``source_name`` 속성과 ``collect`` 메서드를 구현해야 한다.
    """

    source_name: str = "unknown"

    def __init__(self):
        self._consecutive_failures: int = 0
        self._last_success_data: dict | None = None

    # ── 추상 메서드 ──────────────────────────────────────────

    @abstractmethod
    async def collect(self) -> dict:
        """데이터 수집 실행. 성공 시 수집 데이터 dict 반환."""
        ...

    # ── 재시도 로직 ──────────────────────────────────────────

    async def collect_with_retry(self) -> dict:
        """최대 MAX_RETRIES회 재시도하며 수집.

        지수 백오프(1초, 2초, 4초)를 적용한다.
        모든 재시도 실패 시 마지막 성공 데이터를 반환한다.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            start_time = datetime.now(UTC)
            try:
                data = await self.collect()
                end_time = datetime.now(UTC)

                # 성공 처리
                self._last_success_data = data
                self._consecutive_failures = 0
                self.log_collection("success", start_time, end_time)
                return data

            except Exception as exc:
                end_time = datetime.now(UTC)
                last_error = exc
                self._consecutive_failures += 1

                self.log_collection(
                    "failure",
                    start_time,
                    end_time,
                    error_message=str(exc),
                )

                # 연속 실패 임계값 도달 시 알림
                if self._consecutive_failures == CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
                    self._alert_consecutive_failures()

                # 마지막 시도가 아니면 지수 백오프 대기
                if attempt < MAX_RETRIES - 1:
                    backoff = 2 ** attempt  # 1, 2, 4
                    logger.warning(
                        "[%s] 수집 실패 (시도 %d/%d), %d초 후 재시도: %s",
                        self.source_name,
                        attempt + 1,
                        MAX_RETRIES,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)

        # 모든 재시도 실패
        logger.error(
            "[%s] 모든 재시도 실패 (%d회). 마지막 성공 데이터 반환.",
            self.source_name,
            MAX_RETRIES,
        )

        if self._last_success_data is not None:
            return self._last_success_data

        return {}

    # ── 수집 로그 기록 ───────────────────────────────────────

    def log_collection(
        self,
        status: str,
        start_time: datetime,
        end_time: datetime,
        error_message: str | None = None,
    ):
        """CollectionLog ORM 모델에 수집 로그를 저장한다."""
        session = get_session()
        try:
            log_entry = CollectionLog(
                source=self.source_name,
                start_time=start_time,
                end_time=end_time,
                status=status,
                error_message=error_message,
                consecutive_failures=self._consecutive_failures,
            )
            session.add(log_entry)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("[%s] 수집 로그 저장 실패", self.source_name)
        finally:
            session.close()

    # ── 연속 실패 알림 ───────────────────────────────────────

    def _alert_consecutive_failures(self):
        """연속 실패 횟수가 임계값에 도달했을 때 알림을 발송한다.

        현재는 로깅으로 대체한다.
        """
        logger.critical(
            "[%s] 연속 %d회 수집 실패! 관리자 확인 필요.",
            self.source_name,
            self._consecutive_failures,
        )
