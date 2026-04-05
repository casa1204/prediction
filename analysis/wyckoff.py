"""와이코프 패턴 분석 모듈 — 축적/분배 패턴 감지, 시장 단계 판별, 거래량 분석."""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.models import WyckoffData

logger = logging.getLogger(__name__)

MIN_CANDLES = 200


@dataclass
class WyckoffEvent:
    """감지된 와이코프 이벤트."""

    event_type: str  # "PS","SC","AR","ST","Spring","SOS","LPS" / "PSY","BC","AR","ST","UTAD","LPSY","SOW"
    pattern_type: str  # "accumulation" or "distribution"
    timestamp: datetime
    price: float
    volume: float
    is_trend_reversal: bool = False


@dataclass
class MarketPhase:
    """현재 시장 단계."""

    phase: str  # "Accumulation","Markup","Distribution","Markdown"
    confidence_score: float  # 0~100


@dataclass
class WyckoffPhaseResult:
    """현재 Wyckoff Phase."""

    phase: str  # "A"~"E"
    progress: float  # 0~100


@dataclass
class VolumeAnalysis:
    """가격-거래량 관계 분석."""

    relationship: str  # "divergence" or "convergence"
    details: str


class WyckoffModule:
    """와이코프 패턴을 감지하고 시장 단계를 분석한다."""

    def __init__(self) -> None:
        self._events: list[WyckoffEvent] = []
        self._market_phase: MarketPhase | None = None
        self._wyckoff_phase: WyckoffPhaseResult | None = None

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """단순 이동 평균 계산 (NaN 패딩)."""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1 : i + 1])
        return result

    @staticmethod
    def _find_support_resistance(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 20
    ) -> tuple[float, float]:
        """최근 window 캔들에서 지지선(support)과 저항선(resistance) 계산."""
        recent_lows = lows[-window:]
        recent_highs = highs[-window:]
        support = float(np.min(recent_lows))
        resistance = float(np.max(recent_highs))
        return support, resistance

    @staticmethod
    def _is_high_volume(volume: float, avg_volume: float, threshold: float = 1.5) -> bool:
        """거래량이 평균 대비 threshold 배 이상인지 확인."""
        if avg_volume <= 0:
            return False
        return volume >= avg_volume * threshold

    @staticmethod
    def _is_low_volume(volume: float, avg_volume: float, threshold: float = 0.7) -> bool:
        """거래량이 평균 대비 threshold 배 이하인지 확인."""
        if avg_volume <= 0:
            return False
        return volume <= avg_volume * threshold

    @staticmethod
    def _price_range(high: float, low: float) -> float:
        """캔들의 가격 범위."""
        return abs(high - low)

    # ------------------------------------------------------------------
    # 축적(Accumulation) 패턴 감지
    # ------------------------------------------------------------------

    def detect_accumulation(self, df: pd.DataFrame) -> list[WyckoffEvent]:
        """축적 패턴 이벤트(PS, SC, AR, ST, Spring, SOS, LPS) 감지.

        Parameters
        ----------
        df : pd.DataFrame
            timestamp, open, high, low, close, volume 컬럼 포함. 최소 200캔들.

        Returns
        -------
        list[WyckoffEvent]
            감지된 축적 이벤트 리스트.
        """
        if len(df) < MIN_CANDLES:
            logger.info("캔들 수 %d개 — 최소 %d개 필요.", len(df), MIN_CANDLES)
            return []

        events: list[WyckoffEvent] = []
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        timestamps = df["timestamp"].values

        vol_ma = self._rolling_mean(volumes, 20)
        price_ma_50 = self._rolling_mean(closes, 50)

        # 하락 추세 구간 탐색 (축적은 하락 추세 후 발생)
        # 50MA 기울기가 음수인 구간 찾기
        lookback = min(100, len(df) - 50)
        scan_start = max(50, len(df) - lookback)

        ps_found = False
        sc_found = False
        ar_found = False
        st_found = False
        spring_found = False

        support_level = float(np.min(lows[scan_start:]))
        resistance_level = float(np.max(highs[scan_start:]))
        trading_range = resistance_level - support_level

        if trading_range <= 0:
            return events

        for i in range(scan_start, len(df)):
            ts = pd.Timestamp(timestamps[i])
            price = closes[i]
            vol = volumes[i]
            avg_vol = vol_ma[i] if not np.isnan(vol_ma[i]) else vol

            # PS (Preliminary Support): 하락 추세 중 높은 거래량으로 지지 시도
            if not ps_found and i > scan_start + 5:
                # 이전 5캔들이 하락 추세이고 현재 거래량이 높으면
                recent_trend = closes[i] - closes[i - 5]
                if recent_trend < 0 and self._is_high_volume(vol, avg_vol, 1.3):
                    events.append(WyckoffEvent(
                        event_type="PS",
                        pattern_type="accumulation",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    ps_found = True
                    continue

            # SC (Selling Climax): PS 이후 급격한 하락 + 매우 높은 거래량
            if ps_found and not sc_found:
                price_drop = (closes[i] - closes[i - 3]) / closes[i - 3] if closes[i - 3] > 0 else 0
                if price_drop < -0.03 and self._is_high_volume(vol, avg_vol, 2.0):
                    support_level = min(support_level, lows[i])
                    events.append(WyckoffEvent(
                        event_type="SC",
                        pattern_type="accumulation",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    sc_found = True
                    continue

            # AR (Automatic Rally): SC 이후 반등
            if sc_found and not ar_found:
                price_rise = (closes[i] - closes[i - 3]) / closes[i - 3] if closes[i - 3] > 0 else 0
                if price_rise > 0.02:
                    resistance_level = max(resistance_level, highs[i])
                    events.append(WyckoffEvent(
                        event_type="AR",
                        pattern_type="accumulation",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    ar_found = True
                    continue

            # ST (Secondary Test): AR 이후 SC 근처까지 재하락, 거래량 감소
            if ar_found and not st_found:
                near_support = abs(price - support_level) / trading_range < 0.15 if trading_range > 0 else False
                if near_support and self._is_low_volume(vol, avg_vol, 0.8):
                    events.append(WyckoffEvent(
                        event_type="ST",
                        pattern_type="accumulation",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    st_found = True
                    continue

            # Spring: 지지선 아래로 일시적 하락 후 반등 (추세 전환 신호)
            if st_found and not spring_found:
                if lows[i] < support_level and closes[i] > support_level:
                    events.append(WyckoffEvent(
                        event_type="Spring",
                        pattern_type="accumulation",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                        is_trend_reversal=True,
                    ))
                    spring_found = True
                    continue

            # SOS (Sign of Strength): Spring 이후 강한 상승 + 높은 거래량
            if spring_found and not any(e.event_type == "SOS" for e in events):
                price_rise = (closes[i] - closes[i - 3]) / closes[i - 3] if closes[i - 3] > 0 else 0
                if price_rise > 0.02 and self._is_high_volume(vol, avg_vol, 1.3):
                    events.append(WyckoffEvent(
                        event_type="SOS",
                        pattern_type="accumulation",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    continue

            # LPS (Last Point of Support): SOS 이후 되돌림, 낮은 거래량
            if any(e.event_type == "SOS" for e in events) and not any(e.event_type == "LPS" for e in events):
                price_drop = (closes[i] - closes[i - 2]) / closes[i - 2] if closes[i - 2] > 0 else 0
                if price_drop < -0.01 and self._is_low_volume(vol, avg_vol, 0.8):
                    events.append(WyckoffEvent(
                        event_type="LPS",
                        pattern_type="accumulation",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    break  # 축적 패턴 완성

        self._events.extend(events)
        return events

    # ------------------------------------------------------------------
    # 분배(Distribution) 패턴 감지
    # ------------------------------------------------------------------

    def detect_distribution(self, df: pd.DataFrame) -> list[WyckoffEvent]:
        """분배 패턴 이벤트(PSY, BC, AR, ST, UTAD, LPSY, SOW) 감지.

        Parameters
        ----------
        df : pd.DataFrame
            timestamp, open, high, low, close, volume 컬럼 포함. 최소 200캔들.

        Returns
        -------
        list[WyckoffEvent]
            감지된 분배 이벤트 리스트.
        """
        if len(df) < MIN_CANDLES:
            logger.info("캔들 수 %d개 — 최소 %d개 필요.", len(df), MIN_CANDLES)
            return []

        events: list[WyckoffEvent] = []
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        timestamps = df["timestamp"].values

        vol_ma = self._rolling_mean(volumes, 20)

        lookback = min(100, len(df) - 50)
        scan_start = max(50, len(df) - lookback)

        psy_found = False
        bc_found = False
        ar_found = False
        st_found = False
        utad_found = False

        support_level = float(np.min(lows[scan_start:]))
        resistance_level = float(np.max(highs[scan_start:]))
        trading_range = resistance_level - support_level

        if trading_range <= 0:
            return events

        for i in range(scan_start, len(df)):
            ts = pd.Timestamp(timestamps[i])
            price = closes[i]
            vol = volumes[i]
            avg_vol = vol_ma[i] if not np.isnan(vol_ma[i]) else vol

            # PSY (Preliminary Supply): 상승 추세 중 높은 거래량으로 저항 시도
            if not psy_found and i > scan_start + 5:
                recent_trend = closes[i] - closes[i - 5]
                if recent_trend > 0 and self._is_high_volume(vol, avg_vol, 1.3):
                    events.append(WyckoffEvent(
                        event_type="PSY",
                        pattern_type="distribution",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    psy_found = True
                    continue

            # BC (Buying Climax): PSY 이후 급격한 상승 + 매우 높은 거래량
            if psy_found and not bc_found:
                price_rise = (closes[i] - closes[i - 3]) / closes[i - 3] if closes[i - 3] > 0 else 0
                if price_rise > 0.03 and self._is_high_volume(vol, avg_vol, 2.0):
                    resistance_level = max(resistance_level, highs[i])
                    events.append(WyckoffEvent(
                        event_type="BC",
                        pattern_type="distribution",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    bc_found = True
                    continue

            # AR (Automatic Reaction): BC 이후 하락
            if bc_found and not ar_found:
                price_drop = (closes[i] - closes[i - 3]) / closes[i - 3] if closes[i - 3] > 0 else 0
                if price_drop < -0.02:
                    support_level = min(support_level, lows[i])
                    events.append(WyckoffEvent(
                        event_type="AR",
                        pattern_type="distribution",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    ar_found = True
                    continue

            # ST (Secondary Test): AR 이후 BC 근처까지 재상승, 거래량 감소
            if ar_found and not st_found:
                near_resistance = abs(price - resistance_level) / trading_range < 0.15 if trading_range > 0 else False
                if near_resistance and self._is_low_volume(vol, avg_vol, 0.8):
                    events.append(WyckoffEvent(
                        event_type="ST",
                        pattern_type="distribution",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    st_found = True
                    continue

            # UTAD (Upthrust After Distribution): 저항선 위로 일시적 상승 후 하락 (추세 전환)
            if st_found and not utad_found:
                if highs[i] > resistance_level and closes[i] < resistance_level:
                    events.append(WyckoffEvent(
                        event_type="UTAD",
                        pattern_type="distribution",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                        is_trend_reversal=True,
                    ))
                    utad_found = True
                    continue

            # LPSY (Last Point of Supply): UTAD 이후 약한 반등 + 낮은 거래량
            if utad_found and not any(e.event_type == "LPSY" for e in events):
                price_rise = (closes[i] - closes[i - 2]) / closes[i - 2] if closes[i - 2] > 0 else 0
                if price_rise > 0.005 and self._is_low_volume(vol, avg_vol, 0.8):
                    events.append(WyckoffEvent(
                        event_type="LPSY",
                        pattern_type="distribution",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    continue

            # SOW (Sign of Weakness): LPSY 이후 강한 하락 + 높은 거래량
            if any(e.event_type == "LPSY" for e in events) and not any(e.event_type == "SOW" for e in events):
                price_drop = (closes[i] - closes[i - 3]) / closes[i - 3] if closes[i - 3] > 0 else 0
                if price_drop < -0.02 and self._is_high_volume(vol, avg_vol, 1.3):
                    events.append(WyckoffEvent(
                        event_type="SOW",
                        pattern_type="distribution",
                        timestamp=ts,
                        price=price,
                        volume=vol,
                    ))
                    break  # 분배 패턴 완성

        self._events.extend(events)
        return events

    # ------------------------------------------------------------------
    # 시장 단계 판별
    # ------------------------------------------------------------------

    def determine_market_phase(self, df: pd.DataFrame) -> MarketPhase:
        """현재 시장 단계(Accumulation/Markup/Distribution/Markdown) 판별.

        가격 추세, 거래량 패턴, 이동평균 관계를 종합하여 판별한다.
        신뢰도 점수(0~100) 포함.

        Parameters
        ----------
        df : pd.DataFrame
            timestamp, close, volume 컬럼 포함.

        Returns
        -------
        MarketPhase
        """
        closes = df["close"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        n = len(closes)

        if n < 20:
            return MarketPhase(phase="Accumulation", confidence_score=0.0)

        # 단기/장기 이동평균
        ma_short = self._rolling_mean(closes, min(20, n))
        ma_long = self._rolling_mean(closes, min(50, n))

        # 최근 가격 추세 (20캔들)
        recent = closes[-20:]
        trend_slope = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0

        # 거래량 추세
        vol_recent = volumes[-20:]
        vol_slope = (np.mean(vol_recent[-5:]) - np.mean(vol_recent[:5])) / np.mean(vol_recent[:5]) if np.mean(vol_recent[:5]) > 0 else 0

        # 가격 변동성 (표준편차 / 평균)
        volatility = np.std(recent) / np.mean(recent) if np.mean(recent) > 0 else 0

        # 이동평균 관계
        last_ma_short = ma_short[-1] if not np.isnan(ma_short[-1]) else closes[-1]
        last_ma_long = ma_long[-1] if not np.isnan(ma_long[-1]) else closes[-1]
        ma_diff = (last_ma_short - last_ma_long) / last_ma_long if last_ma_long > 0 else 0

        # 점수 기반 판별
        scores = {
            "Accumulation": 0.0,
            "Markup": 0.0,
            "Distribution": 0.0,
            "Markdown": 0.0,
        }

        # 추세 기반 점수
        if trend_slope > 0.05:
            scores["Markup"] += 40
        elif trend_slope > 0.01:
            scores["Markup"] += 20
            scores["Distribution"] += 10
        elif trend_slope < -0.05:
            scores["Markdown"] += 40
        elif trend_slope < -0.01:
            scores["Markdown"] += 20
            scores["Accumulation"] += 10
        else:
            # 횡보 구간
            scores["Accumulation"] += 20
            scores["Distribution"] += 20

        # 거래량 추세 기반 점수
        if vol_slope > 0.3:
            scores["Markup"] += 15
            scores["Markdown"] += 10
        elif vol_slope < -0.2:
            scores["Accumulation"] += 15
            scores["Distribution"] += 15

        # 변동성 기반 점수
        if volatility < 0.02:
            scores["Accumulation"] += 15
            scores["Distribution"] += 15
        elif volatility > 0.05:
            scores["Markup"] += 10
            scores["Markdown"] += 10

        # 이동평균 관계 기반 점수
        if ma_diff > 0.02:
            scores["Markup"] += 20
        elif ma_diff < -0.02:
            scores["Markdown"] += 20
        else:
            scores["Accumulation"] += 10
            scores["Distribution"] += 10

        # 감지된 이벤트 기반 보정
        accum_events = [e for e in self._events if e.pattern_type == "accumulation"]
        dist_events = [e for e in self._events if e.pattern_type == "distribution"]
        if accum_events:
            scores["Accumulation"] += 15
        if dist_events:
            scores["Distribution"] += 15

        # 최고 점수 단계 선택
        best_phase = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = min(100.0, (scores[best_phase] / total * 100) if total > 0 else 0.0)

        self._market_phase = MarketPhase(phase=best_phase, confidence_score=round(confidence, 1))
        return self._market_phase

    # ------------------------------------------------------------------
    # Wyckoff Phase (A~E) 판별
    # ------------------------------------------------------------------

    def determine_wyckoff_phase(self, df: pd.DataFrame) -> WyckoffPhaseResult:
        """현재 Wyckoff Phase(A~E) 판별 및 진행 상태 산출.

        Phase A: 기존 추세 정지 (PS, SC, AR, ST)
        Phase B: 원인 구축 (횡보, ST 반복)
        Phase C: 테스트 (Spring / UTAD)
        Phase D: 추세 전환 확인 (SOS/LPS 또는 LPSY/SOW)
        Phase E: 새로운 추세 진행

        Parameters
        ----------
        df : pd.DataFrame
            timestamp, close, volume 컬럼 포함.

        Returns
        -------
        WyckoffPhaseResult
        """
        if not self._events:
            # 이벤트가 없으면 가격 추세로 추정
            closes = df["close"].values.astype(float)
            if len(closes) < 20:
                return WyckoffPhaseResult(phase="A", progress=0.0)

            recent_trend = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] > 0 else 0
            if abs(recent_trend) > 0.1:
                return WyckoffPhaseResult(phase="E", progress=80.0)
            return WyckoffPhaseResult(phase="A", progress=10.0)

        # 이벤트 유형으로 Phase 판별
        event_types = {e.event_type for e in self._events}

        # Phase D/E 확인 (SOS+LPS 또는 LPSY+SOW 존재)
        has_sos_lps = "SOS" in event_types and "LPS" in event_types
        has_lpsy_sow = "LPSY" in event_types and "SOW" in event_types

        if has_sos_lps or has_lpsy_sow:
            # 추세 전환 확인 후 새 추세 진행 중인지 확인
            closes = df["close"].values.astype(float)
            last_event = self._events[-1]
            last_event_ts = last_event.timestamp
            # 마지막 이벤트 이후 캔들 수 확인
            df_ts = pd.to_datetime(df["timestamp"])
            after_event = df_ts >= last_event_ts
            candles_after = after_event.sum()

            if candles_after > 20:
                return WyckoffPhaseResult(phase="E", progress=90.0)
            return WyckoffPhaseResult(phase="D", progress=75.0)

        # Phase C 확인 (Spring 또는 UTAD 존재)
        has_spring = "Spring" in event_types
        has_utad = "UTAD" in event_types

        if has_spring or has_utad:
            return WyckoffPhaseResult(phase="C", progress=55.0)

        # Phase B 확인 (ST 존재, 횡보 구간)
        has_st = "ST" in event_types

        if has_st:
            return WyckoffPhaseResult(phase="B", progress=35.0)

        # Phase A (PS, SC, AR 등 초기 이벤트만 존재)
        progress = min(25.0, len(self._events) * 8.0)
        self._wyckoff_phase = WyckoffPhaseResult(phase="A", progress=progress)
        return self._wyckoff_phase

    # ------------------------------------------------------------------
    # 거래량 분석
    # ------------------------------------------------------------------

    def analyze_volume(self, df: pd.DataFrame) -> VolumeAnalysis:
        """가격-거래량 확산/수렴 관계 분석.

        - 수렴(convergence): 가격과 거래량이 같은 방향으로 움직임
        - 확산(divergence): 가격과 거래량이 반대 방향으로 움직임

        Parameters
        ----------
        df : pd.DataFrame
            timestamp, close, volume 컬럼 포함.

        Returns
        -------
        VolumeAnalysis
        """
        closes = df["close"].values.astype(float)
        volumes = df["volume"].values.astype(float)
        n = len(closes)

        if n < 10:
            return VolumeAnalysis(
                relationship="convergence",
                details="데이터 부족으로 분석 불가 (최소 10캔들 필요)",
            )

        # 최근 20캔들 (또는 가용 데이터) 분석
        window = min(20, n)
        recent_closes = closes[-window:]
        recent_volumes = volumes[-window:]

        # 가격 변화율과 거래량 변화율 계산
        price_changes = np.diff(recent_closes)
        vol_changes = np.diff(recent_volumes)

        if len(price_changes) == 0:
            return VolumeAnalysis(
                relationship="convergence",
                details="변화 데이터 부족",
            )

        # 방향 일치 비율 계산
        same_direction = 0
        total = len(price_changes)
        for pc, vc in zip(price_changes, vol_changes):
            if (pc > 0 and vc > 0) or (pc < 0 and vc < 0) or (pc == 0 and vc == 0):
                same_direction += 1

        convergence_ratio = same_direction / total if total > 0 else 0.5

        # 가격 추세 방향
        price_trend = "상승" if recent_closes[-1] > recent_closes[0] else "하락"
        vol_trend = "증가" if np.mean(recent_volumes[-5:]) > np.mean(recent_volumes[:5]) else "감소"

        if convergence_ratio >= 0.55:
            relationship = "convergence"
            details = (
                f"가격-거래량 수렴: 가격 {price_trend}, 거래량 {vol_trend}. "
                f"방향 일치율 {convergence_ratio:.0%}. "
                f"현재 추세가 거래량에 의해 뒷받침되고 있음."
            )
        else:
            relationship = "divergence"
            details = (
                f"가격-거래량 확산: 가격 {price_trend}, 거래량 {vol_trend}. "
                f"방향 일치율 {convergence_ratio:.0%}. "
                f"추세 약화 또는 전환 가능성 존재."
            )

        return VolumeAnalysis(relationship=relationship, details=details)

    # ------------------------------------------------------------------
    # DB 저장
    # ------------------------------------------------------------------

    def save_to_db(
        self,
        events: list[WyckoffEvent],
        session: Session,
        market_phase: MarketPhase | None = None,
        wyckoff_phase: WyckoffPhaseResult | None = None,
    ) -> int:
        """감지된 와이코프 이벤트를 WyckoffData 테이블에 저장한다.

        Parameters
        ----------
        events : list[WyckoffEvent]
            저장할 이벤트 리스트.
        session : Session
            SQLAlchemy 세션.
        market_phase : MarketPhase | None
            시장 단계 (모든 레코드에 동일 적용).
        wyckoff_phase : WyckoffPhaseResult | None
            와이코프 Phase (모든 레코드에 동일 적용).

        Returns
        -------
        int
            새로 저장된 레코드 수.
        """
        if not events:
            return 0

        mp = market_phase or self._market_phase
        wp = wyckoff_phase or self._wyckoff_phase

        saved = 0
        for evt in events:
            record = WyckoffData(
                timestamp=evt.timestamp,
                event_type=evt.event_type,
                pattern_type=evt.pattern_type,
                price=evt.price,
                volume=evt.volume,
                market_phase=mp.phase if mp else None,
                wyckoff_phase=wp.phase if wp else None,
                confidence_score=mp.confidence_score if mp else None,
                is_trend_reversal=evt.is_trend_reversal,
            )
            session.add(record)
            saved += 1

        if saved > 0:
            session.commit()
            logger.info("%d개의 와이코프 이벤트 레코드를 저장했습니다.", saved)

        return saved
