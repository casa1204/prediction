"""엘리엇 파동 분석 모듈 — Impulse/Corrective 패턴 감지 및 피보나치 목표가 계산."""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sqlalchemy.orm import Session

from db.models import ElliottWaveData

logger = logging.getLogger(__name__)

# 피보나치 되돌림 비율
FIBONACCI_RATIOS = {
    "0.236": 0.236,
    "0.382": 0.382,
    "0.5": 0.5,
    "0.618": 0.618,
    "0.786": 0.786,
}

MIN_CANDLES = 200


@dataclass
class WavePattern:
    """감지된 엘리엇 파동 패턴."""

    wave_number: str  # "1","2","3","4","5","A","B","C"
    wave_type: str  # "impulse" or "corrective"
    start_price: float
    end_price: float
    start_time: datetime
    end_time: datetime
    wave_degree: str  # "Primary","Intermediate" 등
    is_valid: bool = True
    fibonacci_targets: dict = field(default_factory=dict)


@dataclass
class WavePosition:
    """현재 진행 중인 파동 위치."""

    current_wave: str  # "1"~"5" 또는 "A"~"C"
    next_direction: str  # "up" or "down"
    wave_degree: str  # "Primary","Intermediate" 등


class ElliottWaveModule:
    """엘리엇 파동 패턴을 감지하고 분석한다."""

    def __init__(self) -> None:
        self._waves: list[WavePattern] = []

    # ------------------------------------------------------------------
    # 극점(local extrema) 탐색
    # ------------------------------------------------------------------

    @staticmethod
    def _find_extrema(
        prices: np.ndarray, order: int = 10
    ) -> list[tuple[int, float, str]]:
        """가격 배열에서 극대/극소 인덱스를 찾아 (index, price, type) 리스트로 반환.

        Parameters
        ----------
        prices : np.ndarray
            종가 배열.
        order : int
            argrelextrema 비교 윈도우 크기.

        Returns
        -------
        list[tuple[int, float, str]]
            (인덱스, 가격, "max"|"min") 튜플 리스트 (인덱스 오름차순 정렬).
        """
        maxima = argrelextrema(prices, np.greater_equal, order=order)[0]
        minima = argrelextrema(prices, np.less_equal, order=order)[0]

        extrema: list[tuple[int, float, str]] = []
        for idx in maxima:
            extrema.append((int(idx), float(prices[idx]), "max"))
        for idx in minima:
            extrema.append((int(idx), float(prices[idx]), "min"))

        # 인덱스 순 정렬
        extrema.sort(key=lambda x: x[0])

        # 연속 같은 타입 제거 — 교대(alternating) 극점만 유지
        filtered: list[tuple[int, float, str]] = []
        for pt in extrema:
            if not filtered or filtered[-1][2] != pt[2]:
                filtered.append(pt)
            else:
                # 같은 타입이면 더 극단적인 값으로 교체
                if pt[2] == "max" and pt[1] > filtered[-1][1]:
                    filtered[-1] = pt
                elif pt[2] == "min" and pt[1] < filtered[-1][1]:
                    filtered[-1] = pt
        return filtered

    # ------------------------------------------------------------------
    # 파동 차수(degree) 결정
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_degree(candle_count: int) -> str:
        """캔들 수에 따라 파동 차수를 결정한다."""
        if candle_count >= 1000:
            return "Primary"
        elif candle_count >= 500:
            return "Intermediate"
        else:
            return "Minor"

    # ------------------------------------------------------------------
    # Impulse Wave (5파) 감지
    # ------------------------------------------------------------------

    def _detect_impulse(
        self,
        extrema: list[tuple[int, float, str]],
        timestamps: pd.Series,
        degree: str,
    ) -> list[WavePattern]:
        """극점 리스트에서 Impulse Wave(1~5파) 패턴을 감지한다.

        상승 충격파 기준:
        - 1파: min → max (상승)
        - 2파: max → min (하락, 1파 시작 위)
        - 3파: min → max (상승)
        - 4파: max → min (하락, 1파 끝 위)
        - 5파: min → max (상승)
        """
        waves: list[WavePattern] = []
        i = 0
        while i + 5 < len(extrema):
            # 상승 충격파: min 시작
            if extrema[i][2] == "min":
                pts = extrema[i : i + 6]  # 6개 극점 → 5개 파동
                p0, p1, p2, p3, p4, p5 = (
                    pts[0][1],
                    pts[1][1],
                    pts[2][1],
                    pts[3][1],
                    pts[4][1],
                    pts[5][1],
                )
                # 기본 구조 확인: min-max-min-max-min-max
                types = [pt[2] for pt in pts]
                if types != ["min", "max", "min", "max", "min", "max"]:
                    i += 1
                    continue

                # 상승 추세 확인
                if not (p1 > p0 and p3 > p1 and p5 > p0):
                    i += 1
                    continue

                wave_data = [
                    ("1", p0, p1, pts[0][0], pts[1][0]),
                    ("2", p1, p2, pts[1][0], pts[2][0]),
                    ("3", p2, p3, pts[2][0], pts[3][0]),
                    ("4", p3, p4, pts[3][0], pts[4][0]),
                    ("5", p4, p5, pts[4][0], pts[5][0]),
                ]

                impulse_waves = []
                for wn, sp, ep, si, ei in wave_data:
                    w = WavePattern(
                        wave_number=wn,
                        wave_type="impulse",
                        start_price=sp,
                        end_price=ep,
                        start_time=timestamps.iloc[si],
                        end_time=timestamps.iloc[ei],
                        wave_degree=degree,
                    )
                    w.fibonacci_targets = self.calculate_fibonacci_targets(w)
                    impulse_waves.append(w)

                # 규칙 검증
                valid = self._validate_impulse_rules(impulse_waves)
                for w in impulse_waves:
                    w.is_valid = valid
                waves.extend(impulse_waves)
                i += 5
            # 하락 충격파: max 시작
            elif extrema[i][2] == "max":
                pts = extrema[i : i + 6]
                if len(pts) < 6:
                    break
                p0, p1, p2, p3, p4, p5 = (
                    pts[0][1],
                    pts[1][1],
                    pts[2][1],
                    pts[3][1],
                    pts[4][1],
                    pts[5][1],
                )
                types = [pt[2] for pt in pts]
                if types != ["max", "min", "max", "min", "max", "min"]:
                    i += 1
                    continue

                if not (p1 < p0 and p3 < p1 and p5 < p0):
                    i += 1
                    continue

                wave_data = [
                    ("1", p0, p1, pts[0][0], pts[1][0]),
                    ("2", p1, p2, pts[1][0], pts[2][0]),
                    ("3", p2, p3, pts[2][0], pts[3][0]),
                    ("4", p3, p4, pts[3][0], pts[4][0]),
                    ("5", p4, p5, pts[4][0], pts[5][0]),
                ]

                impulse_waves = []
                for wn, sp, ep, si, ei in wave_data:
                    w = WavePattern(
                        wave_number=wn,
                        wave_type="impulse",
                        start_price=sp,
                        end_price=ep,
                        start_time=timestamps.iloc[si],
                        end_time=timestamps.iloc[ei],
                        wave_degree=degree,
                    )
                    w.fibonacci_targets = self.calculate_fibonacci_targets(w)
                    impulse_waves.append(w)

                valid = self._validate_impulse_rules_bearish(impulse_waves)
                for w in impulse_waves:
                    w.is_valid = valid
                waves.extend(impulse_waves)
                i += 5
            else:
                i += 1
        return waves

    # ------------------------------------------------------------------
    # Corrective Wave (A-B-C) 감지
    # ------------------------------------------------------------------

    def _detect_corrective(
        self,
        extrema: list[tuple[int, float, str]],
        timestamps: pd.Series,
        degree: str,
    ) -> list[WavePattern]:
        """극점 리스트에서 Corrective Wave(A-B-C) 패턴을 감지한다."""
        waves: list[WavePattern] = []
        i = 0
        while i + 3 < len(extrema):
            pts = extrema[i : i + 4]
            types = [pt[2] for pt in pts]

            # 하락 조정: max-min-max-min
            if types == ["max", "min", "max", "min"]:
                p0, p1, p2, p3 = pts[0][1], pts[1][1], pts[2][1], pts[3][1]
                if p1 < p0 and p2 > p1 and p3 < p2:
                    wave_data = [
                        ("A", p0, p1, pts[0][0], pts[1][0]),
                        ("B", p1, p2, pts[1][0], pts[2][0]),
                        ("C", p2, p3, pts[2][0], pts[3][0]),
                    ]
                    for wn, sp, ep, si, ei in wave_data:
                        w = WavePattern(
                            wave_number=wn,
                            wave_type="corrective",
                            start_price=sp,
                            end_price=ep,
                            start_time=timestamps.iloc[si],
                            end_time=timestamps.iloc[ei],
                            wave_degree=degree,
                        )
                        w.fibonacci_targets = self.calculate_fibonacci_targets(w)
                        waves.append(w)
                    i += 3
                    continue

            # 상승 조정: min-max-min-max
            if types == ["min", "max", "min", "max"]:
                p0, p1, p2, p3 = pts[0][1], pts[1][1], pts[2][1], pts[3][1]
                if p1 > p0 and p2 < p1 and p3 > p2:
                    wave_data = [
                        ("A", p0, p1, pts[0][0], pts[1][0]),
                        ("B", p1, p2, pts[1][0], pts[2][0]),
                        ("C", p2, p3, pts[2][0], pts[3][0]),
                    ]
                    for wn, sp, ep, si, ei in wave_data:
                        w = WavePattern(
                            wave_number=wn,
                            wave_type="corrective",
                            start_price=sp,
                            end_price=ep,
                            start_time=timestamps.iloc[si],
                            end_time=timestamps.iloc[ei],
                            wave_degree=degree,
                        )
                        w.fibonacci_targets = self.calculate_fibonacci_targets(w)
                        waves.append(w)
                    i += 3
                    continue
            i += 1
        return waves

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def detect_waves(self, df: pd.DataFrame) -> list[WavePattern]:
        """Impulse Wave(1~5파)와 Corrective Wave(A-B-C) 감지.

        Parameters
        ----------
        df : pd.DataFrame
            timestamp, close 컬럼을 포함하는 OHLCV 데이터. 최소 200개 캔들 필요.

        Returns
        -------
        list[WavePattern]
            감지된 파동 패턴 리스트. 200개 미만이면 빈 리스트.
        """
        if df.empty or len(df) < MIN_CANDLES:
            logger.info(
                "캔들 수 %d개 — 최소 %d개 필요. 파동 분석을 건너뜁니다.",
                len(df),
                MIN_CANDLES,
            )
            return []

        prices = df["close"].values.astype(float)
        timestamps = df["timestamp"].reset_index(drop=True)
        degree = self._determine_degree(len(df))

        # 극점 탐색 (order는 데이터 크기에 비례)
        order = max(5, len(prices) // 40)
        extrema = self._find_extrema(prices, order=order)

        if len(extrema) < 4:
            logger.info("극점이 %d개로 부족하여 파동 감지를 건너뜁니다.", len(extrema))
            return []

        impulse = self._detect_impulse(extrema, timestamps, degree)
        corrective = self._detect_corrective(extrema, timestamps, degree)

        self._waves = impulse + corrective
        return self._waves

    def validate_wave_rules(self, wave: WavePattern) -> bool:
        """엘리엇 파동 규칙 검증.

        단독 WavePattern이 아닌, 같은 세트의 5개 파동을 기준으로 검증한다.
        단일 파동만 전달되면 해당 파동이 속한 세트를 self._waves에서 찾는다.

        규칙:
        - 2파는 1파 시작점 아래로 내려가지 않음 (상승) / 위로 올라가지 않음 (하락)
        - 3파는 가장 짧은 충격파가 아님
        - 4파는 1파 영역과 겹치지 않음
        """
        if wave.wave_type != "impulse":
            return True  # 조정파에는 이 규칙 미적용

        # 같은 세트의 impulse 파동 찾기
        impulse_set = self._find_impulse_set(wave)
        if impulse_set is None or len(impulse_set) != 5:
            return wave.is_valid

        return self._check_rules(impulse_set)

    def calculate_fibonacci_targets(self, wave: WavePattern) -> dict[str, float]:
        """피보나치 되돌림 목표가 계산.

        각 비율에 대해: start_price + (end_price - start_price) * ratio

        Returns
        -------
        dict[str, float]
            {"0.236": ..., "0.382": ..., "0.5": ..., "0.618": ..., "0.786": ...}
        """
        diff = wave.end_price - wave.start_price
        return {key: wave.start_price + diff * ratio for key, ratio in FIBONACCI_RATIOS.items()}

    def get_current_position(self) -> WavePosition:
        """현재 진행 중인 파동 위치와 예상 다음 파동 방향 반환."""
        if not self._waves:
            return WavePosition(
                current_wave="1", next_direction="up", wave_degree="Minor"
            )

        last = self._waves[-1]
        current_wave = last.wave_number
        degree = last.wave_degree

        # 다음 방향 결정
        next_direction = self._predict_next_direction(last)

        return WavePosition(
            current_wave=current_wave,
            next_direction=next_direction,
            wave_degree=degree,
        )

    # ------------------------------------------------------------------
    # DB 저장
    # ------------------------------------------------------------------

    def save_to_db(self, waves: list[WavePattern], session: Session) -> int:
        """감지된 파동 패턴을 ElliottWaveData 테이블에 저장한다.

        Returns
        -------
        int
            새로 저장된 레코드 수.
        """
        if not waves:
            return 0

        saved = 0
        for w in waves:
            record = ElliottWaveData(
                timestamp=w.start_time,
                wave_number=w.wave_number,
                wave_type=w.wave_type,
                start_price=w.start_price,
                end_price=w.end_price,
                start_time=w.start_time,
                end_time=w.end_time,
                wave_degree=w.wave_degree,
                is_valid=w.is_valid,
                fibonacci_targets=w.fibonacci_targets,
            )
            session.add(record)
            saved += 1

        if saved > 0:
            session.commit()
            logger.info("%d개의 엘리엇 파동 레코드를 저장했습니다.", saved)

        return saved

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _find_impulse_set(self, wave: WavePattern) -> list[WavePattern] | None:
        """wave가 속한 impulse 5파 세트를 self._waves에서 찾는다."""
        impulse_waves = [w for w in self._waves if w.wave_type == "impulse"]
        # wave_number 순서로 그룹핑 (1-2-3-4-5)
        for i in range(len(impulse_waves) - 4):
            group = impulse_waves[i : i + 5]
            numbers = [w.wave_number for w in group]
            if numbers == ["1", "2", "3", "4", "5"] and wave in group:
                return group
        return None

    def _check_rules(self, impulse_set: list[WavePattern]) -> bool:
        """5파 impulse 세트에 대해 3가지 규칙을 검증한다."""
        w1, w2, w3, w4, w5 = impulse_set

        # 상승 충격파인지 하락 충격파인지 판별
        is_bullish = w1.end_price > w1.start_price

        if is_bullish:
            return self._check_bullish_rules(w1, w2, w3, w4, w5)
        else:
            return self._check_bearish_rules(w1, w2, w3, w4, w5)

    @staticmethod
    def _check_bullish_rules(
        w1: WavePattern,
        w2: WavePattern,
        w3: WavePattern,
        w4: WavePattern,
        w5: WavePattern,
    ) -> bool:
        """상승 충격파 규칙 검증."""
        # 규칙 1: 2파는 1파 시작점 아래로 내려가지 않음
        if w2.end_price < w1.start_price:
            return False

        # 규칙 2: 3파는 가장 짧은 충격파가 아님
        len1 = abs(w1.end_price - w1.start_price)
        len3 = abs(w3.end_price - w3.start_price)
        len5 = abs(w5.end_price - w5.start_price)
        if len3 <= len1 and len3 <= len5:
            return False

        # 규칙 3: 4파는 1파 영역과 겹치지 않음
        if w4.end_price < w1.end_price:
            return False

        return True

    @staticmethod
    def _check_bearish_rules(
        w1: WavePattern,
        w2: WavePattern,
        w3: WavePattern,
        w4: WavePattern,
        w5: WavePattern,
    ) -> bool:
        """하락 충격파 규칙 검증."""
        # 규칙 1: 2파는 1파 시작점 위로 올라가지 않음
        if w2.end_price > w1.start_price:
            return False

        # 규칙 2: 3파는 가장 짧은 충격파가 아님
        len1 = abs(w1.end_price - w1.start_price)
        len3 = abs(w3.end_price - w3.start_price)
        len5 = abs(w5.end_price - w5.start_price)
        if len3 <= len1 and len3 <= len5:
            return False

        # 규칙 3: 4파는 1파 영역과 겹치지 않음 (하락 시 4파 고점이 1파 저점 위)
        if w4.end_price > w1.end_price:
            return False

        return True

    def _validate_impulse_rules(self, impulse_waves: list[WavePattern]) -> bool:
        """상승 충격파 5파 세트 규칙 검증 (detect 시 사용)."""
        if len(impulse_waves) != 5:
            return False
        return self._check_bullish_rules(*impulse_waves)

    def _validate_impulse_rules_bearish(self, impulse_waves: list[WavePattern]) -> bool:
        """하락 충격파 5파 세트 규칙 검증 (detect 시 사용)."""
        if len(impulse_waves) != 5:
            return False
        return self._check_bearish_rules(*impulse_waves)

    @staticmethod
    def _predict_next_direction(wave: WavePattern) -> str:
        """마지막 파동을 기반으로 다음 방향을 예측한다."""
        if wave.wave_type == "impulse":
            is_bullish = wave.end_price > wave.start_price
            if wave.wave_number in ("1", "3"):
                return "down"  # 조정 하락 예상
            elif wave.wave_number == "2":
                return "up" if is_bullish else "down"
            elif wave.wave_number == "4":
                return "up" if is_bullish else "down"
            elif wave.wave_number == "5":
                return "down" if is_bullish else "up"  # 조정파 시작
        elif wave.wave_type == "corrective":
            if wave.wave_number == "A":
                return "up"  # B파 반등
            elif wave.wave_number == "B":
                return "down"  # C파 하락
            elif wave.wave_number == "C":
                return "up"  # 새 충격파 시작
        return "up"
