import { useEffect, useRef, useMemo } from 'react';
import { createChart } from 'lightweight-charts';
import { useDailyPriceHistory, useElliottWave, useWyckoff } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

/** ISO 문자열 → Unix timestamp (초) */
function toUnix(isoStr) {
  if (!isoStr) return 0;
  return Math.floor(new Date(isoStr).getTime() / 1000);
}

/**
 * 엘리엇 파동 데이터를 Lightweight Charts 마커 배열로 변환.
 * 충격파(impulse) → 초록 arrowUp belowBar
 * 조정파(corrective) → 주황 arrowDown aboveBar
 */
function buildWaveMarkers(waves) {
  if (!waves || waves.length === 0) return [];
  return waves
    .filter((w) => w.start_time)
    .map((w) => {
      const isImpulse = w.wave_type === 'impulse';
      return {
        time: toUnix(w.start_time),
        position: isImpulse ? 'belowBar' : 'aboveBar',
        color: isImpulse ? '#00c853' : '#ff9800',
        shape: isImpulse ? 'arrowUp' : 'arrowDown',
        text: `${w.wave_number}파`,
      };
    });
}

/**
 * 와이코프 이벤트를 Lightweight Charts 마커 배열로 변환.
 * Spring/UTAD(추세 전환) → 빨간 circle aboveBar
 * 나머지 → 파란 circle belowBar
 */
function buildWyckoffMarkers(events) {
  if (!events || events.length === 0) return [];
  const reversalTypes = new Set(['Spring', 'UTAD']);
  return events
    .filter((e) => e.timestamp)
    .map((e) => {
      const isReversal = reversalTypes.has(e.event_type) || e.is_trend_reversal;
      return {
        time: toUnix(e.timestamp),
        position: isReversal ? 'aboveBar' : 'belowBar',
        color: isReversal ? '#ff1744' : '#2196f3',
        shape: 'circle',
        text: e.event_type,
      };
    });
}

export default function AnalysisChart() {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);

  const { data: priceData, isLoading: priceLoading } = useDailyPriceHistory();
  const { data: elliottData } = useElliottWave();
  const { data: wyckoffData } = useWyckoff();

  const candles = priceData?.candles;
  const waves = elliottData?.waves;
  const events = wyckoffData?.events;

  // 마커 병합 (time 기준 정렬 필수)
  const markers = useMemo(() => {
    const waveM = buildWaveMarkers(waves);
    const wyckoffM = buildWyckoffMarkers(events);
    return [...waveM, ...wyckoffM].sort((a, b) => a.time - b.time);
  }, [waves, events]);

  // 피보나치 목표가 추출
  const fibTargets = useMemo(() => {
    if (!waves || waves.length === 0) return null;
    const first = waves.find((w) => w.fibonacci_targets);
    return first?.fibonacci_targets || null;
  }, [waves]);

  // 차트 생성 (데이터 로딩 완료 후)
  useEffect(() => {
    if (!chartContainerRef.current || priceLoading) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth || 600,
      height: 480,
      layout: {
        background: { color: '#16213e' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1e3a5f' },
        horzLines: { color: '#1e3a5f' },
      },
      crosshair: { mode: 0 },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
      },
      rightPriceScale: { borderColor: '#374151' },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00c853',
      downColor: '#ff1744',
      borderDownColor: '#ff1744',
      borderUpColor: '#00c853',
      wickDownColor: '#ff1744',
      wickUpColor: '#00c853',
    });

    chartRef.current = chart;
    seriesRef.current = candleSeries;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [priceLoading]);

  // 캔들 데이터 세팅
  useEffect(() => {
    if (!seriesRef.current || !candles || candles.length === 0) return;
    seriesRef.current.setData(candles);
    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  // 마커 세팅
  useEffect(() => {
    if (!seriesRef.current) return;
    seriesRef.current.setMarkers(markers);
  }, [markers]);

  // 피보나치 수평선
  useEffect(() => {
    const series = seriesRef.current;
    if (!series || !fibTargets) return;

    const FIB_COLORS = {
      '0.236': '#80cbc4',
      '0.382': '#4db6ac',
      '0.5': '#ffd600',
      '0.618': '#ff9800',
      '0.786': '#f44336',
    };

    const lines = Object.entries(fibTargets).map(([ratio, price]) =>
      series.createPriceLine({
        price,
        color: FIB_COLORS[ratio] || '#ffd600',
        lineWidth: 1,
        lineStyle: 2, // Dashed
        title: `Fib ${ratio}`,
      }),
    );

    return () => {
      lines.forEach((line) => {
        try { series.removePriceLine(line); } catch { /* already removed */ }
      });
    };
  }, [fibTargets]);

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        XRP/USD 기술 분석 차트
        <span className="ml-2 text-xs text-gray-500">
          엘리엇 파동 · 와이코프 이벤트 · 피보나치 오버레이
          <InfoTooltip ko="일봉 캔들스틱 차트 위에 엘리엇 파동 번호(초록=충격파, 주황=조정파)와 와이코프 이벤트(파랑=일반, 빨강=추세전환)가 오버레이됩니다. 피보나치 되돌림 수평선도 표시됩니다. [출처: 가격 데이터 - Binance Klines API / 엘리엇 파동·와이코프 - 일봉 데이터 기반 자체 알고리즘 계산]" en="Daily candlestick chart with Elliott Wave and Wyckoff overlays. [Source: Price - Binance Klines API / Elliott Wave & Wyckoff - calculated from daily candle data]" />
        </span>
      </h3>
      {priceLoading && (
        <div className="flex items-center justify-center h-[480px] text-gray-500 text-sm">
          차트 데이터 로딩 중...
        </div>
      )}
      <div ref={chartContainerRef} style={{ display: priceLoading ? 'none' : 'block' }} />
      {/* 범례 */}
      <div className="flex flex-wrap gap-4 mt-3 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#00c853]" /> 충격파 (1~5)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#ff9800]" /> 조정파 (A~C)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#2196f3]" /> 와이코프 이벤트
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-[#ff1744]" /> 추세 전환 (Spring/UTAD)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 border border-[#ffd600] rounded-sm" /> 피보나치 되돌림
        </span>
      </div>
    </div>
  );
}
