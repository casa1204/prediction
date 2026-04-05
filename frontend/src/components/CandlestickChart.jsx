import { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';
import { useCurrentPrice } from '../hooks/useQueries';

export default function CandlestickChart() {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const { data } = useCurrentPrice();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 400,
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

    const volumeSeries = chart.addHistogramSeries({
      color: '#0f3460',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    });

    // Generate sample data for display when no real data is available
    const now = Math.floor(Date.now() / 1000);
    const basePrice = data?.price || 0.55;
    const sampleCandles = [];
    const sampleVolume = [];

    for (let i = 100; i >= 0; i--) {
      const time = now - i * 3600;
      const open = basePrice + (Math.random() - 0.5) * 0.05;
      const close = open + (Math.random() - 0.5) * 0.04;
      const high = Math.max(open, close) + Math.random() * 0.02;
      const low = Math.min(open, close) - Math.random() * 0.02;
      const vol = Math.random() * 1e8;

      sampleCandles.push({ time, open, high, low, close });
      sampleVolume.push({
        time,
        value: vol,
        color: close >= open ? 'rgba(0,200,83,0.3)' : 'rgba(255,23,68,0.3)',
      });
    }

    candleSeries.setData(sampleCandles);
    volumeSeries.setData(sampleVolume);
    chart.timeScale().fitContent();

    chartRef.current = chart;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data?.price]);

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">XRP/USD 캔들스틱 차트</h3>
      <div ref={chartContainerRef} />
    </div>
  );
}
