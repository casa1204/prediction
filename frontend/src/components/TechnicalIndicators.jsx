import { useTechnicalIndicators } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

function IndicatorCard({ title, children, tooltip }) {
  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4">
      <h4 className="text-sm font-medium text-gray-300 mb-3">{title}{tooltip && <InfoTooltip ko={tooltip.ko} en={tooltip.en} />}</h4>
      {children}
    </div>
  );
}

export default function TechnicalIndicators() {
  const { data, isLoading, isError } = useTechnicalIndicators();

  if (isLoading) return <div className="text-gray-500 text-sm">기술 지표 로딩 중...</div>;
  if (isError) return <div className="text-danger text-sm">기술 지표를 불러올 수 없습니다.</div>;

  const indicators = data?.indicators;

  if (!indicators) {
    return <div className="text-gray-500 text-sm text-center py-4">기술 지표 데이터 수집 중...</div>;
  }

  return (
    <div className="space-y-4">
      <IndicatorCard title="RSI (14)" tooltip={{ ko: "상대강도지수. 70 이상이면 과매수(하락 가능성), 30 이하면 과매도(상승 가능성)를 나타냅니다.", en: "Relative Strength Index. Above 70 indicates overbought (potential decline), below 30 indicates oversold (potential rise)." }}>
        {indicators.rsi_14 != null ? (
          <div className="flex items-center gap-4">
            <p className={`text-3xl font-bold ${
              indicators.rsi_14 >= 70 ? 'text-danger' : indicators.rsi_14 <= 30 ? 'text-success' : 'text-gray-200'
            }`}>
              {indicators.rsi_14.toFixed(2)}
            </p>
            <p className="text-xs text-gray-400">
              {indicators.rsi_14 >= 70 ? '과매수 구간' : indicators.rsi_14 <= 30 ? '과매도 구간' : '중립 구간'}
            </p>
          </div>
        ) : (
          <p className="text-xs text-gray-500 text-center py-4">데이터 없음</p>
        )}
      </IndicatorCard>
      <IndicatorCard title="MACD (12, 26, 9)" tooltip={{ ko: "이동평균수렴확산. MACD가 시그널선 위로 교차하면 매수 신호, 아래로 교차하면 매도 신호입니다.", en: "Moving Average Convergence Divergence. Bullish when MACD crosses above signal line, bearish when below." }}>
        {indicators.macd != null ? (
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-xs text-gray-400">MACD</p>
              <p className="text-lg font-bold text-gray-200">{indicators.macd.toFixed(6)}</p>
            </div>
            <div>
              <p className="text-xs text-gray-400">Signal</p>
              <p className="text-lg font-bold text-gray-200">{indicators.macd_signal?.toFixed(6) ?? '-'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-400">Histogram</p>
              <p className={`text-lg font-bold ${(indicators.macd_histogram ?? 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                {indicators.macd_histogram?.toFixed(6) ?? '-'}
              </p>
            </div>
          </div>
        ) : (
          <p className="text-xs text-gray-500 text-center py-4">데이터 없음</p>
        )}
      </IndicatorCard>
      <IndicatorCard title="볼린저밴드 (20)" tooltip={{ ko: "가격 변동성을 나타냅니다. 가격이 상단에 닿으면 과매수, 하단에 닿으면 과매도 가능성이 있습니다.", en: "Measures price volatility. Price touching upper band suggests overbought, lower band suggests oversold." }}>
        {indicators.bb_upper != null ? (
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-xs text-gray-400">상단</p>
              <p className="text-lg font-bold text-danger">{indicators.bb_upper.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-xs text-gray-400">중단</p>
              <p className="text-lg font-bold text-warning">{indicators.bb_middle?.toFixed(4) ?? '-'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-400">하단</p>
              <p className="text-lg font-bold text-success">{indicators.bb_lower?.toFixed(4) ?? '-'}</p>
            </div>
          </div>
        ) : (
          <p className="text-xs text-gray-500 text-center py-4">데이터 없음</p>
        )}
      </IndicatorCard>
      <IndicatorCard title="이동평균선 (SMA / EMA)" tooltip={{ ko: "일정 기간의 평균 가격입니다. 단기 이동평균이 장기 이동평균 위에 있으면 상승 추세, 아래면 하락 추세입니다.", en: "Average price over a period. Short-term MA above long-term MA indicates uptrend, below indicates downtrend." }}>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-center text-xs">
          {[
            { label: 'SMA(5)', value: indicators.sma_5 },
            { label: 'SMA(10)', value: indicators.sma_10 },
            { label: 'SMA(20)', value: indicators.sma_20 },
            { label: 'SMA(50)', value: indicators.sma_50 },
            { label: 'SMA(200)', value: indicators.sma_200 },
            { label: 'EMA(12)', value: indicators.ema_12 },
            { label: 'EMA(26)', value: indicators.ema_26 },
          ].map((item) => (
            <div key={item.label} className="bg-primary/50 rounded p-2">
              <p className="text-gray-400">{item.label}</p>
              <p className="text-sm font-medium text-gray-200">{item.value?.toFixed(4) ?? '-'}</p>
            </div>
          ))}
        </div>
      </IndicatorCard>
    </div>
  );
}
