import { useState } from 'react';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip,
} from 'recharts';
import { useFeatureImportance } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

const MODELS = [
  { key: 'lstm', label: 'LSTM' },
  { key: 'xgboost', label: 'XGBoost' },
  { key: 'rf', label: 'Random Forest' },
  { key: 'transformer', label: 'Transformer' },
  { key: 'lgbm', label: 'LightGBM' },
];

const FEATURE_LABELS = {
  close: '종가',
  rsi_14: 'RSI(14)',
  macd: 'MACD',
  macd_signal: 'MACD 시그널',
  macd_histogram: 'MACD 히스토그램',
  bb_upper: '볼린저 상단',
  bb_middle: '볼린저 중간',
  bb_lower: '볼린저 하단',
  sma_5: 'SMA(5)',
  sma_10: 'SMA(10)',
  sma_20: 'SMA(20)',
  sma_50: 'SMA(50)',
  sma_100: 'SMA(100)',
  sma_200: 'SMA(200)',
  ema_12: 'EMA(12)',
  ema_26: 'EMA(26)',
  obv: 'OBV',
  disparity_50: '이격도(50)',
  disparity_100: '이격도(100)',
  disparity_200: '이격도(200)',
  current_wave_number: '엘리엇 파동',
  next_direction: '파동 방향',
  market_phase: '시장 단계',
  wyckoff_phase: '와이코프 단계',
  confidence_score: '신뢰도',
  google_trend_score: 'XRP 검색트렌드',
  sns_mention_score: 'SNS 언급량',
  sns_sentiment_score: 'SNS 감성',
  fear_greed_index: '공포탐욕지수',
  trend_macro_score: '거시경제 트렌드',
  trend_etf_regulatory_score: 'ETF/규제 트렌드',
  trend_sentiment_score: '매수/매도 심리',
  attention_ratio: '관심 비율(XRP/BTC)',
  fomo_spread: 'FOMO 스프레드',
  macro_aggregate: '거시경제 종합',
  active_wallets: '활성 지갑',
  new_wallets: '신규 지갑',
  transaction_count: '거래 건수',
  total_volume_xrp: '총 거래량',
  whale_tx_count: '고래 거래 건수',
  whale_tx_volume: '고래 거래량',
  etf_total_volume: 'ETF 총 거래량',
};

export default function FeatureImportance() {
  const [selectedModel, setSelectedModel] = useState('lstm');
  const { data, isLoading, isError } = useFeatureImportance(selectedModel);

  const featureImportance = data?.feature_importance;
  const chartData = featureImportance
    ? Object.entries(featureImportance)
        .map(([name, value]) => ({
          name: FEATURE_LABELS[name] || name,
          importance: value,
        }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 15)
    : [];

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-gray-300 mb-3">피처 기여도<InfoTooltip ko="각 입력 데이터(피처)가 가격 예측에 얼마나 기여하는지를 나타냅니다. 기여도가 높은 피처일수록 예측에 큰 영향을 미칩니다." en="Shows how much each input feature contributes to the price prediction. Higher importance means greater influence on the prediction." /></h4>
        <div className="flex gap-1">
          {MODELS.map((m) => (
            <button
              key={m.key}
              onClick={() => setSelectedModel(m.key)}
              className={`text-xs px-2 py-1 rounded transition-colors ${
                selectedModel === m.key
                  ? 'bg-accent text-white'
                  : 'bg-primary/50 text-gray-400 hover:bg-accent/30'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading && <p className="text-xs text-gray-500 text-center py-4">로딩 중...</p>}
      {isError && <p className="text-xs text-danger text-center py-4">데이터를 불러올 수 없습니다.</p>}

      {!isLoading && !isError && chartData.length === 0 && (
        <p className="text-xs text-gray-500 text-center py-4">피처 기여도 데이터 수집 중...</p>
      )}

      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
            <XAxis type="number" tick={{ fontSize: 10, fill: '#9ca3af' }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#9ca3af' }} width={120} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #374151' }}
              formatter={(v) => [(v * 100).toFixed(2) + '%', '기여도']}
            />
            <Bar dataKey="importance" fill="#e94560" radius={[0, 4, 4, 0]} name="기여도" />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
