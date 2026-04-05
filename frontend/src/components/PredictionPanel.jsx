import { useState } from 'react';
import { usePredictions } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

const TIMEFRAMES = [
  { key: 'short', label: '단기 (1~7일)' },
  { key: 'mid', label: '중기 (1주~1개월)' },
  { key: 'long', label: '장기 (1~3개월)' },
];

const MODEL_LABELS = {
  lstm: 'LSTM',
  xgboost: 'XGBoost',
  rf: 'Random Forest',
  transformer: 'Transformer',
  ensemble: '앙상블 (통합)',
};

function DirectionBadge({ direction }) {
  const isUp = direction === 'up';
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
        isUp ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'
      }`}
    >
      {isUp ? '▲ 상승' : '▼ 하락'}
    </span>
  );
}

function ModelCard({ name, result, isEnsemble }) {
  if (!result) return null;

  return (
    <div
      className={`rounded-lg p-3 border ${
        isEnsemble
          ? 'bg-accent/40 border-highlight/50'
          : 'bg-primary/50 border-gray-700'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className={`text-xs font-medium ${isEnsemble ? 'text-highlight' : 'text-gray-300'}`}>
          {MODEL_LABELS[name] || name}
        </span>
        <DirectionBadge direction={result.predicted_direction} />
      </div>
      <p className="text-lg font-bold">${(result.predicted_price ?? 0).toFixed(4)}</p>
      <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-gray-400">
        <div>
          <p>상승</p>
          <p className="text-success font-medium">{(result.up_probability ?? 0).toFixed(1)}%</p>
        </div>
        <div>
          <p>하락</p>
          <p className="text-danger font-medium">{(result.down_probability ?? 0).toFixed(1)}%</p>
        </div>
        <div>
          <p>신뢰도</p>
          <p className="text-warning font-medium">{(result.confidence ?? 0).toFixed(1)}%</p>
        </div>
      </div>
    </div>
  );
}

export default function PredictionPanel() {
  const [timeframe, setTimeframe] = useState('short');
  const { data, isLoading, isError } = usePredictions(timeframe);

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">예측 결과<InfoTooltip ko="5개의 ML 모델(LSTM, XGBoost, Random Forest, Transformer, LightGBM)이 각각 독립적으로 예측한 결과를 앙상블(가중 평균)로 통합한 최종 예측입니다. 단기(1일), 중기(7일), 장기(30일) 후의 가격을 예측합니다. [출처: Binance 일봉 1000일 + 기술지표 + 페어 상관관계 + 심리/온체인 데이터 기반 자체 ML 모델 학습 결과]" en="Ensemble prediction from 5 ML models. [Source: Trained on Binance daily candles, technical indicators, pair correlations, sentiment & onchain data]" /></h3>

      {/* Timeframe selector */}
      <div className="flex gap-1 mb-4">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.key}
            onClick={() => setTimeframe(tf.key)}
            className={`flex-1 text-xs py-1.5 rounded-md transition-colors ${
              timeframe === tf.key
                ? 'bg-accent text-white'
                : 'bg-primary/50 text-gray-400 hover:bg-accent/30'
            }`}
          >
            {tf.label}
          </button>
        ))}
      </div>

      {isLoading && <p className="text-xs text-gray-500 text-center py-4">로딩 중...</p>}
      {isError && <p className="text-xs text-danger text-center py-4">데이터를 불러올 수 없습니다.</p>}

      {data && (
        <div className="space-y-3">
          {/* All predictions from data.predictions */}
          {data.predictions && Object.keys(data.predictions).length > 0 ? (
            <>
              {/* Ensemble result first if present */}
              {data.predictions.ensemble && (
                <ModelCard name="ensemble" result={data.predictions.ensemble} isEnsemble />
              )}

              {/* Individual models */}
              {Object.entries(data.predictions)
                .filter(([name]) => name !== 'ensemble')
                .map(([name, result]) => (
                  <ModelCard key={name} name={name} result={result} isEnsemble={false} />
                ))}
            </>
          ) : (
            <p className="text-xs text-gray-500 text-center py-4">예측 데이터 수집 중...</p>
          )}

          {data.warning && (
            <p className="text-xs text-warning bg-warning/10 rounded p-2 mt-2">
              ⚠️ {data.warning}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
