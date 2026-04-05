import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell,
} from 'recharts';
import { useHitRates } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

const MODEL_COLORS = {
  lstm: '#e94560',
  xgboost: '#00c853',
  rf: '#ffd600',
  transformer: '#2196f3',
  ensemble: '#ff9800',
};

const MODEL_LABELS = {
  lstm: 'LSTM',
  xgboost: 'XGBoost',
  rf: 'Random Forest',
  transformer: 'Transformer',
  ensemble: '앙상블',
};

function TimeSeriesChart({ history }) {
  if (!history || history.length === 0) {
    return <p className="text-xs text-gray-500 text-center py-4">시계열 데이터 없음</p>;
  }

  const models = [...new Set(history.map((h) => h.model_name))];

  // Pivot data by date
  const dateMap = {};
  history.forEach((h) => {
    const date = h.date?.split('T')[0] || h.date;
    if (!dateMap[date]) dateMap[date] = { date };
    dateMap[date][`${h.model_name}_dir`] = h.direction_hit_rate;
  });
  const chartData = Object.values(dateMap).sort((a, b) => a.date.localeCompare(b.date));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
        <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#9ca3af' }} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#9ca3af' }} />
        <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #374151' }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {models.map((m) => (
          <Line
            key={m}
            type="monotone"
            dataKey={`${m}_dir`}
            stroke={MODEL_COLORS[m] || '#6b7280'}
            dot={false}
            strokeWidth={m === 'ensemble' ? 3 : 1.5}
            name={`${MODEL_LABELS[m] || m} 방향`}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function ComparisonBarChart({ latest }) {
  if (!latest || latest.length === 0) {
    return <p className="text-xs text-gray-500 text-center py-4">비교 데이터 없음</p>;
  }

  const chartData = latest.map((l) => ({
    model: MODEL_LABELS[l.model_name] || l.model_name,
    direction: l.direction_hit_rate ?? 0,
    range: l.range_hit_rate ?? 0,
    name: l.model_name,
  }));

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
        <XAxis dataKey="model" tick={{ fontSize: 10, fill: '#9ca3af' }} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#9ca3af' }} />
        <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #374151' }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="direction" name="방향 적중률" radius={[4, 4, 0, 0]}>
          {chartData.map((d, i) => (
            <Cell key={i} fill={MODEL_COLORS[d.name] || '#6b7280'} />
          ))}
        </Bar>
        <Bar dataKey="range" name="범위 적중률" fill="#0f3460" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function HitRateChart() {
  const { data, isLoading, isError } = useHitRates();

  if (isLoading) return <div className="text-gray-500 text-sm">적중률 로딩 중...</div>;
  if (isError) return <div className="text-danger text-sm">적중률 데이터를 불러올 수 없습니다.</div>;

  const hitRates = data?.hit_rates || [];

  // Derive history and latest from the flat hit_rates array
  const history = hitRates;

  // Get latest record per model
  const latestMap = {};
  hitRates.forEach((h) => {
    if (!latestMap[h.model_name] || h.date > latestMap[h.model_name].date) {
      latestMap[h.model_name] = h;
    }
  });
  const latest = Object.values(latestMap);

  if (hitRates.length === 0) {
    return (
      <div className="bg-secondary rounded-xl border border-gray-700 p-4">
        <h4 className="text-sm font-medium text-gray-300 mb-3">적중률<InfoTooltip ko="예측 모델의 정확도를 추적합니다. 방향 적중률은 상승/하락 예측이 맞은 비율, 범위 적중률은 예측 가격이 실제 가격의 허용 오차 내에 있었던 비율입니다." en="Tracks prediction model accuracy. Direction hit rate measures correct up/down predictions. Range hit rate measures predictions within tolerance (±3% short, ±5% mid, ±10% long)." /></h4>
        <p className="text-xs text-gray-500 text-center py-4">적중률 데이터 수집 중...</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-secondary rounded-xl border border-gray-700 p-4">
        <h4 className="text-sm font-medium text-gray-300 mb-3">적중률 시계열 추이<InfoTooltip ko="예측 모델의 정확도를 추적합니다. 방향 적중률은 상승/하락 예측이 맞은 비율, 범위 적중률은 예측 가격이 실제 가격의 허용 오차 내에 있었던 비율입니다." en="Tracks prediction model accuracy. Direction hit rate measures correct up/down predictions. Range hit rate measures predictions within tolerance (±3% short, ±5% mid, ±10% long)." /></h4>
        <TimeSeriesChart history={history} />
      </div>
      <div className="bg-secondary rounded-xl border border-gray-700 p-4">
        <h4 className="text-sm font-medium text-gray-300 mb-3">모델 간 적중률 비교<InfoTooltip ko="예측 모델의 정확도를 추적합니다. 방향 적중률은 상승/하락 예측이 맞은 비율, 범위 적중률은 예측 가격이 실제 가격의 허용 오차 내에 있었던 비율입니다." en="Tracks prediction model accuracy. Direction hit rate measures correct up/down predictions. Range hit rate measures predictions within tolerance (±3% short, ±5% mid, ±10% long)." /></h4>
        <ComparisonBarChart latest={latest} />
      </div>
    </div>
  );
}
