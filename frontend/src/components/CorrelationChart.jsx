import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, ReferenceLine, LabelList,
} from 'recharts';
import { useCorrelations } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

const COLORS = ['#e94560', '#00c853', '#ffd600', '#2196f3', '#ff9800', '#9c27b0', '#00bcd4'];

export default function CorrelationChart() {
  const { data, isLoading, isError } = useCorrelations();

  if (isLoading) return <div className="text-gray-500 text-sm">상관계수 로딩 중...</div>;
  if (isError) return <div className="text-danger text-sm">상관계수 데이터를 불러올 수 없습니다.</div>;

  const correlations = data?.correlations || {};
  // XRP 포함 페어는 자기 자신과의 상관이라 제외
  const chartData = Object.entries(correlations)
    .filter(([name]) => !name.startsWith('XRP/'))
    .map(([assetName, c]) => ({
      asset: assetName,
      correlation: c?.correlation_with_xrp ?? 0,
    }));

  if (chartData.length === 0) {
    return (
      <div className="bg-secondary rounded-xl border border-gray-700 p-4">
        <h4 className="text-sm font-medium text-gray-300 mb-3">상관 자산 상관계수<InfoTooltip ko="XRP와 다른 자산의 90일 롤링 가격 상관계수입니다. +1에 가까우면 같은 방향, -1에 가까우면 반대 방향으로 움직입니다. [출처: 암호화폐 - Binance Klines API / 지수(S&P500, NASDAQ, DXY, VIX) - Yahoo Finance / 상관계수 - 90일 롤링 피어슨 상관계수 자체 계산]" en="90-day rolling correlation between XRP and other assets. [Source: Crypto - Binance / Indices - Yahoo Finance / Correlation - calculated internally]" /></h4>
        <p className="text-xs text-gray-500 text-center py-4">상관계수 데이터 수집 중...</p>
      </div>
    );
  }

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4">
      <h4 className="text-sm font-medium text-gray-300 mb-3">상관 자산 상관계수<InfoTooltip ko="XRP와 다른 자산의 90일 롤링 가격 상관계수입니다. +1에 가까우면 같은 방향, -1에 가까우면 반대 방향으로 움직입니다. [출처: 암호화폐 - Binance Klines API / 지수(S&P500, NASDAQ, DXY, VIX) - Yahoo Finance / 상관계수 - 90일 롤링 피어슨 상관계수 자체 계산]" en="90-day rolling correlation between XRP and other assets. [Source: Crypto - Binance / Indices - Yahoo Finance / Correlation - calculated internally]" /></h4>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={chartData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
          <XAxis type="number" domain={[-1, 1]} tick={{ fontSize: 10, fill: '#9ca3af' }} />
          <YAxis type="category" dataKey="asset" tick={{ fontSize: 10, fill: '#9ca3af' }} width={80} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #374151' }}
            formatter={(v) => [v.toFixed(4), '상관계수']}
          />
          <ReferenceLine x={0} stroke="#6b7280" />
          <Bar dataKey="correlation" name="상관계수" radius={[0, 4, 4, 0]}>
            <LabelList dataKey="correlation" position="right" fill="#e5e7eb" fontSize={11} formatter={(v) => v.toFixed(2)} />
            {chartData.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
