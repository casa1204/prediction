import useBinanceWs from '../hooks/useBinanceWs';
import InfoTooltip from './InfoTooltip';

function StatCard({ label, value, sub, color, tooltip }) {
  return (
    <div className="bg-secondary rounded-xl p-4 border border-gray-700">
      <p className="text-xs text-gray-400 mb-1">{label}{tooltip && <InfoTooltip ko={tooltip.ko} en={tooltip.en} />}</p>
      <p className={`text-xl font-bold ${color || 'text-white'}`}>{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
    </div>
  );
}

export default function PriceOverview() {
  const { price, change24h, volume24h, high24h, low24h, connected } = useBinanceWs();

  if (price == null) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-secondary rounded-xl p-4 border border-gray-700 animate-pulse h-24" />
        ))}
      </div>
    );
  }

  const isPositive = (change24h ?? 0) >= 0;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
      <StatCard
        label="현재 가격"
        value={`${price.toFixed(4)}`}
        sub={connected ? '🟢 실시간' : '🔴 오프라인'}
        tooltip={{ ko: "바이낸스 WebSocket에서 실시간 수신하는 XRP/USDT 가격입니다. [출처: Binance WebSocket API]", en: "Real-time XRP/USDT price from Binance WebSocket. [Source: Binance WebSocket API]" }}
      />
      <StatCard
        label="24시간 변동률"
        value={`${isPositive ? '+' : ''}${(change24h ?? 0).toFixed(2)}%`}
        color={isPositive ? 'text-success' : 'text-danger'}
        tooltip={{ ko: "최근 24시간 동안의 가격 변동 비율입니다. [출처: Binance WebSocket API]", en: "Price change percentage over the last 24 hours. [Source: Binance WebSocket API]" }}
      />
      <StatCard
        label="24시간 고/저"
        value={`${(high24h ?? 0).toFixed(4)}`}
        sub={`저가: ${(low24h ?? 0).toFixed(4)}`}
        tooltip={{ ko: "최근 24시간 동안의 최고가와 최저가입니다. [출처: Binance WebSocket API]", en: "Highest and lowest prices in the last 24 hours. [Source: Binance WebSocket API]" }}
      />
      <StatCard
        label="24시간 거래대금"
        value={(volume24h ?? 0) >= 1e9
          ? `${((volume24h ?? 0) / 1e9).toFixed(2)}B`
          : `${((volume24h ?? 0) / 1e6).toFixed(2)}M`}
        sub="USDT"
        tooltip={{ ko: "최근 24시간 동안의 총 거래 금액(USDT)입니다. [출처: Binance WebSocket API]", en: "Total trading volume in USDT over the last 24 hours. [Source: Binance WebSocket API]" }}
      />
    </div>
  );
}
