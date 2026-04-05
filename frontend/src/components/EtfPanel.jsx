import { useEtf } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

export default function EtfPanel() {
  const { data, isLoading, error } = useEtf();

  if (isLoading) return <div className="bg-gray-800 rounded-xl p-4 text-gray-400">ETF 데이터 로딩 중...</div>;
  if (error) return <div className="bg-gray-800 rounded-xl p-4 text-red-400">ETF 데이터 로드 실패</div>;

  const etfs = data?.etfs || [];

  if (etfs.length === 0) {
    return (
      <div className="bg-gray-800 rounded-xl p-4">
        <div className="flex items-center mb-3">
          <h3 className="text-lg font-semibold text-white">XRP ETF</h3>
          <InfoTooltip
            ko="XRP 현물 ETF의 가격과 거래량입니다. 프리미엄은 ETF 가격이 XRP 현물보다 높은 비율, 디스카운트는 낮은 비율입니다. 프리미엄 확대는 기관 수요 증가를 의미할 수 있습니다. [출처: Yahoo Finance (yfinance) / 프리미엄 - ETF 가격과 XRP 현물가 비교 자체 계산]"
            en="XRP spot ETF prices and volumes. Premium/discount calculated vs XRP spot. [Source: Yahoo Finance (yfinance) / Premium - calculated internally]"
          />
        </div>
        <p className="text-gray-500 text-sm">ETF 데이터가 없습니다.</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <div className="flex items-center mb-4">
        <h3 className="text-lg font-semibold text-white">XRP ETF</h3>
        <InfoTooltip
          ko="XRP 현물 ETF의 가격과 거래량입니다. 프리미엄은 ETF 가격이 XRP 현물보다 높은 비율, 디스카운트는 낮은 비율입니다. 프리미엄 확대는 기관 수요 증가를 의미할 수 있습니다. [출처: Yahoo Finance (yfinance) / 프리미엄 - ETF 가격과 XRP 현물가 비교 자체 계산]"
          en="XRP spot ETF prices and volumes. Premium/discount calculated vs XRP spot. [Source: Yahoo Finance (yfinance) / Premium - calculated internally]"
        />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {etfs.map((etf) => (
          <EtfCard key={etf.ticker} etf={etf} />
        ))}
      </div>
    </div>
  );
}

function EtfCard({ etf }) {
  const pd = etf.premium_discount;
  const isPremium = pd != null && pd >= 0;
  const pdColor = pd == null ? 'text-gray-400' : isPremium ? 'text-green-400' : 'text-red-400';
  const pdLabel = pd == null ? 'N/A' : `${pd >= 0 ? '+' : ''}${pd.toFixed(2)}%`;

  const volumeStr = etf.volume != null
    ? etf.volume >= 1_000_000
      ? `${(etf.volume / 1_000_000).toFixed(1)}M`
      : etf.volume >= 1_000
        ? `${(etf.volume / 1_000).toFixed(1)}K`
        : etf.volume.toLocaleString()
    : 'N/A';

  return (
    <div className="bg-gray-700/50 rounded-lg p-3 border border-gray-600/50">
      <div className="flex justify-between items-start mb-2">
        <div>
          <span className="text-white font-bold text-sm">{etf.ticker}</span>
          <p className="text-gray-400 text-xs mt-0.5 truncate" title={etf.name}>{etf.name}</p>
        </div>
        <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${isPremium ? 'bg-green-900/40 text-green-400' : pd != null ? 'bg-red-900/40 text-red-400' : 'bg-gray-600 text-gray-400'}`}>
          {pdLabel}
        </span>
      </div>
      <div className="flex justify-between items-end">
        <div>
          <p className="text-white text-sm font-medium">${etf.price != null ? etf.price.toFixed(2) : 'N/A'}</p>
          <p className="text-gray-400 text-xs">Vol: {volumeStr}</p>
        </div>
        <span className={`text-xs ${pdColor}`}>
          {pd != null ? (isPremium ? 'Premium' : 'Discount') : ''}
        </span>
      </div>
    </div>
  );
}
