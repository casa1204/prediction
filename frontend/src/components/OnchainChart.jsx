import { useOnchain } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

function StatBox({ label, value, sub }) {
  return (
    <div className="bg-primary/50 rounded-lg p-3 text-center">
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-lg font-bold text-gray-200">{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
    </div>
  );
}

function formatNumber(n) {
  if (n == null) return '-';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}

export default function OnchainChart() {
  const { data, isLoading, isError } = useOnchain();

  if (isLoading) return <div className="text-gray-500 text-sm">온체인 데이터 로딩 중...</div>;
  if (isError) return <div className="text-danger text-sm">온체인 데이터를 불러올 수 없습니다.</div>;

  const onchain = data?.onchain;

  if (!onchain) {
    return (
      <div className="bg-secondary rounded-xl border border-gray-700 p-4">
        <h4 className="text-sm font-medium text-gray-300">🐋 고래 활동<InfoTooltip ko="XRPL 네트워크에서 10만 XRP 이상의 대규모 거래(고래 거래)를 추적합니다. 최근 100개 레저(약 5분)를 스캔합니다. 고래 활동 증가는 큰 가격 변동의 전조일 수 있습니다. [출처: XRPL JSON-RPC API (s1.ripple.com)]" en="Tracks large transactions (100K+ XRP) on XRPL. Scans last 100 ledgers (~5 min). [Source: XRPL JSON-RPC API (s1.ripple.com)]" /></h4>
        <p className="text-xs text-gray-500 text-center py-4">온체인 데이터 수집 중...</p>
      </div>
    );
  }

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4 space-y-4">
      <h4 className="text-sm font-medium text-gray-300">🐋 고래 활동 (최근 100 레저)<InfoTooltip ko="XRPL 네트워크에서 10만 XRP 이상의 대규모 거래(고래 거래)를 추적합니다. 최근 100개 레저(약 5분)를 스캔하여 집계합니다. [출처: XRPL JSON-RPC API (s1.ripple.com)]" en="Tracks large transactions (100K+ XRP) on XRPL. Scans last 100 ledgers (~5 min). [Source: XRPL JSON-RPC API (s1.ripple.com)]" /></h4>
      <div className="grid grid-cols-2 gap-3">
        <StatBox
          label="고래 거래 건수"
          value={formatNumber(onchain.whale_tx_count)}
          sub="100만 XRP 이상"
        />
        <StatBox
          label="고래 거래량"
          value={`${formatNumber(onchain.whale_tx_volume)} XRP`}
        />
      </div>
    </div>
  );
}
