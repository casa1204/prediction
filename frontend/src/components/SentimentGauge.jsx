import { useSentiment } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

function GaugeCard({ label, value, icon, tooltip }) {
  const pct = Math.min(100, Math.max(0, value ?? 0));
  let color = 'bg-gray-500';
  let textColor = 'text-gray-300';
  if (pct >= 75) { color = 'bg-success'; textColor = 'text-success'; }
  else if (pct >= 50) { color = 'bg-warning'; textColor = 'text-warning'; }
  else if (pct >= 25) { color = 'bg-orange-500'; textColor = 'text-orange-400'; }
  else { color = 'bg-danger'; textColor = 'text-danger'; }

  const sentiment =
    pct >= 75 ? '극도의 탐욕' :
    pct >= 60 ? '탐욕' :
    pct >= 40 ? '중립' :
    pct >= 25 ? '공포' : '극도의 공포';

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4 flex flex-col items-center">
      <span className="text-2xl mb-1">{icon}</span>
      <p className="text-xs text-gray-400 mb-2">{label}{tooltip && <InfoTooltip ko={tooltip.ko} en={tooltip.en} />}</p>
      <div className="relative w-20 h-20 mb-2">
        <svg viewBox="0 0 36 36" className="w-full h-full">
          <path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="#374151"
            strokeWidth="3"
          />
          <path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeDasharray={`${pct}, 100`}
            className={textColor}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`text-lg font-bold ${textColor}`}>{pct.toFixed(0)}</span>
        </div>
      </div>
      <p className={`text-xs font-medium ${textColor}`}>{sentiment}</p>
    </div>
  );
}

export default function SentimentGauge() {
  const { data, isLoading, isError } = useSentiment();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="bg-secondary rounded-xl border border-gray-700 p-4 animate-pulse h-40" />
        ))}
      </div>
    );
  }

  if (isError) return <div className="text-danger text-sm">심리 데이터를 불러올 수 없습니다.</div>;

  const sentiment = data?.sentiment;

  if (!sentiment) {
    return <div className="text-gray-500 text-sm text-center py-4">심리 데이터 수집 중...</div>;
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      <GaugeCard
        label="공포탐욕지수"
        value={sentiment.fear_greed_index}
        icon="😱"
        tooltip={{ ko: "시장 참여자들의 심리를 0(극도의 공포)~100(극도의 탐욕)으로 나타냅니다. 극도의 공포는 매수 기회, 극도의 탐욕은 매도 기회일 수 있습니다. [출처: Alternative.me Crypto Fear & Greed Index API]", en: "Measures market sentiment from 0 (Extreme Fear) to 100 (Extreme Greed). [Source: Alternative.me Crypto Fear & Greed Index API]" }}
      />
      <GaugeCard
        label="XRP 검색 트렌드"
        value={sentiment.google_trend_score}
        icon="🔍"
        tooltip={{ ko: "Google에서 'XRP', '리플', 'Ripple' 키워드의 검색 관심도입니다. 검색량 급증은 가격 변동의 선행 지표가 될 수 있습니다. [출처: Google Trends (pytrends)]", en: "Google search interest for 'XRP', 'Ripple' keywords. [Source: Google Trends via pytrends]" }}
      />
      <GaugeCard
        label="거시경제 트렌드"
        value={sentiment.trend_macro_score}
        icon="🏛️"
        tooltip={{ ko: "'Fed Interest Rate', 'Inflation', 'CPI', 'Recession', 'Rate Cut' 키워드의 검색 관심도입니다. 거시경제 이슈에 대한 대중의 관심을 반영합니다. [출처: Google Trends (pytrends)]", en: "Search interest for macroeconomic keywords. [Source: Google Trends via pytrends]" }}
      />
      <GaugeCard
        label="ETF/규제 트렌드"
        value={sentiment.trend_etf_regulatory_score}
        icon="📋"
        tooltip={{ ko: "'XRP ETF', 'Spot ETF', 'RLUSD', 'SEC', 'Crypto Regulation' 키워드의 검색 관심도입니다. ETF 승인 및 규제 관련 관심을 반영합니다. [출처: Google Trends (pytrends)]", en: "Search interest for ETF/regulatory keywords. [Source: Google Trends via pytrends]" }}
      />
      <GaugeCard
        label="매수/매도 심리"
        value={sentiment.trend_sentiment_score}
        icon="💰"
        tooltip={{ ko: "'Buy Crypto', 'Sell Crypto', 'Bitcoin', 'Crypto Scam' 키워드의 검색 관심도입니다. 대중의 매수/매도 심리를 반영합니다. [출처: Google Trends (pytrends)]", en: "Search interest for buying/selling sentiment keywords. [Source: Google Trends via pytrends]" }}
      />
      {sentiment.fomo_spread != null && (
        <div className="bg-secondary rounded-xl border border-gray-700 p-4 flex flex-col items-center justify-center">
          <span className="text-2xl mb-1">📊</span>
          <p className="text-xs text-gray-400 mb-2">FOMO 스프레드
            <InfoTooltip
              ko="'Buy Crypto' 검색량 - 'Sell Crypto' 검색량. 양수면 매수 심리 우세, 음수면 매도 심리 우세입니다. [출처: Google Trends 데이터 기반 자체 계산]"
              en="'Buy Crypto' minus 'Sell Crypto' search volume. [Source: Calculated from Google Trends data]"
            />
          </p>
          <span className={`text-2xl font-bold ${sentiment.fomo_spread >= 0 ? 'text-success' : 'text-danger'}`}>
            {sentiment.fomo_spread >= 0 ? '+' : ''}{sentiment.fomo_spread?.toFixed(1)}
          </span>
          <p className={`text-xs mt-1 ${sentiment.fomo_spread >= 0 ? 'text-success' : 'text-danger'}`}>
            {sentiment.fomo_spread >= 0 ? '매수 심리 우세' : '매도 심리 우세'}
          </p>
        </div>
      )}
    </div>
  );
}
