import { useWyckoff } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip,
} from 'recharts';

const PHASE_COLORS = {
  Accumulation: 'text-success',
  Markup: 'text-success',
  Distribution: 'text-danger',
  Markdown: 'text-danger',
};

const EVENT_COLORS = {
  PS: '#9ca3af', SC: '#ff1744', AR: '#00c853', ST: '#ffd600',
  Spring: '#00c853', SOS: '#2196f3', LPS: '#4caf50',
  PSY: '#9ca3af', BC: '#00c853', UTAD: '#ff1744', LPSY: '#ff9800', SOW: '#ff1744',
};

function ConfidenceBar({ score }) {
  const pct = Math.min(100, Math.max(0, score || 0));
  const color = pct >= 70 ? 'bg-success' : pct >= 40 ? 'bg-warning' : 'bg-danger';
  return (
    <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
      <div className={`${color} h-2 rounded-full transition-all`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function EventTimeline({ events }) {
  if (!events || events.length === 0) {
    return <p className="text-xs text-gray-500">감지된 이벤트 없음</p>;
  }

  return (
    <div className="space-y-1 max-h-48 overflow-y-auto">
      {events.map((e, i) => (
        <div key={i} className="flex items-center gap-2 text-xs">
          <span
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{ backgroundColor: EVENT_COLORS[e.event_type] || '#6b7280' }}
          />
          <span className="text-gray-400 w-16 flex-shrink-0">
            {e.timestamp ? new Date(e.timestamp).toLocaleDateString('ko-KR') : '-'}
          </span>
          <span className="font-medium text-gray-200">{e.event_type}</span>
          <span className="text-gray-500">${(e.price ?? 0).toFixed(4)}</span>
          {e.is_trend_reversal && (
            <span className="text-highlight text-[10px] font-medium">전환</span>
          )}
        </div>
      ))}
    </div>
  );
}

function VolumeAnalysisChart({ volumeData }) {
  if (!volumeData || volumeData.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={volumeData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
        <XAxis dataKey="timestamp" tick={{ fontSize: 9, fill: '#9ca3af' }} />
        <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} />
        <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #374151' }} />
        <Bar dataKey="volume" fill="#0f3460" name="거래량" />
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function WyckoffPanel() {
  const { data, isLoading, isError } = useWyckoff();

  if (isLoading) return <div className="text-gray-500 text-sm">와이코프 분석 로딩 중...</div>;
  if (isError) return <div className="text-danger text-sm">와이코프 데이터를 불러올 수 없습니다.</div>;

  const marketPhase = data?.market_phase;
  const wyckoffPhase = data?.wyckoff_phase;
  const confidenceScore = data?.confidence_score;
  const events = data?.events || [];

  if (!marketPhase && events.length === 0) {
    return (
      <div className="bg-secondary rounded-xl border border-gray-700 p-4">
        <h4 className="text-sm font-medium text-gray-300">와이코프 패턴 분석<InfoTooltip ko="와이코프 방법론으로 시장의 축적(Accumulation), 마크업(Markup), 분배(Distribution), 마크다운(Markdown) 4단계를 분석합니다. 대형 기관(스마트 머니)의 매집/매도 패턴을 감지합니다." en="Analyzes 4 market phases (Accumulation, Markup, Distribution, Markdown) using Wyckoff methodology. Detects institutional (smart money) accumulation and distribution patterns." /></h4>
        <p className="text-xs text-gray-500 text-center py-4">와이코프 데이터 수집 중...</p>
      </div>
    );
  }

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4 space-y-4">
      <h4 className="text-sm font-medium text-gray-300">와이코프 패턴 분석<InfoTooltip ko="와이코프 방법론으로 시장의 축적(Accumulation), 마크업(Markup), 분배(Distribution), 마크다운(Markdown) 4단계를 분석합니다. 대형 기관(스마트 머니)의 매집/매도 패턴을 감지합니다." en="Analyzes 4 market phases (Accumulation, Markup, Distribution, Markdown) using Wyckoff methodology. Detects institutional (smart money) accumulation and distribution patterns." /></h4>

      {/* Market phase + Wyckoff phase summary */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-accent/30 rounded-lg p-3">
          <p className="text-xs text-gray-400">시장 단계</p>
          <p className={`text-lg font-bold ${PHASE_COLORS[marketPhase] || 'text-gray-200'}`}>
            {marketPhase || '-'}
          </p>
          <div className="mt-1">
            <p className="text-xs text-gray-400">신뢰도 {(confidenceScore ?? 0).toFixed(0)}%</p>
            <ConfidenceBar score={confidenceScore} />
          </div>
        </div>
        <div className="bg-accent/30 rounded-lg p-3">
          <p className="text-xs text-gray-400">Wyckoff Phase</p>
          <p className="text-lg font-bold text-gray-200">{wyckoffPhase || '-'}</p>
        </div>
      </div>

      {/* Event timeline */}
      <div>
        <p className="text-xs text-gray-400 mb-2">이벤트 타임라인</p>
        <EventTimeline events={events} />
      </div>

      {/* Volume analysis from events */}
      {events.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 mb-2">거래량 분석</p>
          <VolumeAnalysisChart volumeData={events.filter((e) => e.volume != null).map((e) => ({
            timestamp: e.timestamp ? new Date(e.timestamp).toLocaleDateString('ko-KR') : '-',
            volume: e.volume,
          }))} />
        </div>
      )}
    </div>
  );
}
