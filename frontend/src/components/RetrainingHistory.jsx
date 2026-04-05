import { useRetrainingHistory } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

function StatusBadge({ replaced }) {
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
        replaced ? 'bg-success/20 text-success' : 'bg-gray-600/30 text-gray-400'
      }`}
    >
      {replaced ? '교체됨' : '유지'}
    </span>
  );
}

function MethodBadge({ method }) {
  const isIncremental = method === 'incremental';
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
        isIncremental ? 'bg-accent/40 text-blue-300' : 'bg-highlight/20 text-highlight'
      }`}
    >
      {isIncremental ? 'Incremental' : 'Full'}
    </span>
  );
}

function formatDate(ts) {
  if (!ts) return '-';
  return new Date(ts).toLocaleString('ko-KR', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });
}

function formatMetric(v) {
  if (v == null) return '-';
  return v.toFixed(4);
}

export default function RetrainingHistory() {
  const { data, isLoading, isError } = useRetrainingHistory();

  if (isLoading) return <div className="text-gray-500 text-sm">재학습 이력 로딩 중...</div>;
  if (isError) return <div className="text-danger text-sm">재학습 이력을 불러올 수 없습니다.</div>;

  const history = data || [];

  if (history.length === 0) {
    return (
      <div className="bg-secondary rounded-xl border border-gray-700 p-4">
        <h4 className="text-sm font-medium text-gray-300 mb-3">재학습 이력<InfoTooltip ko="모델이 새 데이터로 재학습된 이력입니다. Champion(현재 모델)과 Challenger(새 모델)의 성능을 비교하여 더 나은 모델로 자동 교체합니다." en="History of model retraining with new data. Compares Champion (current model) vs Challenger (new model) performance and automatically swaps if improved." /></h4>
        <p className="text-xs text-gray-500 text-center py-4">재학습 이력이 없습니다.</p>
      </div>
    );
  }

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4">
      <h4 className="text-sm font-medium text-gray-300 mb-3">재학습 이력<InfoTooltip ko="모델이 새 데이터로 재학습된 이력입니다. Champion(현재 모델)과 Challenger(새 모델)의 성능을 비교하여 더 나은 모델로 자동 교체합니다." en="History of model retraining with new data. Compares Champion (current model) vs Challenger (new model) performance and automatically swaps if improved." /></h4>
      <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700 sticky top-0 bg-secondary z-10">
              <th className="text-left py-2 px-2">일시</th>
              <th className="text-left py-2 px-2">모델</th>
              <th className="text-left py-2 px-2">방식</th>
              <th className="text-right py-2 px-2">Champion MAE</th>
              <th className="text-right py-2 px-2">Challenger MAE</th>
              <th className="text-right py-2 px-2">Champion 방향</th>
              <th className="text-right py-2 px-2">Challenger 방향</th>
              <th className="text-center py-2 px-2">결과</th>
            </tr>
          </thead>
          <tbody>
            {history.map((h, i) => (
              <tr key={i} className="border-b border-gray-800 hover:bg-accent/10">
                <td className="py-2 px-2 text-gray-300">{formatDate(h.timestamp)}</td>
                <td className="py-2 px-2 text-gray-300 font-medium">{h.model_name}</td>
                <td className="py-2 px-2"><MethodBadge method={h.method} /></td>
                <td className="py-2 px-2 text-right text-gray-400">{formatMetric(h.champion_mae)}</td>
                <td className="py-2 px-2 text-right text-gray-400">{formatMetric(h.challenger_mae)}</td>
                <td className="py-2 px-2 text-right text-gray-400">
                  {h.champion_direction_accuracy != null ? `${h.champion_direction_accuracy.toFixed(1)}%` : '-'}
                </td>
                <td className="py-2 px-2 text-right text-gray-400">
                  {h.challenger_direction_accuracy != null ? `${h.challenger_direction_accuracy.toFixed(1)}%` : '-'}
                </td>
                <td className="py-2 px-2 text-center"><StatusBadge replaced={h.replaced} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {history.some((h) => h.error_message) && (
        <div className="mt-3 space-y-1">
          {history
            .filter((h) => h.error_message)
            .map((h, i) => (
              <p key={i} className="text-xs text-danger bg-danger/10 rounded p-2">
                ⚠️ {h.model_name}: {h.error_message}
              </p>
            ))}
        </div>
      )}
    </div>
  );
}
