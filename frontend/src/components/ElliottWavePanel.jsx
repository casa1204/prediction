import { useElliottWave } from '../hooks/useQueries';
import InfoTooltip from './InfoTooltip';

function WaveTag({ number, type, active }) {
  const colors = {
    impulse: 'bg-success/20 text-success border-success/40',
    corrective: 'bg-warning/20 text-warning border-warning/40',
  };
  return (
    <span
      className={`inline-flex items-center px-2 py-1 rounded border text-xs font-medium ${
        colors[type] || 'bg-gray-700 text-gray-300 border-gray-600'
      } ${active ? 'ring-2 ring-highlight' : ''}`}
    >
      {number}파
    </span>
  );
}

function FibonacciTable({ targets }) {
  if (!targets || Object.keys(targets).length === 0) return null;

  const ratios = ['0.236', '0.382', '0.5', '0.618', '0.786'];
  return (
    <div className="mt-3">
      <p className="text-xs text-gray-400 mb-2">피보나치 되돌림 목표가</p>
      <div className="grid grid-cols-5 gap-2">
        {ratios.map((r) => (
          <div key={r} className="bg-primary/50 rounded p-2 text-center">
            <p className="text-xs text-gray-500">{(parseFloat(r) * 100).toFixed(1)}%</p>
            <p className="text-sm font-medium text-gray-200">
              ${targets[r] != null ? targets[r].toFixed(4) : '-'}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ElliottWavePanel() {
  const { data, isLoading, isError } = useElliottWave();

  if (isLoading) return <div className="text-gray-500 text-sm">엘리엇 파동 로딩 중...</div>;
  if (isError) return <div className="text-danger text-sm">엘리엇 파동 데이터를 불러올 수 없습니다.</div>;

  const waves = data?.waves || [];
  const position = data?.current_position;

  if (!position && waves.length === 0) {
    return (
      <div className="bg-secondary rounded-xl border border-gray-700 p-4">
        <h4 className="text-sm font-medium text-gray-300">엘리엇 파동 분석<InfoTooltip ko="엘리엇 파동 이론에 따라 가격 움직임을 5개의 충격파(1~5)와 3개의 조정파(A~C)로 분석합니다. 현재 어느 파동 단계에 있는지 파악하여 다음 가격 방향을 예측합니다." en="Analyzes price movements into 5 impulse waves (1-5) and 3 corrective waves (A-C) based on Elliott Wave Theory. Identifies current wave position to predict next price direction." /></h4>
        <p className="text-xs text-gray-500 text-center py-4">파동 데이터 수집 중...</p>
      </div>
    );
  }

  return (
    <div className="bg-secondary rounded-xl border border-gray-700 p-4 space-y-4">
      <h4 className="text-sm font-medium text-gray-300">엘리엇 파동 분석<InfoTooltip ko="엘리엇 파동 이론에 따라 가격 움직임을 5개의 충격파(1~5)와 3개의 조정파(A~C)로 분석합니다. 현재 어느 파동 단계에 있는지 파악하여 다음 가격 방향을 예측합니다." en="Analyzes price movements into 5 impulse waves (1-5) and 3 corrective waves (A-C) based on Elliott Wave Theory. Identifies current wave position to predict next price direction." /></h4>

      {/* Current position summary */}
      {position && (
        <div className="bg-accent/30 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-gray-400">현재 파동 위치</p>
              <p className="text-lg font-bold">
                {position.wave_number ?? '-'}파
                <span className="text-sm text-gray-400 ml-2">
                  ({position.wave_type || '-'})
                </span>
              </p>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-400">파동 유형</p>
              <p className={`text-lg font-bold ${
                position.wave_type === 'impulse' ? 'text-success' : 'text-warning'
              }`}>
                {position.wave_type === 'impulse' ? '충격파' : position.wave_type === 'corrective' ? '조정파' : position.wave_type || '-'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Wave count */}
      <div>
        <p className="text-xs text-gray-400 mb-2">파동 카운트</p>
        <div className="flex flex-wrap gap-2">
          {waves.length > 0 ? (
            waves.map((w, i) => (
              <WaveTag
                key={i}
                number={w.wave_number}
                type={w.wave_type}
                active={position && w.wave_number === position.wave_number}
              />
            ))
          ) : (
            <span className="text-xs text-gray-500">감지된 파동 없음</span>
          )}
        </div>
      </div>

      {/* Fibonacci targets from wave data */}
      {waves.length > 0 && waves[0]?.fibonacci_targets && (
        <FibonacciTable targets={waves[0].fibonacci_targets} />
      )}
    </div>
  );
}
