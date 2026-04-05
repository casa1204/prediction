import { useState } from 'react';
import { useAiAnalysis, useAiAnalysisHistory } from '../hooks/useQueries';

function MarkdownRenderer({ text }) {
  const html = text
    .replace(/^### (.+)$/gm, '<h3 class="text-lg font-semibold text-white mt-6 mb-2">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 class="text-xl font-bold text-accent mt-8 mb-3">$1</h2>')
    .replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold text-white mt-8 mb-4">$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-white">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^- (.+)$/gm, '<li class="ml-4 list-disc text-gray-300">$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li class="ml-4 list-decimal text-gray-300">$2</li>')
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>');
  return (
    <div
      className="prose prose-invert max-w-none text-gray-300 leading-relaxed"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function formatDate(ts) {
  if (!ts) return '';
  return new Date(ts).toLocaleString('ko-KR', {
    year: 'numeric', month: 'long', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

export default function AiAnalysis() {
  const [selectedId, setSelectedId] = useState(null);
  const { data: latestData, isLoading: latestLoading } = useAiAnalysis(true);
  const { data: historyData, isLoading: historyLoading } = useAiAnalysisHistory();

  const history = historyData || [];
  const selectedReport = selectedId
    ? history.find((r) => r.id === selectedId)
    : null;
  const displayReport = selectedReport || latestData;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white">🤖 AI 시장 분석</h2>
        <p className="text-sm text-gray-400 mt-1">
          재학습 완료 시 Gemini가 자동으로 종합 분석 리포트를 생성합니다
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* 왼쪽: 과거 리포트 목록 */}
        <div className="xl:col-span-1">
          <div className="bg-secondary rounded-xl border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3">📋 분석 이력</h3>
            {historyLoading ? (
              <p className="text-xs text-gray-500">로딩 중...</p>
            ) : history.length === 0 ? (
              <p className="text-xs text-gray-500">아직 분석 이력이 없습니다. 재학습 완료 후 자동 생성됩니다.</p>
            ) : (
              <div className="space-y-1 max-h-[600px] overflow-y-auto">
                {history.map((r) => (
                  <button
                    key={r.id}
                    onClick={() => setSelectedId(r.id === selectedId ? null : r.id)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors ${
                      r.id === selectedId
                        ? 'bg-accent text-white'
                        : 'text-gray-400 hover:bg-accent/20 hover:text-gray-200'
                    }`}
                  >
                    <p className="font-medium">{formatDate(r.timestamp)}</p>
                    <p className="text-gray-500 truncate mt-0.5">
                      {r.analysis?.substring(0, 60)}...
                    </p>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* 오른쪽: 분석 내용 */}
        <div className="xl:col-span-3">
          {latestLoading && !displayReport ? (
            <div className="bg-secondary rounded-xl border border-gray-700 p-12 text-center">
              <div className="animate-spin h-8 w-8 border-2 border-accent border-t-transparent rounded-full mx-auto mb-4" />
              <p className="text-gray-400">분석 리포트 로딩 중...</p>
            </div>
          ) : displayReport?.analysis ? (
            <div className="bg-secondary rounded-xl border border-gray-700 p-6">
              <div className="flex items-center gap-2 mb-4 pb-4 border-b border-gray-700">
                <span className="text-lg">📊</span>
                <span className="text-sm text-gray-400">
                  {displayReport.timestamp
                    ? formatDate(displayReport.timestamp)
                    : '최신 분석'}
                  {displayReport.model_used && (
                    <span className="ml-2 text-gray-600">({displayReport.model_used})</span>
                  )}
                </span>
                {selectedId && (
                  <button
                    onClick={() => setSelectedId(null)}
                    className="ml-auto text-xs text-accent hover:text-accent/80"
                  >
                    최신으로 돌아가기
                  </button>
                )}
              </div>
              <MarkdownRenderer text={displayReport.analysis} />
            </div>
          ) : (
            <div className="bg-secondary rounded-xl border border-gray-700 p-12 text-center">
              <span className="text-5xl mb-4 block">🔍</span>
              <p className="text-gray-400 text-lg">
                아직 분석 리포트가 없습니다
              </p>
              <p className="text-gray-500 text-sm mt-2">
                재학습이 완료되면 자동으로 AI 분석이 생성됩니다
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
