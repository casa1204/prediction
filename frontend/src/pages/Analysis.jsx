import AnalysisChart from '../components/AnalysisChart';
import TechnicalIndicators from '../components/TechnicalIndicators';
import ElliottWavePanel from '../components/ElliottWavePanel';
import WyckoffPanel from '../components/WyckoffPanel';

export default function Analysis() {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">기술 분석</h2>

      <AnalysisChart />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ElliottWavePanel />
        <WyckoffPanel />
      </div>

      <TechnicalIndicators />
    </div>
  );
}
