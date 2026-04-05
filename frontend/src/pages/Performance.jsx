import HitRateChart from '../components/HitRateChart';
import FeatureImportance from '../components/FeatureImportance';
import RetrainingHistory from '../components/RetrainingHistory';

export default function Performance() {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">성능 모니터링</h2>

      <HitRateChart />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <FeatureImportance />
        <RetrainingHistory />
      </div>
    </div>
  );
}
