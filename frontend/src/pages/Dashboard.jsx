import PriceOverview from '../components/PriceOverview';
import CandlestickChart from '../components/CandlestickChart';
import PredictionPanel from '../components/PredictionPanel';
import SentimentGauge from '../components/SentimentGauge';
import OnchainChart from '../components/OnchainChart';
import CorrelationChart from '../components/CorrelationChart';
import EtfPanel from '../components/EtfPanel';

export default function Dashboard() {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">메인 대시보드</h2>

      <div className="bg-yellow-900/20 border border-yellow-700/40 rounded-lg px-4 py-3 text-xs text-yellow-300/80">
        ⚠️ 본 대시보드의 모든 예측 및 분석 결과는 AI/ML 모델에 의해 자동 생성된 데이터이며, 재정적 조언이 아닙니다. 투자 결정의 참고 자료로만 활용하시고, 실제 투자는 본인의 판단과 책임 하에 진행하시기 바랍니다.
      </div>

      <PriceOverview />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2">
          <CandlestickChart />
        </div>
        <div>
          <PredictionPanel />
        </div>
      </div>

      <EtfPanel />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <SentimentGauge />
        <OnchainChart />
      </div>

      <CorrelationChart />
    </div>
  );
}
