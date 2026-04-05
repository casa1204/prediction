# 요구사항 문서

## 소개

XRP 암호화폐의 가격을 단기, 중기, 장기로 예측하는 대시보드 시스템이다. 기술적 지표, 엘리엇 파동 분석, 와이코프 패턴 분석, 상관 자산 페어(XRP/BTC, XRP/ETH, XRP/USD, BTC/USD, ETH/USD, S&P500, 나스닥), 소셜/검색 트렌드, 공포탐욕지수, 온체인 데이터(지갑 활동, 거래량) 등 다양한 학습 데이터를 활용하여 여러 머신러닝 모델(LSTM, XGBoost, Random Forest, Transformer)을 독립적으로 학습시키고, 앙상블 방식으로 통합하여 최종 가격 예측 결과를 시각적으로 표시한다.

## 용어 정의

- **Dashboard**: XRP 가격 예측 결과와 관련 데이터를 시각적으로 표시하는 웹 기반 사용자 인터페이스
- **Prediction_Engine**: 수집된 데이터를 기반으로 XRP 가격을 예측하는 머신러닝 모델 시스템
- **Data_Collector**: 외부 API 및 데이터 소스로부터 학습 데이터를 수집하는 모듈
- **Technical_Indicator_Module**: 차트 기술 지표(RSI, MACD, 볼린저밴드, 이동평균선 등)를 계산하는 모듈
- **Pair_Data_Module**: XRP/BTC, XRP/ETH, XRP/USD, BTC/USD, ETH/USD, S&P500, 나스닥지수 등 상관 자산의 가격 데이터를 수집·관리하는 모듈
- **Sentiment_Module**: 구글 검색 트렌드, SNS 검색량, 공포탐욕지수 등 시장 심리 데이터를 수집·관리하는 모듈
- **Onchain_Module**: XRP 지갑 활동, 거래량 등 온체인 데이터를 수집·관리하는 모듈
- **Short_Term_Prediction**: 1일~7일 범위의 단기 가격 예측
- **Mid_Term_Prediction**: 1주~1개월 범위의 중기 가격 예측
- **Long_Term_Prediction**: 1개월~3개월 범위의 장기 가격 예측
- **Fear_Greed_Index**: 시장의 공포와 탐욕 수준을 0~100 사이 수치로 나타내는 지표
- **Elliott_Wave_Module**: 엘리엇 파동 이론에 기반하여 가격 차트의 파동 패턴(충격파 5파, 조정파 3파)을 감지하고 분석하는 모듈
- **Impulse_Wave**: 엘리엇 파동의 충격파(1~5파)로, 주 추세 방향으로 진행하는 5개의 파동 구조
- **Corrective_Wave**: 엘리엇 파동의 조정파(A-B-C)로, 주 추세의 반대 방향으로 진행하는 3개의 파동 구조
- **Wave_Degree**: 엘리엇 파동의 시간 프레임별 등급(Grand Supercycle, Supercycle, Cycle, Primary, Intermediate, Minor, Minute, Minuette, Sub-Minuette)
- **Wyckoff_Module**: 와이코프 방법론에 기반하여 가격 차트의 축적(Accumulation) 및 분배(Distribution) 패턴을 감지하고 현재 시장 단계를 분석하는 모듈
- **Accumulation_Phase**: 와이코프 이론에서 대형 기관(스마트 머니)이 저가에 자산을 매집하는 단계로, PS → SC → AR → ST → Spring → SOS → LPS 순서로 진행
- **Distribution_Phase**: 와이코프 이론에서 대형 기관이 고가에 자산을 매도하는 단계로, PSY → BC → AR → ST → UTAD → LPSY → SOW 순서로 진행
- **Markup_Phase**: 와이코프 이론에서 축적 단계 이후 가격이 상승 추세로 전환되는 단계
- **Markdown_Phase**: 와이코프 이론에서 분배 단계 이후 가격이 하락 추세로 전환되는 단계
- **Spring**: 와이코프 축적 패턴에서 지지선 아래로 일시적으로 하락한 후 빠르게 반등하는 이벤트로, 약세 트레이더를 함정에 빠뜨리는 역할
- **Upthrust**: 와이코프 분배 패턴에서 저항선 위로 일시적으로 상승한 후 빠르게 하락하는 이벤트로, 강세 트레이더를 함정에 빠뜨리는 역할
- **Wyckoff_Phase**: 와이코프 패턴 내의 세부 단계(Phase A: 기존 추세 정지, Phase B: 원인 구축, Phase C: 테스트, Phase D: 추세 전환 확인, Phase E: 새로운 추세 진행)
- **Ensemble_Engine**: 여러 개별 학습 모델(LSTM, XGBoost, Random Forest, Transformer 등)의 예측 결과를 가중 평균, 투표 등의 방식으로 통합하여 최종 예측을 산출하는 앙상블 시스템
- **Individual_Model**: Ensemble_Engine을 구성하는 개별 학습 모델로, 각각 독립적으로 학습되어 예측 결과를 산출한다
- **Ensemble_Weight**: 앙상블 통합 시 각 Individual_Model에 부여되는 가중치로, 모델의 과거 정확도에 기반하여 동적으로 조정된다
- **Hit_Rate**: 예측 결과와 실제 결과를 비교하여 산출하는 적중률로, 방향 적중률(Direction_Hit_Rate)과 범위 적중률(Range_Hit_Rate)로 구분된다
- **Direction_Hit_Rate**: 예측한 가격 방향(상승/하락)이 실제 가격 방향과 일치한 비율(0~100%)
- **Range_Hit_Rate**: 예측 가격이 실제 가격의 허용 오차 범위(단기: ±3%, 중기: ±5%, 장기: ±10%) 내에 있었던 비율(0~100%)
- **Hit_Rate_Tracker**: 각 Individual_Model과 Ensemble_Engine의 적중률을 타임프레임별로 시계열 추적하고 이력을 관리하는 모듈
- **Retraining_Scheduler**: 새로운 데이터가 축적되면 주기적으로 모델 재학습을 수행하고, 재학습 전후 성능을 비교하여 모델 교체 여부를 결정하는 모듈
- **Incremental_Learning**: 기존 학습된 모델에 새로운 데이터만 추가하여 학습하는 증분 학습 방식
- **Full_Retraining**: 전체 학습 데이터셋을 사용하여 모델을 처음부터 다시 학습하는 전체 재학습 방식
- **Champion_Model**: 현재 운영 중인 모델로, 새로 학습된 Challenger_Model과 성능 비교 후 교체 여부가 결정된다
- **Challenger_Model**: 재학습을 통해 새로 생성된 모델로, Champion_Model과 성능 비교를 거쳐 교체 여부가 결정된다

## 요구사항

### 요구사항 1: 기술 지표 데이터 수집 및 계산

**사용자 스토리:** 데이터 분석가로서, XRP 차트의 다양한 기술 지표를 자동으로 계산하고 싶다. 이를 통해 가격 예측 모델의 학습 피처로 활용할 수 있다.

#### 인수 조건

1. WHEN XRP 가격 데이터가 수집되면, THE Technical_Indicator_Module SHALL RSI(14일), MACD(12,26,9), 볼린저밴드(20일), SMA(5,10,20,50,200일), EMA(12,26일) 지표를 계산한다
2. WHEN 새로운 캔들 데이터가 도착하면, THE Technical_Indicator_Module SHALL 1분 이내에 모든 기술 지표를 갱신한다
3. IF 가격 데이터에 결측값이 존재하면, THEN THE Technical_Indicator_Module SHALL 해당 구간을 보간법으로 처리하고 로그에 기록한다
4. THE Technical_Indicator_Module SHALL 계산된 모든 지표를 타임스탬프와 함께 구조화된 형식으로 저장한다
5. WHEN 충분한 가격 이력 데이터(최소 200개 캔들)가 존재하면, THE Elliott_Wave_Module SHALL 현재 차트에서 Impulse_Wave(1~5파)와 Corrective_Wave(A-B-C) 패턴을 감지한다
6. THE Elliott_Wave_Module SHALL 감지된 각 파동에 대해 파동 번호, 시작 가격, 종료 가격, 시작 시간, 종료 시간, Wave_Degree를 산출한다
7. WHEN 새로운 캔들 데이터가 도착하면, THE Elliott_Wave_Module SHALL 현재 진행 중인 파동의 위치(몇 번째 파동인지)와 예상 다음 파동 방향(상승/하락)을 갱신한다
8. THE Elliott_Wave_Module SHALL 피보나치 되돌림(0.236, 0.382, 0.5, 0.618, 0.786) 비율을 활용하여 각 파동의 목표 가격 수준을 계산한다
9. IF 감지된 파동 패턴이 엘리엇 파동 규칙(2파는 1파 시작점 아래로 내려가지 않음, 3파는 가장 짧은 충격파가 아님, 4파는 1파 영역과 겹치지 않음)을 위반하면, THEN THE Elliott_Wave_Module SHALL 해당 패턴을 무효로 표시하고 대안 카운트를 제시한다
10. WHEN 충분한 가격 이력 데이터(최소 200개 캔들)가 존재하면, THE Wyckoff_Module SHALL 축적(Accumulation) 패턴의 세부 이벤트(PS, SC, AR, ST, Spring, SOS, LPS)를 감지한다
11. WHEN 충분한 가격 이력 데이터(최소 200개 캔들)가 존재하면, THE Wyckoff_Module SHALL 분배(Distribution) 패턴의 세부 이벤트(PSY, BC, AR, ST, UTAD, LPSY, SOW)를 감지한다
12. THE Wyckoff_Module SHALL 현재 시장 단계를 Accumulation_Phase, Markup_Phase, Distribution_Phase, Markdown_Phase 중 하나로 판별하고 해당 단계의 신뢰도 점수(0~100)를 산출한다
13. THE Wyckoff_Module SHALL 현재 Wyckoff_Phase(Phase A~E)를 판별하고 각 Phase의 진행 상태를 산출한다
14. WHEN 새로운 캔들 데이터가 도착하면, THE Wyckoff_Module SHALL 거래량 분석을 수행하여 가격 움직임과 거래량 간의 확산/수렴 관계를 확인하고 패턴 판별의 근거로 활용한다
15. IF 감지된 와이코프 패턴에서 Spring 또는 Upthrust 이벤트가 발생하면, THEN THE Wyckoff_Module SHALL 해당 이벤트의 발생 시간, 가격, 거래량을 기록하고 추세 전환 신호로 표시한다
16. THE Wyckoff_Module SHALL 감지된 모든 패턴 이벤트를 타임스탬프, 가격, 거래량, 이벤트 유형과 함께 구조화된 형식으로 저장한다

### 요구사항 2: 상관 자산 페어 데이터 수집

**사용자 스토리:** 데이터 분석가로서, XRP/BTC, XRP/ETH, XRP/USD, BTC/USD, ETH/USD, S&P500, 나스닥지수 등 상관 자산의 가격 데이터를 수집하고 싶다. 이를 통해 자산 간 상관관계를 학습 데이터로 활용할 수 있다.

#### 인수 조건

1. THE Pair_Data_Module SHALL XRP/BTC, XRP/ETH, XRP/USD, BTC/USD, ETH/USD, S&P500, 나스닥 종합지수의 가격 데이터를 수집한다
2. WHEN 암호화폐 페어 데이터를 수집할 때, THE Pair_Data_Module SHALL 1시간 간격으로 최신 가격 데이터를 갱신한다
3. WHEN 전통 금융 지수(S&P500, 나스닥) 데이터를 수집할 때, THE Pair_Data_Module SHALL 미국 동부시간(ET) 기준 해당 시장의 거래 시간으로 일별 종가 데이터를 수집한다
4. THE Pair_Data_Module SHALL 모든 타임스탬프를 미국 동부시간(ET, America/New_York)으로 통일하여 저장한다
4. IF 외부 API 호출이 실패하면, THEN THE Pair_Data_Module SHALL 최대 3회 재시도하고, 실패 시 마지막 성공 데이터를 유지하며 오류를 로그에 기록한다
5. THE Pair_Data_Module SHALL 각 자산과 XRP 간의 상관계수를 일별로 계산하여 저장한다

### 요구사항 3: 시장 심리 데이터 수집

**사용자 스토리:** 데이터 분석가로서, 구글 검색 트렌드, SNS 검색량, 공포탐욕지수 등 시장 심리 데이터를 수집하고 싶다. 이를 통해 시장 참여자들의 심리를 예측 모델에 반영할 수 있다.

#### 인수 조건

1. THE Sentiment_Module SHALL "XRP", "리플", "Ripple" 키워드에 대한 구글 검색 트렌드 데이터를 일별로 수집한다
2. THE Sentiment_Module SHALL 주요 SNS 플랫폼(트위터/X)에서 XRP 관련 언급량과 감성 점수를 일별로 수집한다
3. THE Sentiment_Module SHALL 암호화폐 공포탐욕지수(Fear_Greed_Index)를 일별로 수집한다
4. IF 검색 트렌드 API 또는 SNS API 호출이 실패하면, THEN THE Sentiment_Module SHALL 오류를 로그에 기록하고 마지막 성공 데이터를 유지한다
5. THE Sentiment_Module SHALL 수집된 모든 심리 데이터를 0~100 범위로 정규화하여 저장한다

### 요구사항 4: 온체인 데이터 수집

**사용자 스토리:** 데이터 분석가로서, XRP 네트워크의 지갑 활동과 거래량 데이터를 수집하고 싶다. 이를 통해 네트워크 활성도를 예측 모델에 반영할 수 있다.

#### 인수 조건

1. THE Onchain_Module SHALL XRP 네트워크의 일별 활성 지갑 수, 신규 지갑 수, 거래 건수를 수집한다
2. THE Onchain_Module SHALL XRP 네트워크의 일별 총 거래량(XRP 단위)을 수집한다
3. THE Onchain_Module SHALL 대규모 거래(고래 거래, 100만 XRP 이상)의 건수와 총량을 일별로 수집한다
4. IF 온체인 데이터 소스 접근이 실패하면, THEN THE Onchain_Module SHALL 오류를 로그에 기록하고 마지막 성공 데이터를 유지한다
5. THE Onchain_Module SHALL 수집된 온체인 데이터를 타임스탬프와 함께 구조화된 형식으로 저장한다

### 요구사항 5: 멀티 모델 앙상블 가격 예측

**사용자 스토리:** 데이터 분석가로서, 여러 학습 모델을 독립적으로 학습시키고 앙상블 방식으로 통합하여 XRP 가격을 예측하고 싶다. 이를 통해 단일 모델 대비 예측 안정성과 정확도를 높일 수 있다.

#### 인수 조건

1. THE Prediction_Engine SHALL 기술 지표, 엘리엇 파동 분석 결과, 와이코프 분석 결과, 페어 데이터, 심리 데이터, 온체인 데이터를 통합하여 학습 데이터셋을 구성한다
2. THE Prediction_Engine SHALL LSTM, XGBoost, Random Forest, Transformer 모델을 Individual_Model로 각각 독립적으로 학습한다
3. THE Prediction_Engine SHALL 각 Individual_Model에 대해 Short_Term_Prediction(1일~7일), Mid_Term_Prediction(1주~1개월), Long_Term_Prediction(1개월~3개월) 세 가지 타임프레임별 예측을 수행한다
4. THE Prediction_Engine SHALL 각 Individual_Model의 예측 결과로 예측 가격, 상승 확률(0~100), 하락 확률(0~100), 신뢰도 점수(0~100)를 산출한다
5. THE Ensemble_Engine SHALL 각 Individual_Model의 예측 결과를 가중 평균 방식으로 통합하여 최종 예측 가격을 산출한다
6. THE Ensemble_Engine SHALL 각 Individual_Model의 예측 방향(상승/하락)에 대해 가중 투표 방식으로 최종 예측 방향과 통합 상승 확률, 통합 하락 확률을 산출한다
7. THE Ensemble_Engine SHALL 각 Individual_Model의 과거 예측 정확도를 기반으로 Ensemble_Weight를 동적으로 조정한다
8. WHEN 새로운 데이터가 수집되면, THE Prediction_Engine SHALL 24시간 이내에 모든 Individual_Model의 예측 결과와 앙상블 통합 결과를 갱신한다
9. IF 학습 데이터에 충분한 이력이 없으면(최소 90일 미만), THEN THE Prediction_Engine SHALL 예측 결과에 "데이터 부족" 경고를 포함한다
10. THE Prediction_Engine SHALL 각 피처의 예측 기여도(feature importance)를 Individual_Model별로 계산하여 저장한다
11. THE Ensemble_Engine SHALL 각 Individual_Model의 개별 정확도(MAE, MAPE, 방향 정확도)를 타임프레임별로 모니터링하고 저장한다

### 요구사항 6: 대시보드 시각화

**사용자 스토리:** 사용자로서, XRP 가격 예측 결과와 관련 데이터를 직관적인 대시보드에서 확인하고 싶다. 이를 통해 예측 정보를 쉽게 파악할 수 있다.

#### 인수 조건

1. THE Dashboard SHALL XRP의 현재 가격, 24시간 변동률, 거래량을 실시간으로 표시한다
2. THE Dashboard SHALL 단기, 중기, 장기 예측 결과(예측 가격, 방향, 신뢰도)를 각각 구분하여 표시한다
3. THE Dashboard SHALL XRP 가격 차트에 기술 지표(RSI, MACD, 볼린저밴드, 이동평균선)를 오버레이하여 표시한다
4. THE Dashboard SHALL XRP 가격 차트에 엘리엇 파동 카운트(파동 번호, 파동 구간)와 피보나치 되돌림 수준을 오버레이하여 표시한다
5. THE Dashboard SHALL 현재 엘리엇 파동 위치(진행 중인 파동 번호)와 예상 다음 파동 방향을 요약 패널로 표시한다
6. THE Dashboard SHALL XRP 가격 차트에 와이코프 패턴 이벤트(PS, SC, AR, ST, Spring, SOS, LPS, PSY, BC, UTAD, LPSY, SOW)를 마커로 오버레이하여 표시한다
7. THE Dashboard SHALL 현재 와이코프 시장 단계(Accumulation/Markup/Distribution/Markdown)와 Wyckoff_Phase(A~E), 신뢰도 점수를 요약 패널로 표시한다
8. THE Dashboard SHALL 와이코프 패턴의 거래량 분석 결과(가격-거래량 확산/수렴 관계)를 차트 형태로 표시한다
9. THE Dashboard SHALL 상관 자산(XRP/BTC, XRP/ETH, BTC/USD, ETH/USD, S&P500, 나스닥)과 XRP의 상관계수를 시각적으로 표시한다
10. THE Dashboard SHALL 공포탐욕지수, 검색 트렌드, SNS 감성 점수를 게이지 또는 차트 형태로 표시한다
11. THE Dashboard SHALL 온체인 데이터(활성 지갑 수, 거래량, 고래 거래)를 차트 형태로 표시한다
12. THE Dashboard SHALL 각 Individual_Model(LSTM, XGBoost, Random Forest, Transformer)의 개별 예측 결과(예측 가격, 상승/하락 확률, 신뢰도)를 모델별로 구분하여 표시한다
13. THE Dashboard SHALL Ensemble_Engine의 최종 통합 예측 결과(통합 예측 가격, 통합 상승/하락 확률, 최종 예측 방향)를 별도의 요약 패널로 표시한다
14. THE Dashboard SHALL 각 Individual_Model의 Ensemble_Weight와 개별 정확도 지표를 표시한다
15. THE Dashboard SHALL 각 피처의 예측 기여도를 막대 차트로 표시한다
16. WHEN 사용자가 예측 타임프레임을 선택하면, THE Dashboard SHALL 해당 타임프레임의 예측 결과와 관련 데이터를 2초 이내에 표시한다
17. THE Dashboard SHALL 각 Individual_Model과 Ensemble_Engine의 Direction_Hit_Rate와 Range_Hit_Rate를 타임프레임별(단기/중기/장기)로 구분하여 표시한다
18. THE Dashboard SHALL 적중률 이력을 시계열 차트로 표시하여 모델 성능 추이를 확인할 수 있도록 한다
19. THE Dashboard SHALL 각 Individual_Model의 적중률과 Ensemble_Engine의 적중률을 비교하는 막대 차트를 표시한다
20. THE Dashboard SHALL Retraining_Scheduler의 재학습 이력(재학습 일시, 방식, 성능 변화, 모델 교체 여부)을 테이블 형태로 표시한다

### 요구사항 7: 데이터 파이프라인 및 스케줄링

**사용자 스토리:** 시스템 관리자로서, 데이터 수집과 모델 학습이 자동으로 스케줄링되기를 원한다. 이를 통해 수동 개입 없이 시스템이 지속적으로 운영될 수 있다.

#### 인수 조건

1. THE Data_Collector SHALL 모든 데이터 소스에 대해 정의된 주기(기술 지표: 1시간, 페어: 1시간, 심리: 1일, 온체인: 1일)에 따라 자동으로 데이터를 수집한다
2. WHEN 모든 일별 데이터 수집이 완료되면, THE Prediction_Engine SHALL 자동으로 모델 재학습을 시작한다
3. IF 데이터 수집 작업이 연속 3회 실패하면, THEN THE Data_Collector SHALL 관리자에게 알림을 발송한다
4. THE Data_Collector SHALL 각 수집 작업의 시작 시간, 종료 시간, 성공/실패 상태를 로그에 기록한다
5. THE Retraining_Scheduler SHALL 재학습 주기를 일별 또는 주별로 설정할 수 있도록 하며, 기본 재학습 주기는 일별로 설정한다
6. WHEN 재학습 주기가 도래하면, THE Retraining_Scheduler SHALL 새로 축적된 데이터를 포함하여 각 Individual_Model의 재학습을 수행한다
7. THE Retraining_Scheduler SHALL 각 Individual_Model에 대해 Incremental_Learning 또는 Full_Retraining 방식을 선택하여 재학습을 수행한다
8. WHEN 재학습이 완료되면, THE Retraining_Scheduler SHALL Challenger_Model의 성능 지표(MAE, MAPE, Direction_Hit_Rate, Range_Hit_Rate)를 Champion_Model의 성능 지표와 비교한다
9. WHEN Challenger_Model의 성능 지표가 Champion_Model보다 개선된 경우, THE Retraining_Scheduler SHALL Champion_Model을 Challenger_Model로 교체한다
10. IF Challenger_Model의 성능 지표가 Champion_Model보다 저하된 경우, THEN THE Retraining_Scheduler SHALL 기존 Champion_Model을 유지하고 교체를 수행하지 않는다
11. THE Retraining_Scheduler SHALL 각 재학습의 실행 일시, 재학습 방식(Incremental/Full), 학습 데이터 기간, Champion_Model 성능, Challenger_Model 성능, 교체 여부를 이력으로 저장한다
12. IF 재학습 과정에서 오류가 발생하면, THEN THE Retraining_Scheduler SHALL 오류를 로그에 기록하고 기존 Champion_Model을 유지하며 관리자에게 알림을 발송한다

### 요구사항 8: 예측 성능 모니터링

**사용자 스토리:** 데이터 분석가로서, 예측 모델의 정확도를 지속적으로 모니터링하고 싶다. 이를 통해 모델의 신뢰성을 평가하고 개선할 수 있다.

#### 인수 조건

1. THE Prediction_Engine SHALL 각 Individual_Model과 Ensemble_Engine의 과거 예측 결과와 실제 가격을 비교하여 MAE(평균 절대 오차), MAPE(평균 절대 백분율 오차), 방향 정확도를 계산한다
2. THE Dashboard SHALL 각 타임프레임별, 각 Individual_Model별 예측 정확도 지표와 Ensemble_Engine의 통합 정확도 지표를 표시한다
3. WHEN 특정 Individual_Model의 방향 정확도가 50% 미만으로 떨어지면, THE Ensemble_Engine SHALL 해당 모델의 Ensemble_Weight를 하향 조정하고 경고를 표시한다
4. WHEN Ensemble_Engine의 통합 방향 정확도가 50% 미만으로 떨어지면, THE Prediction_Engine SHALL 해당 타임프레임의 예측 결과에 "정확도 저하" 경고를 포함한다
5. THE Prediction_Engine SHALL 최근 30일간의 각 Individual_Model별 예측 이력과 앙상블 통합 예측 이력, 실제 결과를 저장하여 백테스트에 활용할 수 있도록 한다
6. WHEN 예측 대상 기간이 경과하면, THE Hit_Rate_Tracker SHALL 각 Individual_Model과 Ensemble_Engine의 예측 결과와 실제 가격을 비교하여 Direction_Hit_Rate를 계산한다
7. WHEN 예측 대상 기간이 경과하면, THE Hit_Rate_Tracker SHALL 각 Individual_Model과 Ensemble_Engine의 예측 가격이 실제 가격의 허용 오차 범위(단기: ±3%, 중기: ±5%, 장기: ±10%) 내에 있었는지 비교하여 Range_Hit_Rate를 계산한다
8. THE Hit_Rate_Tracker SHALL Direction_Hit_Rate와 Range_Hit_Rate를 타임프레임별(Short_Term_Prediction, Mid_Term_Prediction, Long_Term_Prediction)로 구분하여 산출한다
9. THE Hit_Rate_Tracker SHALL 각 Individual_Model별 적중률과 Ensemble_Engine의 적중률을 별도로 산출하여 모델 간 성능 비교가 가능하도록 한다
10. THE Hit_Rate_Tracker SHALL 적중률 산출 결과를 일별 시계열 데이터로 저장하여 모델 성능 추이를 추적할 수 있도록 한다
11. WHEN 특정 Individual_Model의 Direction_Hit_Rate가 30일 연속 Ensemble_Engine의 Direction_Hit_Rate보다 10%p 이상 낮으면, THE Hit_Rate_Tracker SHALL 해당 모델에 대해 "성능 저하 지속" 경고를 생성한다
