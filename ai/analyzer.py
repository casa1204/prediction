"""AI 시장 분석 모듈 — Gemini Pro를 활용한 XRP 시장 분석 리포트 생성.

DB에서 최신 데이터(기술지표, 예측결과, 심리, 거시경제, 온체인, ETF)를 수집하여
Gemini Pro에 구조화된 프롬프트로 전달하고, 한글 시장 분석 리포트를 생성한다.
"""

import logging
from google import genai
from config import GEMINI_API_KEY
from db.database import get_session
from db.models import (
    EtfData, OnchainData, PairData, PredictionRecord,
    SentimentData, TechnicalIndicator, ElliottWaveData, WyckoffData,
)
from sqlalchemy import desc

logger = logging.getLogger(__name__)


def _gather_context() -> str:
    """DB에서 최신 데이터를 수집하여 LLM 컨텍스트 문자열로 구성한다."""
    session = get_session()
    try:
        parts = []

        # 기술지표
        ti = session.query(TechnicalIndicator).order_by(desc(TechnicalIndicator.timestamp)).first()
        if ti:
            def f(v, fmt=".2f"):
                return f"{v:{fmt}}" if v is not None else "N/A"
            parts.append(f"""## 기술 지표 ({ti.timestamp})
- RSI(14): {f(ti.rsi_14)}
- MACD: {f(ti.macd, ".4f")}, Signal: {f(ti.macd_signal, ".4f")}, Histogram: {f(ti.macd_histogram, ".4f")}
- 볼린저밴드: 상단={f(ti.bb_upper, ".4f")}, 중간={f(ti.bb_middle, ".4f")}, 하단={f(ti.bb_lower, ".4f")}
- SMA: 5={f(ti.sma_5, ".4f")}, 20={f(ti.sma_20, ".4f")}, 50={f(ti.sma_50, ".4f")}, 200={f(ti.sma_200, ".4f")}
- EMA: 12={f(ti.ema_12, ".4f")}, 26={f(ti.ema_26, ".4f")}
- OBV: {f(ti.obv, ".0f")}
- 이격도: 50일={f(ti.disparity_50)}%, 100일={f(ti.disparity_100)}%, 200일={f(ti.disparity_200)}%""")

        # 예측 결과
        for tf_label, tf_key in [("단기(1일)", "short"), ("중기(7일)", "mid"), ("장기(30일)", "long")]:
            pred = (
                session.query(PredictionRecord)
                .filter(PredictionRecord.timeframe == tf_key, PredictionRecord.model_name == "ensemble")
                .order_by(desc(PredictionRecord.timestamp)).first()
            )
            if pred:
                parts.append(f"""## {tf_label} 예측
- 앙상블 예측가: ${pred.predicted_price:.4f}
- 방향: {pred.predicted_direction} (상승확률: {pred.up_probability:.1f}%, 하락확률: {pred.down_probability:.1f}%)""")

        # 심리 데이터
        sent = session.query(SentimentData).order_by(desc(SentimentData.timestamp)).first()
        if sent:
            s = []
            if sent.fear_greed_index is not None: s.append(f"공포탐욕지수: {sent.fear_greed_index:.0f}")
            if sent.google_trend_score is not None: s.append(f"XRP 검색트렌드: {sent.google_trend_score:.0f}")
            if sent.trend_macro_score is not None: s.append(f"거시경제 트렌드: {sent.trend_macro_score:.0f}")
            if sent.trend_etf_regulatory_score is not None: s.append(f"ETF/규제 트렌드: {sent.trend_etf_regulatory_score:.0f}")
            if sent.fomo_spread is not None: s.append(f"FOMO 스프레드: {sent.fomo_spread:.1f}")
            if sent.attention_ratio is not None: s.append(f"관심비율(XRP/BTC): {sent.attention_ratio:.4f}")
            if s:
                parts.append("## 시장 심리\n" + "\n".join(f"- {x}" for x in s))

        # 상관 자산
        from sqlalchemy import func
        sub = session.query(PairData.asset_name, func.max(PairData.timestamp).label("mt")).group_by(PairData.asset_name).subquery()
        pairs = session.query(PairData).join(sub, (PairData.asset_name == sub.c.asset_name) & (PairData.timestamp == sub.c.mt)).all()
        if pairs:
            lines = [f"- {p.asset_name}: ${p.price:.4f} (상관계수: {p.correlation_with_xrp:.4f})" for p in pairs if p.correlation_with_xrp]
            if lines:
                parts.append("## 상관 자산\n" + "\n".join(lines))

        # 온체인
        oc = session.query(OnchainData).order_by(desc(OnchainData.timestamp)).first()
        if oc:
            parts.append(f"""## 온체인 데이터
- 거래건수: {oc.transaction_count:,}
- 총 거래량: {oc.total_volume_xrp:,.0f} XRP
- 고래거래: {oc.whale_tx_count}건 / {oc.whale_tx_volume:,.0f} XRP""")

        # 엘리엇 파동
        ew = session.query(ElliottWaveData).order_by(desc(ElliottWaveData.timestamp)).first()
        if ew:
            parts.append(f"## 엘리엇 파동\n- 현재 파동: {ew.wave_number}파 ({ew.wave_type})")

        # 와이코프
        wk = session.query(WyckoffData).order_by(desc(WyckoffData.timestamp)).first()
        if wk:
            parts.append(f"## 와이코프 분석\n- 시장 단계: {wk.market_phase}\n- 와이코프 단계: {wk.wyckoff_phase}\n- 신뢰도: {wk.confidence_score:.0f}%")

        # ETF
        etfs = session.query(EtfData).order_by(desc(EtfData.timestamp)).limit(6).all()
        if etfs:
            lines = []
            for e in etfs:
                if e.price:
                    vol = f"{e.volume:,}" if e.volume else "N/A"
                    prem = f"{e.premium_discount:.2f}%" if e.premium_discount is not None else "N/A"
                    lines.append(f"- {e.ticker}: ${e.price:.2f} (거래량: {vol}, 프리미엄: {prem})")
            if lines:
                parts.append("## XRP ETF\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "데이터가 아직 수집되지 않았습니다."

    finally:
        session.close()


async def generate_analysis() -> str:
    """Gemini Pro를 사용하여 XRP 시장 분석 리포트를 생성한다."""
    if not GEMINI_API_KEY:
        return "Gemini API 키가 설정되지 않았습니다. `.env` 파일에 `GEMINI_API_KEY`를 추가하세요."

    context = _gather_context()

    prompt = f"""당신은 암호화폐 시장 전문 애널리스트입니다. 아래 XRP 관련 데이터를 분석하여 한글로 종합 시장 분석 리포트를 작성하세요.

{context}

다음 구조로 작성하세요:
1. **시장 요약** — 현재 XRP 시장 상황을 2~3문장으로 요약
2. **기술적 분석** — RSI, MACD, 볼린저밴드, 이동평균선 기반 분석. 과매수/과매도, 골든크로스/데드크로스 여부
3. **거시경제 환경** — DXY, VIX, S&P500, NASDAQ과의 상관관계 기반 분석
4. **시장 심리** — 공포탐욕지수, 검색트렌드, FOMO 스프레드 기반 대중 심리 분석
5. **온체인 활동** — 고래 거래, 거래량 기반 네트워크 활동 분석
6. **엘리엇 파동 / 와이코프** — 현재 파동 위치와 시장 단계 해석
7. **AI 예측 결과 해석** — 단기/중기/장기 예측 결과와 신뢰도 분석
8. **종합 의견** — 매수/매도/관망 중 하나를 추천하고 근거 제시
9. **리스크 요인** — 주의해야 할 위험 요소

마크다운 형식으로 작성하세요. 숫자와 데이터를 적극 활용하세요."""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        logger.exception("[ai] Gemini 분석 생성 실패")
        return f"AI 분석 생성 중 오류가 발생했습니다: {e}"

async def generate_and_save_analysis() -> str:
    """AI 분석을 생성하고 DB에 저장한다. 재학습 완료 후 자동 호출용."""
    from db.models import AiAnalysisRecord
    from datetime import datetime, timezone

    analysis = await generate_analysis()

    # 에러 메시지가 아닌 경우에만 저장
    if not analysis.startswith("AI 분석 생성 중 오류") and not analysis.startswith("Gemini API 키"):
        session = get_session()
        try:
            record = AiAnalysisRecord(
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                analysis=analysis,
                model_used="gemini-2.5-flash",
            )
            session.add(record)
            session.commit()
            logger.info("[ai] 분석 리포트 DB 저장 완료")
        except Exception:
            session.rollback()
            logger.exception("[ai] 분석 리포트 DB 저장 실패")
        finally:
            session.close()

    return analysis
