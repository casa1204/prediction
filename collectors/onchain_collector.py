"""OnchainCollector — XRPL 공식 API에서 XRP 온체인 데이터를 수집한다.

XRPL JSON-RPC API (s1.ripple.com)를 사용하며, API 키가 불필요하다.
활성 지갑, 거래 건수, 총 거래량, 고래 거래를 수집하여
OnchainData ORM 모델로 DB에 저장한다.
모든 타임스탬프는 미국 동부시간(America/New_York)으로 변환된다.

Requirements: 4.1, 4.2, 4.3
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from collectors.base_collector import BaseCollector
from config import TIMEZONE, WHALE_THRESHOLD_XRP
from db.database import get_session
from db.models import OnchainData

logger = logging.getLogger(__name__)

_ET = ZoneInfo(TIMEZONE)

# XRPL 공식 JSON-RPC 엔드포인트 (키 불필요, 무료)
_XRPL_RPC_URL = "https://s1.ripple.com:51234/"


class OnchainCollector(BaseCollector):
    """XRPL 공식 API에서 XRP 온체인 데이터를 수집하는 수집기.

    XRPL JSON-RPC의 server_info, ledger, account_tx 등을 활용한다.
    API 키가 필요하지 않다.
    """

    source_name = "onchain"

    def __init__(self, rpc_url: str = _XRPL_RPC_URL):
        super().__init__()
        self._rpc_url = rpc_url

    async def collect(self) -> dict:
        now_et = datetime.now(timezone.utc).astimezone(_ET).replace(tzinfo=None)
        timestamp = now_et.replace(hour=0, minute=0, second=0, microsecond=0)

        async with httpx.AsyncClient(timeout=120.0) as client:
            # 레저에서 모든 온체인 메트릭을 한 번에 수집
            metrics = await self._collect_from_ledger(client)

        record = {
            "timestamp": timestamp,
            "active_wallets": metrics["active_wallets"],
            "new_wallets": 0,
            "transaction_count": metrics["transaction_count"],
            "total_volume_xrp": metrics["total_volume_xrp"],
            "whale_tx_count": metrics["whale_tx_count"],
            "whale_tx_volume": metrics["whale_tx_volume"],
        }

        _save_record(record)

        logger.info(
            "[onchain] 온체인 데이터 수집 완료 — "
            "tx=%d, vol=%.0f XRP, whale=%d/%.0f XRP",
            record["transaction_count"],
            record["total_volume_xrp"],
            record["whale_tx_count"],
            record["whale_tx_volume"],
        )

        return {
            "source": self.source_name,
            "timestamp": timestamp.isoformat(),
            **{k: v for k, v in record.items() if k != "timestamp"},
        }

    async def _collect_from_ledger(self, client: httpx.AsyncClient) -> dict:
        """최근 100개 레저를 순회하여 거래 건수, 총 거래량, 고래 거래를 수집한다.

        단일 레저는 3~4초 분량이므로 100개 ≈ 약 5분치 데이터.
        고래 거래 감지 확률을 높이기 위해 여러 레저를 조회한다.
        """
        try:
            # 1) 최신 validated 레저 인덱스 조회
            resp = await client.post(
                self._rpc_url,
                json={
                    "method": "ledger",
                    "params": [{"ledger_index": "validated"}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            latest_index = data.get("result", {}).get("ledger", {}).get("ledger_index")
            if not latest_index:
                logger.warning("[onchain] 최신 레저 인덱스를 가져올 수 없음")
                return _empty_metrics()

            latest_index = int(latest_index)

            # 2) 최근 100개 레저 순회 (10개씩 배치로 요청)
            total_tx_count = 0
            total_xrp_volume = 0.0
            all_payment_amounts: list[dict] = []

            batch_size = 10
            num_ledgers = 100

            for batch_start in range(0, num_ledgers, batch_size):
                batch_indices = [
                    latest_index - i
                    for i in range(batch_start, min(batch_start + batch_size, num_ledgers))
                ]

                for ledger_idx in batch_indices:
                    try:
                        ledger_resp = await client.post(
                            self._rpc_url,
                            json={
                                "method": "ledger",
                                "params": [{
                                    "ledger_index": ledger_idx,
                                    "transactions": True,
                                    "expand": True,
                                }],
                            },
                        )
                        ledger_resp.raise_for_status()
                        ledger_data = ledger_resp.json()

                        ledger = ledger_data.get("result", {}).get("ledger", {})
                        transactions = ledger.get("transactions", [])
                        total_tx_count += len(transactions)

                        for tx in transactions:
                            tx_data = tx if isinstance(tx, dict) else {}
                            tx_type = tx_data.get("TransactionType", "")
                            if tx_type != "Payment":
                                continue

                            amount = tx_data.get("Amount", "0")
                            if isinstance(amount, str):
                                try:
                                    xrp_amount = float(amount) / 1_000_000
                                    total_xrp_volume += xrp_amount
                                    all_payment_amounts.append({"amount": xrp_amount})
                                except (ValueError, TypeError):
                                    continue

                    except Exception:
                        logger.debug("[onchain] 레저 %d 조회 실패, 건너뜀", ledger_idx)
                        continue

            # 3) 고래 거래 필터링
            whale_txs = filter_whale_transactions(all_payment_amounts, WHALE_THRESHOLD_XRP)

            # 4) 총 계정 수 (server_state에서)
            active_wallets = 0
            try:
                info_resp = await client.post(
                    self._rpc_url,
                    json={"method": "server_state", "params": [{}]},
                )
                info_data = info_resp.json()
                vl = info_data.get("result", {}).get("state", {}).get("validated_ledger", {})
                active_wallets = vl.get("accounts", 0) or 0
            except Exception:
                pass

            logger.info(
                "[onchain] %d개 레저 조회: tx=%d, payments=%d, vol=%.0f XRP, whale=%d",
                num_ledgers, total_tx_count, len(all_payment_amounts),
                total_xrp_volume, len(whale_txs),
            )

            return {
                "active_wallets": active_wallets,
                "transaction_count": total_tx_count,
                "total_volume_xrp": total_xrp_volume,
                "whale_tx_count": len(whale_txs),
                "whale_tx_volume": sum(tx.get("amount", 0) for tx in whale_txs),
            }

        except Exception:
            logger.exception("[onchain] 레저 데이터 수집 실패")
            raise


def _empty_metrics() -> dict:
    """빈 메트릭 반환."""
    return {
        "active_wallets": 0,
        "transaction_count": 0,
        "total_volume_xrp": 0.0,
        "whale_tx_count": 0,
        "whale_tx_volume": 0.0,
    }


def filter_whale_transactions(
    transactions: list[dict], threshold: float = WHALE_THRESHOLD_XRP
) -> list[dict]:
    """거래 목록에서 고래 거래(threshold XRP 이상)를 필터링한다."""
    return [
        tx for tx in transactions
        if float(tx.get("amount", 0)) >= threshold
    ]


def _save_record(record: dict) -> None:
    """OnchainData 레코드를 DB에 저장한다. 중복 타임스탬프는 업데이트한다."""
    session = get_session()
    try:
        existing = (
            session.query(OnchainData)
            .filter(OnchainData.timestamp == record["timestamp"])
            .first()
        )
        if existing:
            for key, value in record.items():
                if key != "timestamp":
                    setattr(existing, key, value)
            session.commit()
            logger.info("[onchain] 기존 레코드 업데이트: %s", record["timestamp"])
            return

        session.add(OnchainData(
            timestamp=record["timestamp"],
            active_wallets=record["active_wallets"],
            new_wallets=record["new_wallets"],
            transaction_count=record["transaction_count"],
            total_volume_xrp=record["total_volume_xrp"],
            whale_tx_count=record["whale_tx_count"],
            whale_tx_volume=record["whale_tx_volume"],
        ))
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("[onchain] DB 저장 실패")
        raise
    finally:
        session.close()
