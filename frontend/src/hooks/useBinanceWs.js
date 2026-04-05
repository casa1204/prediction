import { useState, useEffect, useRef } from 'react';

const WS_URL = 'wss://stream.binance.com:9443/ws/xrpusdt@ticker';

/**
 * 바이낸스 WebSocket으로 XRP/USDT 실시간 티커를 구독한다.
 * 반환: { price, change24h, volume24h, high24h, low24h, connected }
 */
export default function useBinanceWs() {
  const [ticker, setTicker] = useState({
    price: null,
    change24h: null,
    volume24h: null,
    high24h: null,
    low24h: null,
    connected: false,
  });
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setTicker((prev) => ({ ...prev, connected: true }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setTicker({
            price: parseFloat(data.c),       // 현재가
            change24h: parseFloat(data.P),   // 24h 변동률 %
            volume24h: parseFloat(data.q),   // 24h 거래대금 (USDT)
            high24h: parseFloat(data.h),     // 24h 고가
            low24h: parseFloat(data.l),      // 24h 저가
            connected: true,
          });
        } catch { /* ignore parse errors */ }
      };

      ws.onclose = () => {
        setTicker((prev) => ({ ...prev, connected: false }));
        // 3초 후 재연결
        reconnectTimer.current = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, []);

  return ticker;
}
