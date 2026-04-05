import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import useBinanceWs from '../hooks/useBinanceWs';

const navItems = [
  { to: '/', label: '대시보드', icon: '📊' },
  { to: '/analysis', label: '기술 분석', icon: '📈' },
  { to: '/performance', label: '성능', icon: '🎯' },
  { to: '/ai', label: 'AI 분석', icon: '🤖' },
];

function Sidebar() {
  return (
    <aside className="hidden md:flex fixed left-0 top-0 h-full w-56 bg-secondary border-r border-gray-700 flex-col z-10">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-lg font-bold text-highlight">XRP 예측</h1>
        <p className="text-xs text-gray-400 mt-1">Price Prediction Dashboard</p>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-accent text-white font-medium'
                  : 'text-gray-400 hover:bg-accent/30 hover:text-gray-200'
              }`
            }
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}

function MobileNav() {
  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-secondary border-t border-gray-700 z-20 flex">
      {navItems.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to === '/'}
          className={({ isActive }) =>
            `flex-1 flex flex-col items-center py-2 text-xs transition-colors ${
              isActive ? 'text-accent' : 'text-gray-500'
            }`
          }
        >
          <span className="text-lg">{item.icon}</span>
          <span className="mt-0.5">{item.label}</span>
        </NavLink>
      ))}
    </nav>
  );
}

function Header() {
  const { price, change24h, volume24h, high24h, low24h, connected } = useBinanceWs();
  const isPositive = (change24h ?? 0) >= 0;

  return (
    <header className="h-14 bg-secondary border-b border-gray-700 flex items-center justify-between px-4 md:px-6">
      <div className="flex items-center gap-3 md:gap-6 overflow-x-auto">
        <span className="text-sm text-gray-400 shrink-0">XRP/USDT</span>
        {price == null ? (
          <span className="text-sm text-gray-500">연결 중...</span>
        ) : (
          <>
            <span className="text-lg font-bold shrink-0">${price.toFixed(4)}</span>
            <span className={`text-sm font-medium shrink-0 ${isPositive ? 'text-success' : 'text-danger'}`}>
              {isPositive ? '+' : ''}{(change24h ?? 0).toFixed(2)}%
            </span>
            <span className="text-xs text-gray-400 hidden sm:inline shrink-0">
              H: ${(high24h ?? 0).toFixed(4)} L: ${(low24h ?? 0).toFixed(4)}
            </span>
            <span className="text-xs text-gray-400 hidden sm:inline shrink-0">
              Vol: {(volume24h ?? 0) >= 1e9
                ? `$${((volume24h ?? 0) / 1e9).toFixed(2)}B`
                : `$${((volume24h ?? 0) / 1e6).toFixed(2)}M`}
            </span>
          </>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <span className={`w-2 h-2 rounded-full ${connected ? 'bg-success animate-pulse' : 'bg-danger'}`} />
        <span className="text-xs text-gray-500">{connected ? 'LIVE' : 'OFF'}</span>
      </div>
    </header>
  );
}

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-primary">
      <Sidebar />
      <div className="md:ml-56">
        <Header />
        <main className="p-3 md:p-6 pb-20 md:pb-6">{children}</main>
      </div>
      <MobileNav />
    </div>
  );
}
