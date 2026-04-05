import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Performance from './pages/Performance';
import AiAnalysis from './pages/AiAnalysis';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="/ai" element={<AiAnalysis />} />
      </Routes>
    </Layout>
  );
}
