import { useQuery } from '@tanstack/react-query';
import apiClient from '../api/client';

export function useCurrentPrice() {
  return useQuery({
    queryKey: ['currentPrice'],
    queryFn: () => apiClient.get('/current-price'),
    refetchInterval: 60 * 1000,
  });
}

export function usePredictions(timeframe = 'short') {
  return useQuery({
    queryKey: ['predictions', timeframe],
    queryFn: () => apiClient.get(`/predictions/${timeframe}`),
  });
}

export function useTechnicalIndicators() {
  return useQuery({
    queryKey: ['technicalIndicators'],
    queryFn: () => apiClient.get('/technical-indicators'),
  });
}

export function useDailyPriceHistory() {
  return useQuery({
    queryKey: ['dailyPriceHistory'],
    queryFn: () => apiClient.get('/daily-price-history'),
    staleTime: 5 * 60 * 1000,
  });
}

export function usePriceHistory() {
  return useQuery({
    queryKey: ['priceHistory'],
    queryFn: () => apiClient.get('/price-history'),
    staleTime: 5 * 60 * 1000,
  });
}

export function useElliottWave() {
  return useQuery({
    queryKey: ['elliottWave'],
    queryFn: () => apiClient.get('/elliott-wave'),
  });
}

export function useWyckoff() {
  return useQuery({
    queryKey: ['wyckoff'],
    queryFn: () => apiClient.get('/wyckoff'),
  });
}

export function useCorrelations() {
  return useQuery({
    queryKey: ['correlations'],
    queryFn: () => apiClient.get('/correlations'),
  });
}

export function useSentiment() {
  return useQuery({
    queryKey: ['sentiment'],
    queryFn: () => apiClient.get('/sentiment'),
  });
}

export function useOnchain() {
  return useQuery({
    queryKey: ['onchain'],
    queryFn: () => apiClient.get('/onchain'),
  });
}

export function useHitRates() {
  return useQuery({
    queryKey: ['hitRates'],
    queryFn: () => apiClient.get('/hit-rates'),
  });
}

export function useFeatureImportance(modelName) {
  return useQuery({
    queryKey: ['featureImportance', modelName],
    queryFn: () => apiClient.get(`/feature-importance/${modelName}`),
    enabled: !!modelName,
  });
}

export function useRetrainingHistory() {
  return useQuery({
    queryKey: ['retrainingHistory'],
    queryFn: () => apiClient.get('/retraining-history'),
  });
}

export function useEtf() {
  return useQuery({
    queryKey: ['etf'],
    queryFn: () => apiClient.get('/etf'),
    refetchInterval: 5 * 60 * 1000,
  });
}

export function useModelWeights() {
  return useQuery({
    queryKey: ['modelWeights'],
    queryFn: () => apiClient.get('/model-weights'),
  });
}

export function useAiAnalysis(enabled = false) {
  return useQuery({
    queryKey: ['aiAnalysis'],
    queryFn: () => apiClient.get('/ai-analysis', { timeout: 120000 }),
    enabled,
    staleTime: 10 * 60 * 1000,
    retry: false,
  });
}

export function useAiAnalysisHistory() {
  return useQuery({
    queryKey: ['aiAnalysisHistory'],
    queryFn: () => apiClient.get('/ai-analysis/history'),
    staleTime: 5 * 60 * 1000,
  });
}
