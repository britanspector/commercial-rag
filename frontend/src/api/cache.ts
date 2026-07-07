import { apiClient } from './client'
import type { CacheStatsResponse } from './types'

export async function fetchCacheStats(): Promise<CacheStatsResponse> {
  const { data } = await apiClient.get<CacheStatsResponse>('/cache/stats')
  return data
}
