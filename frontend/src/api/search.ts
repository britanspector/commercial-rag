import { apiClient } from './client'
import type { RAGRequest, SearchResponse } from './types'

const SEARCH_TIMEOUT_MS = 5 * 60 * 1000

export async function postSearch(request: RAGRequest): Promise<SearchResponse> {
  const { data } = await apiClient.post<SearchResponse>('/search', request, {
    timeout: SEARCH_TIMEOUT_MS,
  })
  return data
}
