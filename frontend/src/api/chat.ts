import { apiClient } from './client'
import type { ChatResponse, RAGRequest } from './types'

const CHAT_TIMEOUT_MS = 5 * 60 * 1000

export async function postChat(request: RAGRequest): Promise<ChatResponse> {
  const { data } = await apiClient.post<ChatResponse>('/chat', request, {
    timeout: CHAT_TIMEOUT_MS,
  })
  return data
}
