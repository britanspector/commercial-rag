import { apiClient } from './client'
import type { EvalJobRequest, JobStatusResponse } from './types'

export async function startEval(request: EvalJobRequest): Promise<JobStatusResponse> {
  const { data } = await apiClient.post<JobStatusResponse>('/eval', request)
  return data
}
