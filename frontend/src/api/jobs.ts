import { apiClient } from './client'
import type { JobStatusResponse } from './types'

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  const { data } = await apiClient.get<JobStatusResponse>(`/jobs/${jobId}`)
  return data
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

export async function pollJobUntilDone(
  jobId: string,
  options?: {
    intervalMs?: number
    onUpdate?: (job: JobStatusResponse) => void
  },
): Promise<JobStatusResponse> {
  const intervalMs = options?.intervalMs ?? 2000

  while (true) {
    const job = await fetchJobStatus(jobId)
    options?.onUpdate?.(job)
    if (job.status === 'success' || job.status === 'failed') {
      return job
    }
    await sleep(intervalMs)
  }
}
