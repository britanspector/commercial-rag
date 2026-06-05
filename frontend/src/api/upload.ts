import { apiClient } from './client'
import type { UploadParams, UploadResponse } from './types'

const UPLOAD_TIMEOUT_MS = 30 * 60 * 1000

export async function uploadPdf(params: UploadParams): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', params.file)
  if (params.industry) {
    formData.append('industry', params.industry)
  }
  if (params.industry_label) {
    formData.append('industry_label', params.industry_label)
  }
  formData.append('replace_existing', String(params.replace_existing ?? true))
  formData.append('background', String(params.background ?? false))

  const { data } = await apiClient.post<UploadResponse>('/upload', formData, {
    timeout: UPLOAD_TIMEOUT_MS,
    onUploadProgress: (event) => {
      if (!params.onUploadProgress || !event.total) {
        return
      }
      const percent = Math.round((event.loaded / event.total) * 100)
      params.onUploadProgress(percent)
    },
  })
  return data
}
