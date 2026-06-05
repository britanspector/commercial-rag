import axios from 'axios'

import { API_BASE_URL } from '../config'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120_000,
})

apiClient.interceptors.request.use((config) => {
  if (config.data instanceof FormData) {
    delete config.headers['Content-Type']
  } else if (!config.headers['Content-Type']) {
    config.headers['Content-Type'] = 'application/json'
  }
  return config
})

apiClient.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (axios.isAxiosError(error)) {
      const detail = error.response?.data?.detail
      const message =
        typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map((item) => item.msg).join('; ')
            : error.message
      return Promise.reject(new Error(message || '请求失败'))
    }
    return Promise.reject(error)
  },
)
