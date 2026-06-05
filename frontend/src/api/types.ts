export type QueryType = 'factual' | 'comparative' | 'summary'
export type RecallRoute = 'vector' | 'bm25' | 'hybrid'
export type EvalType = 'generation' | 'retrieval' | 'ragas'

export interface RAGRequest {
  query: string
  stock_code?: string
  query_type?: QueryType
  recall_route?: RecallRoute
  recall_top_k?: number
  rerank_top_k?: number
  refusal_threshold?: number
}

export interface HealthResponse {
  status: string
  pipeline_ready: boolean
  models_loaded: boolean
  audit: Record<string, string | number | boolean>
  defaults: Record<string, string | number>
}

export interface JobStatusResponse {
  job_id: string
  job_type: string
  status: string
  progress?: string
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  duration_ms?: number | null
  result?: Record<string, unknown>
  error?: string
}

export interface EvalJobRequest {
  eval_type?: EvalType
  limit?: number
  skip_ragas?: boolean
  save_detail?: boolean
  resume?: boolean
  compare_routes?: boolean
  route?: RecallRoute
  top_k?: number
  legacy_retriever?: boolean
  pipeline_stage?: 'recall' | 'rerank'
}
