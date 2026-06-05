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

export interface UploadStageResponse {
  name: string
  status: string
  detail?: string
}

export interface UploadResponse {
  doc_id: string
  filename: string
  industry: string
  industry_label: string
  source_pdf_path: string
  display_name?: string
  company_name?: string
  stock_code?: string
  chunk_count: number
  retrievable_chunk_count: number
  milvus_rows_inserted: number
  milvus_total_rows: number
  bm25_total_chunks: number
  replaced_existing: boolean
  stages: UploadStageResponse[]
  job_id?: string
  async_mode?: boolean
}

export interface UploadParams {
  file: File
  industry?: string
  industry_label?: string
  replace_existing?: boolean
  background?: boolean
  onUploadProgress?: (percent: number) => void
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
