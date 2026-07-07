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

export interface RetrievedChunkResponse {
  rank: number
  chunk_id: string
  text: string
  company_name: string
  section_title: string
  page_start: number
  page_end: number
  display_name: string
  doc_id?: string
  source_pdf_path?: string
  score: number
  score_recall?: number
  score_rerank?: number | null
  score_vector?: number | null
  score_bm25?: number | null
}

export interface CitationResponse {
  index: number
  chunk_id: string
  company_name: string
  section_title: string
  page_start: number
  page_end: number
  display_name: string
  score_rerank: number
  doc_id?: string
  source_pdf_path?: string
  filename?: string
  page_label?: string
  source_document?: string
}

export interface EvidenceCheckResponse {
  passed: boolean
  top_rerank_score: number
  refusal_reason?: string
  refusal_message?: string
  citation_count?: number
  checks?: Array<Record<string, string | boolean>>
}

export interface QueryRewriteResponse {
  original_query: string
  query: string
  bm25_query: string
  stock_code: string
  query_type: string
  compare_entities: string[]
  hybrid_vector_weight: number
  query_vector_dim?: number | null
}

export interface RecallStageResponse {
  route: string
  recall_top_k: number
  hit_count: number
  hits: RetrievedChunkResponse[]
}

export interface RerankStageResponse {
  rerank_top_k: number
  hit_count: number
  top_rerank_score: number
  hits: RetrievedChunkResponse[]
}

export interface CacheInfoResponse {
  hit: boolean
  source: string
  similarity?: number | null
  reason?: string
  safety_ok?: boolean
  safety_reason?: string
  latency_ms: number
  lookup_ms?: number
  pipeline_ms?: number
  vector_retrieval?: boolean
  llm_called?: boolean
}

export interface CacheStatsResponse {
  active: boolean
  lookups: number
  hits_l1: number
  hits_l2: number
  misses: number
  lookup_hit_rate: number
  requests: number
  request_hits_l1: number
  request_hits_l2: number
  request_misses: number
  l1_hit_rate: number
  l2_hit_rate: number
  total_hit_rate: number
  avg_latency_ms: number
  avg_hit_latency_ms: number
  avg_miss_latency_ms: number
  avg_latency_saved_ms: number
  vector_retrievals_saved: number
  llm_calls_saved: number
  llm_call_reduction_rate: number
  safety_rejects: number
  stores: number
  exact_entries: number
  semantic_entries: number
  backends: Record<string, unknown>
}

export interface SearchResponse {
  query: string
  top_rerank_score: number
  query_rewrite: QueryRewriteResponse
  recall: RecallStageResponse
  rerank: RerankStageResponse
  cache?: CacheInfoResponse
}

export interface ChatResponse {
  query: string
  answer: string
  refused: boolean
  refusal_reason: string
  refusal_message?: string
  top_rerank_score: number
  citations: CitationResponse[]
  rerank_hits: RetrievedChunkResponse[]
  evidence_check: EvidenceCheckResponse
  cache?: CacheInfoResponse
}

export interface HealthResponse {
  status: string
  pipeline_ready: boolean
  models_loaded: boolean
  audit: Record<string, string | number | boolean>
  cache?: Record<string, string | number | boolean>
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
