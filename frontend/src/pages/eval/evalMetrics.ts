export interface MetricDefinition {
  key: string
  label: string
  description?: string
}

export const RETRIEVAL_METRIC_DEFS: MetricDefinition[] = [
  { key: 'recall_at_5', label: 'Recall@5', description: 'Top-5 召回率' },
  { key: 'recall_at_10', label: 'Recall@10', description: 'Top-10 召回率' },
  { key: 'mrr', label: 'MRR', description: 'Mean Reciprocal Rank' },
  {
    key: 'context_precision_at_5',
    label: 'Context Precision@5',
    description: 'Top-5 上下文精确率',
  },
  {
    key: 'context_precision_at_10',
    label: 'Context Precision@10',
    description: 'Top-10 上下文精确率',
  },
  { key: 'hit_rate', label: 'Hit Rate', description: '至少命中一次的比例' },
]

export const GENERATION_METRIC_DEFS: MetricDefinition[] = [
  {
    key: 'retrieval_hit_rate',
    label: 'Retrieval Hit Rate',
    description: 'Pipeline 检索命中比例',
  },
  {
    key: 'citation_accuracy',
    label: 'Citation Accuracy',
    description: '引用准确率（适用样本）',
  },
  {
    key: 'refusal_accuracy',
    label: 'Refusal Accuracy',
    description: '拒答判断准确率',
  },
  {
    key: 'answer_factually_supported_rate',
    label: 'Answer Supported Rate',
    description: '答案事实支持率',
  },
  {
    key: 'faithfulness_ragas',
    label: 'Faithfulness',
    description: 'RAGAS 忠实度',
  },
  {
    key: 'answer_relevancy_ragas',
    label: 'Answer Relevancy',
    description: 'RAGAS 答案相关性',
  },
]

export function formatMetricValue(value: unknown): string {
  if (value == null || value === '' || (typeof value === 'number' && Number.isNaN(value))) {
    return '-'
  }
  if (typeof value === 'number') {
    if (Number.isInteger(value) && value > 10) {
      return String(value)
    }
    return `${(value * 100).toFixed(1)}%`
  }
  return String(value)
}

export function pickMetrics(
  metrics: Record<string, unknown>,
  defs: MetricDefinition[],
): Array<{ def: MetricDefinition; value: unknown }> {
  return defs
    .filter((def) => def.key in metrics)
    .map((def) => ({ def, value: metrics[def.key] }))
}
