import { Alert, Card, Descriptions, Space, Tag, Typography } from 'antd'

import type { JobStatusResponse } from '../../api/types'
import { EvalMetricsPanel } from './EvalMetricsPanel'
import {
  GENERATION_METRIC_DEFS,
  RETRIEVAL_METRIC_DEFS,
} from './evalMetrics'

const { Text, Paragraph } = Typography

interface EvalResultPanelProps {
  job: JobStatusResponse
  evalType: string
}

export function EvalResultPanel({ job, evalType }: EvalResultPanelProps) {
  const result = job.result ?? {}
  const metrics = (result.metrics as Record<string, unknown> | undefined) ?? {}
  const outputs = (result.outputs as Record<string, string> | undefined) ?? {}

  const questionCount =
    result.question_count ?? metrics.question_count ?? '-'
  const ragasScoredN = result.ragas_scored_n ?? metrics.ragas_scored_n

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card size="small">
        <Descriptions size="small" column={{ xs: 1, sm: 2 }}>
          <Descriptions.Item label="任务 ID">{job.job_id}</Descriptions.Item>
          <Descriptions.Item label="评测类型">
            <Tag>{evalType}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="状态">
            <Tag color="success">{job.status}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="耗时">
            {job.duration_ms != null ? `${job.duration_ms} ms` : '-'}
          </Descriptions.Item>
          <Descriptions.Item label="题目数">{String(questionCount)}</Descriptions.Item>
          {ragasScoredN != null && (
            <Descriptions.Item label="RAGAS 计分样本">
              {String(ragasScoredN)}
            </Descriptions.Item>
          )}
          {result.mode != null && (
            <Descriptions.Item label="模式">{String(result.mode)}</Descriptions.Item>
          )}
          {result.route != null && (
            <Descriptions.Item label="路线">{String(result.route)}</Descriptions.Item>
          )}
        </Descriptions>
      </Card>

      {evalType === 'retrieval' && (
        <EvalMetricsPanel
          title="检索质量指标"
          metrics={metrics}
          defs={RETRIEVAL_METRIC_DEFS}
        />
      )}

      {(evalType === 'generation' || evalType === 'ragas') && (
        <EvalMetricsPanel
          title="生成质量指标"
          metrics={metrics}
          defs={GENERATION_METRIC_DEFS}
        />
      )}

      {evalType === 'generation' && (
        <Alert
          type="info"
          showIcon
          message="检索指标说明"
          description="Recall@K / MRR / Context Precision 请运行「检索评测 (retrieval)」任务获取；generation 任务主要输出引用、拒答与 RAGAS 相关指标。"
        />
      )}

      {Object.keys(outputs).length > 0 && (
        <Card title="输出文件（服务端路径）">
          <Descriptions bordered size="small" column={1}>
            {Object.entries(outputs).map(([key, path]) => (
              <Descriptions.Item key={key} label={key}>
                <Text code>{path || '-'}</Text>
              </Descriptions.Item>
            ))}
          </Descriptions>
          <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
            逐样本明细保存在上述文件中（如 detail_jsonl / results_csv）。
            当前后端任务结果仅返回汇总指标，未内嵌逐条样本。
          </Paragraph>
        </Card>
      )}
    </Space>
  )
}
