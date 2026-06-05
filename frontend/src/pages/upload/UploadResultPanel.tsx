import { Descriptions, Space, Table, Tag, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'

import type { JobStatusResponse, UploadResponse, UploadStageResponse } from '../../api/types'

const STAGE_LABELS: Record<string, string> = {
  parse: 'MinerU 解析',
  chunk: '分块',
  embed: '向量化 / Milvus',
  bm25: 'BM25 索引',
}

const STATUS_COLORS: Record<string, string> = {
  success: 'success',
  failed: 'error',
  skipped: 'default',
  running: 'processing',
  pending: 'default',
}

function stageColumns(): ColumnsType<UploadStageResponse> {
  return [
    {
      title: '阶段',
      dataIndex: 'name',
      width: 160,
      render: (name: string) => STAGE_LABELS[name] ?? name,
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={STATUS_COLORS[status] ?? 'default'}>{status}</Tag>
      ),
    },
    {
      title: '详情',
      dataIndex: 'detail',
      ellipsis: true,
    },
  ]
}

interface UploadResultPanelProps {
  result: UploadResponse
}

export function UploadResultPanel({ result }: UploadResultPanelProps) {
  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Typography.Title level={5} style={{ margin: 0 }}>
        入库结果
      </Typography.Title>
      <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }}>
        <Descriptions.Item label="文档 ID">{result.doc_id || '-'}</Descriptions.Item>
        <Descriptions.Item label="文件名">{result.filename}</Descriptions.Item>
        <Descriptions.Item label="行业">{result.industry_label || result.industry}</Descriptions.Item>
        <Descriptions.Item label="公司">{result.company_name || '-'}</Descriptions.Item>
        <Descriptions.Item label="股票代码">{result.stock_code || '-'}</Descriptions.Item>
        <Descriptions.Item label="显示名">{result.display_name || '-'}</Descriptions.Item>
        <Descriptions.Item label="Chunk 总数">{result.chunk_count}</Descriptions.Item>
        <Descriptions.Item label="可检索 Chunk">{result.retrievable_chunk_count}</Descriptions.Item>
        <Descriptions.Item label="Milvus 写入">{result.milvus_rows_inserted}</Descriptions.Item>
        <Descriptions.Item label="Milvus 总量">{result.milvus_total_rows}</Descriptions.Item>
        <Descriptions.Item label="BM25 总量">{result.bm25_total_chunks}</Descriptions.Item>
        <Descriptions.Item label="覆盖旧数据">
          {result.replaced_existing ? '是' : '否'}
        </Descriptions.Item>
      </Descriptions>
      {result.stages.length > 0 && (
        <Table
          rowKey={(row) => row.name}
          size="small"
          pagination={false}
          columns={stageColumns()}
          dataSource={result.stages}
        />
      )}
    </Space>
  )
}

interface AsyncJobResultPanelProps {
  job: JobStatusResponse
}

export function AsyncJobResultPanel({ job }: AsyncJobResultPanelProps) {
  const result = job.result ?? {}

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Typography.Title level={5} style={{ margin: 0 }}>
        任务结果
      </Typography.Title>
      <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }}>
        <Descriptions.Item label="任务 ID">{job.job_id}</Descriptions.Item>
        <Descriptions.Item label="状态">
          <Tag color={STATUS_COLORS[job.status] ?? 'default'}>{job.status}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="文档 ID">{String(result.doc_id ?? '-')}</Descriptions.Item>
        <Descriptions.Item label="Chunk 数量">{String(result.chunk_count ?? '-')}</Descriptions.Item>
        <Descriptions.Item label="Milvus 写入">
          {String(result.milvus_rows_inserted ?? '-')}
        </Descriptions.Item>
        <Descriptions.Item label="耗时">
          {job.duration_ms != null ? `${job.duration_ms} ms` : '-'}
        </Descriptions.Item>
      </Descriptions>
    </Space>
  )
}
