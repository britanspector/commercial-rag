import { Space, Table, Tag, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'

import type { RetrievedChunkResponse } from '../../api/types'

const { Paragraph, Text } = Typography

type HitStage = 'recall' | 'rerank'

interface ChunkHitsTableProps {
  hits: RetrievedChunkResponse[]
  stage: HitStage
  emptyText?: string
}

function formatPage(row: RetrievedChunkResponse): string {
  if (row.page_start === row.page_end) {
    return String(row.page_start)
  }
  return `${row.page_start}-${row.page_end}`
}

function formatScore(value: number | null | undefined): string {
  return value != null ? value.toFixed(3) : '-'
}

export function ChunkHitsTable({
  hits,
  stage,
  emptyText = '无命中结果',
}: ChunkHitsTableProps) {
  const recallColumns: ColumnsType<RetrievedChunkResponse> = [
    { title: '排名', dataIndex: 'rank', width: 64 },
    {
      title: '公司',
      dataIndex: 'company_name',
      width: 110,
      ellipsis: true,
    },
    {
      title: '章节',
      dataIndex: 'section_title',
      width: 130,
      ellipsis: true,
    },
    {
      title: '页码',
      width: 72,
      render: (_, row) => formatPage(row),
    },
    {
      title: '文档',
      dataIndex: 'display_name',
      width: 140,
      ellipsis: true,
      render: (value: string, row) => value || row.doc_id || '-',
    },
    {
      title: 'recall',
      dataIndex: 'score_recall',
      width: 72,
      render: (value: number) => formatScore(value),
    },
    {
      title: 'vector',
      dataIndex: 'score_vector',
      width: 72,
      render: (value: number | null) => formatScore(value),
    },
    {
      title: 'BM25',
      dataIndex: 'score_bm25',
      width: 72,
      render: (value: number | null) => formatScore(value),
    },
    {
      title: 'score',
      dataIndex: 'score',
      width: 72,
      render: (value: number) => formatScore(value),
    },
  ]

  const rerankColumns: ColumnsType<RetrievedChunkResponse> = [
    { title: '排名', dataIndex: 'rank', width: 64 },
    {
      title: '公司',
      dataIndex: 'company_name',
      width: 110,
      ellipsis: true,
    },
    {
      title: '章节',
      dataIndex: 'section_title',
      width: 130,
      ellipsis: true,
    },
    {
      title: '页码',
      width: 72,
      render: (_, row) => formatPage(row),
    },
    {
      title: '文档',
      dataIndex: 'display_name',
      width: 140,
      ellipsis: true,
      render: (value: string, row) => value || row.doc_id || '-',
    },
    {
      title: 'rerank',
      dataIndex: 'score_rerank',
      width: 80,
      render: (value: number | null) => (
        <Tag color="purple">{formatScore(value)}</Tag>
      ),
    },
    {
      title: 'recall',
      dataIndex: 'score_recall',
      width: 72,
      render: (value: number) => formatScore(value),
    },
    {
      title: 'vector',
      dataIndex: 'score_vector',
      width: 72,
      render: (value: number | null) => formatScore(value),
    },
    {
      title: 'BM25',
      dataIndex: 'score_bm25',
      width: 72,
      render: (value: number | null) => formatScore(value),
    },
  ]

  return (
    <Table
      rowKey="chunk_id"
      size="small"
      pagination={false}
      scroll={{ x: 900 }}
      columns={stage === 'recall' ? recallColumns : rerankColumns}
      dataSource={hits}
      locale={{ emptyText }}
      expandable={{
        expandedRowRender: (row) => (
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            <Text type="secondary">
              chunk_id: {row.chunk_id}
              {row.source_pdf_path ? ` · ${row.source_pdf_path}` : ''}
            </Text>
            <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
              {row.text}
            </Paragraph>
          </Space>
        ),
        rowExpandable: (row) => Boolean(row.text),
      }}
    />
  )
}
