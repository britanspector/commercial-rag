import { Table, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'

import type { RetrievedChunkResponse } from '../../api/types'

const { Paragraph } = Typography

interface RerankHitsTableProps {
  hits: RetrievedChunkResponse[]
}

export function RerankHitsTable({ hits }: RerankHitsTableProps) {
  const columns: ColumnsType<RetrievedChunkResponse> = [
    {
      title: '排名',
      dataIndex: 'rank',
      width: 64,
    },
    {
      title: '公司',
      dataIndex: 'company_name',
      width: 120,
      ellipsis: true,
    },
    {
      title: '章节',
      dataIndex: 'section_title',
      width: 140,
      ellipsis: true,
    },
    {
      title: '页码',
      width: 90,
      render: (_, row) =>
        row.page_start === row.page_end
          ? row.page_start
          : `${row.page_start}-${row.page_end}`,
    },
    {
      title: 'rerank',
      dataIndex: 'score_rerank',
      width: 80,
      render: (value: number | null | undefined) =>
        value != null ? value.toFixed(3) : '-',
    },
    {
      title: 'chunk 摘要',
      dataIndex: 'text',
      ellipsis: true,
      render: (text: string) => (
        <Paragraph
          ellipsis={{ rows: 2, expandable: true, symbol: '展开' }}
          style={{ marginBottom: 0 }}
        >
          {text}
        </Paragraph>
      ),
    },
  ]

  return (
    <Table
      rowKey="chunk_id"
      size="small"
      pagination={false}
      columns={columns}
      dataSource={hits}
      locale={{ emptyText: '无 Rerank 命中' }}
      style={{ marginTop: 8 }}
    />
  )
}
