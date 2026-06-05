import { Descriptions, Tag } from 'antd'

import type { QueryRewriteResponse } from '../../api/types'

interface QueryRewritePanelProps {
  rewrite: QueryRewriteResponse
}

export function QueryRewritePanel({ rewrite }: QueryRewritePanelProps) {
  return (
    <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }}>
      <Descriptions.Item label="原始问题" span={2}>
        {rewrite.original_query}
      </Descriptions.Item>
      <Descriptions.Item label="改写后 query" span={2}>
        {rewrite.query}
      </Descriptions.Item>
      <Descriptions.Item label="BM25 query" span={2}>
        {rewrite.bm25_query}
      </Descriptions.Item>
      <Descriptions.Item label="股票代码">
        {rewrite.stock_code || '-'}
      </Descriptions.Item>
      <Descriptions.Item label="问题类型">
        <Tag>{rewrite.query_type}</Tag>
      </Descriptions.Item>
      <Descriptions.Item label="混合向量权重">
        {rewrite.hybrid_vector_weight}
      </Descriptions.Item>
      <Descriptions.Item label="向量维度">
        {rewrite.query_vector_dim ?? '-'}
      </Descriptions.Item>
      <Descriptions.Item label="对比实体" span={2}>
        {rewrite.compare_entities.length > 0
          ? rewrite.compare_entities.join('、')
          : '-'}
      </Descriptions.Item>
    </Descriptions>
  )
}
