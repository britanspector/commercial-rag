import { Card, Space, Steps, Tag, Typography } from 'antd'

import type { SearchResponse } from '../../api/types'
import { CacheInfoPanel } from '../../components/CacheInfoPanel'
import { ChunkHitsTable } from './ChunkHitsTable'
import { QueryRewritePanel } from './QueryRewritePanel'

const { Text, Title } = Typography

interface SearchResultPanelProps {
  result: SearchResponse
}

export function SearchResultPanel({ result }: SearchResultPanelProps) {
  const { query_rewrite: rewrite, recall, rerank } = result

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card size="small">
        <Space wrap>
          <Text strong>检索问题：</Text>
          <Text>{result.query}</Text>
          <Text type="secondary">|</Text>
          <Text type="secondary">top rerank</Text>
          <Tag color="blue">{result.top_rerank_score.toFixed(3)}</Tag>
        </Space>
      </Card>

      {result.cache && <CacheInfoPanel cache={result.cache} />}

      <Steps
        current={2}
        size="small"
        items={[
          { title: 'Query Rewrite', description: '查询改写' },
          { title: `Recall (${recall.route})`, description: `Top-${recall.recall_top_k}` },
          { title: 'Rerank', description: `Top-${rerank.rerank_top_k}` },
        ]}
      />

      <div>
        <Title level={5}>1. Query Rewrite</Title>
        <QueryRewritePanel rewrite={rewrite} />
      </div>

      <div>
        <Title level={5}>
          2. 召回阶段
          <Tag style={{ marginLeft: 8 }}>{recall.route}</Tag>
          <Text type="secondary" style={{ fontSize: 14, fontWeight: 400 }}>
            {' '}
            命中 {recall.hit_count} / Top-{recall.recall_top_k}
          </Text>
        </Title>
        <ChunkHitsTable
          hits={recall.hits}
          stage="recall"
          emptyText="召回阶段无命中"
        />
      </div>

      <div>
        <Title level={5}>
          3. Rerank 阶段
          <Text type="secondary" style={{ fontSize: 14, fontWeight: 400 }}>
            {' '}
            命中 {rerank.hit_count} / Top-{rerank.rerank_top_k}，最高分{' '}
            {rerank.top_rerank_score.toFixed(3)}
          </Text>
        </Title>
        <ChunkHitsTable
          hits={rerank.hits}
          stage="rerank"
          emptyText="Rerank 阶段无命中"
        />
      </div>
    </Space>
  )
}
