import {
  Alert,
  Card,
  Collapse,
  Descriptions,
  Space,
  Tag,
  Typography,
} from 'antd'

import type { ChatResponse } from '../../api/types'
import { CitationList } from './CitationList'
import { RerankHitsTable } from './RerankHitsTable'

const { Paragraph, Title, Text } = Typography

interface ChatResultPanelProps {
  result: ChatResponse
}

function formatScore(score: number): string {
  return score.toFixed(3)
}

export function ChatResultPanel({ result }: ChatResultPanelProps) {
  const evidence = result.evidence_check

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card
        title="回答"
        extra={
          <Space>
            <Text type="secondary">top rerank</Text>
            <Tag color={result.refused ? 'warning' : 'success'}>
              {formatScore(result.top_rerank_score)}
            </Tag>
            {result.refused ? (
              <Tag color="error">已拒答</Tag>
            ) : (
              <Tag color="success">已回答</Tag>
            )}
          </Space>
        }
      >
        {result.refused ? (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <Alert
              type="warning"
              showIcon
              message="系统拒答"
              description={
                <Space direction="vertical" size={4}>
                  {result.refusal_reason && (
                    <Text>
                      原因代码：<Text code>{result.refusal_reason}</Text>
                    </Text>
                  )}
                  <Text>
                    {result.refusal_message ||
                      evidence?.refusal_message ||
                      result.answer}
                  </Text>
                </Space>
              }
            />
            {result.answer &&
              result.answer !== result.refusal_message &&
              !result.answer.startsWith('[拒答]') && (
                <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
                  {result.answer}
                </Paragraph>
              )}
          </Space>
        ) : (
          <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>
            {result.answer || '（无答案内容）'}
          </Paragraph>
        )}
      </Card>

      <div>
        <Title level={5}>引用来源</Title>
        <CitationList
          citations={result.citations}
          rerankHits={result.rerank_hits}
        />
      </div>

      <Collapse
        items={[
          {
            key: 'debug',
            label: '检索与证据详情（便于后续 Agent Trace 扩展）',
            children: (
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }}>
                  <Descriptions.Item label="证据校验">
                    <Tag color={evidence?.passed ? 'success' : 'error'}>
                      {evidence?.passed ? '通过' : '未通过'}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="top rerank 分数">
                    {formatScore(evidence?.top_rerank_score ?? result.top_rerank_score)}
                  </Descriptions.Item>
                  <Descriptions.Item label="引用数量">
                    {evidence?.citation_count ?? result.citations.length}
                  </Descriptions.Item>
                  <Descriptions.Item label="拒答原因">
                    {result.refusal_reason || '-'}
                  </Descriptions.Item>
                </Descriptions>

                {evidence?.checks && evidence.checks.length > 0 && (
                  <div>
                    <Text type="secondary">校验项</Text>
                    <ul style={{ margin: '8px 0 0', paddingLeft: 20 }}>
                      {evidence.checks.map((check, index) => (
                        <li key={index}>
                          {Object.entries(check)
                            .map(([key, value]) => `${key}: ${String(value)}`)
                            .join(' · ')}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div>
                  <Text strong>Rerank 命中（Top-{result.rerank_hits.length}）</Text>
                  <RerankHitsTable hits={result.rerank_hits} />
                </div>
              </Space>
            ),
          },
        ]}
      />
    </Space>
  )
}
