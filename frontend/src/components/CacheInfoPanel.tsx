import { Card, Descriptions, Space, Tag, Typography } from 'antd'

import type { CacheInfoResponse } from '../api/types'

const { Text } = Typography

interface CacheInfoPanelProps {
  cache: CacheInfoResponse
  compact?: boolean
}

function formatMs(value: number): string {
  return `${value.toFixed(1)} ms`
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function sourceLabel(source: string): string {
  switch (source) {
    case 'l1_exact':
      return 'L1 精确命中'
    case 'l2_semantic':
      return 'L2 语义命中'
    case 'pipeline':
      return 'Pipeline 执行'
    case 'none':
      return '未命中'
    default:
      return source || '未知'
  }
}

function sourceColor(source: string, hit: boolean): string {
  if (!hit) return 'default'
  if (source === 'l1_exact') return 'success'
  if (source === 'l2_semantic') return 'processing'
  return 'warning'
}

export function CacheInfoPanel({ cache, compact = false }: CacheInfoPanelProps) {
  const hitRateText = cache.hit ? '100%' : '0%'

  if (compact) {
    return (
      <Space wrap size="small">
        <Tag color={cache.hit ? 'success' : 'default'}>
          {cache.hit ? '缓存命中' : '缓存未命中'}
        </Tag>
        <Tag color={sourceColor(cache.source, cache.hit)}>
          {sourceLabel(cache.source)}
        </Tag>
        <Text type="secondary">延迟 {formatMs(cache.latency_ms)}</Text>
        {cache.similarity != null && cache.hit && (
          <Text type="secondary">相似度 {formatPercent(cache.similarity)}</Text>
        )}
      </Space>
    )
  }

  return (
    <Card size="small" title="语义缓存（本次请求）">
      <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }}>
        <Descriptions.Item label="命中">
          <Tag color={cache.hit ? 'success' : 'default'}>
            {cache.hit ? '是' : '否'}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="来源">
          <Tag color={sourceColor(cache.source, cache.hit)}>
            {sourceLabel(cache.source)}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="命中率（本次）">{hitRateText}</Descriptions.Item>
        <Descriptions.Item label="总延迟">{formatMs(cache.latency_ms)}</Descriptions.Item>
        <Descriptions.Item label="Lookup 耗时">
          {formatMs(cache.lookup_ms)}
        </Descriptions.Item>
        <Descriptions.Item label="Pipeline 耗时">
          {formatMs(cache.pipeline_ms)}
        </Descriptions.Item>
        {cache.similarity != null && (
          <Descriptions.Item label="语义相似度">
            {formatPercent(cache.similarity)}
          </Descriptions.Item>
        )}
        <Descriptions.Item label="原因">{cache.reason || '-'}</Descriptions.Item>
        <Descriptions.Item label="安全校验">
          <Tag color={cache.safety_ok ? 'success' : 'error'}>
            {cache.safety_ok ? '通过' : '拒绝'}
          </Tag>
        </Descriptions.Item>
        {cache.safety_reason && (
          <Descriptions.Item label="安全说明" span={2}>
            {cache.safety_reason}
          </Descriptions.Item>
        )}
        <Descriptions.Item label="向量检索">
          {cache.vector_retrieval ? '已执行' : '已跳过'}
        </Descriptions.Item>
        <Descriptions.Item label="LLM 调用">
          {cache.llm_called ? '已调用' : '未调用'}
        </Descriptions.Item>
      </Descriptions>
    </Card>
  )
}
