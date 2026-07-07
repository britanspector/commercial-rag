import { useCallback, useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Row,
  Space,
  Spin,
  Statistic,
  Tag,
  Typography,
} from 'antd'
import { ReloadOutlined } from '@ant-design/icons'

import { fetchCacheStats } from '../../api/cache'
import type { CacheStatsResponse } from '../../api/types'

const { Text } = Typography

function formatRate(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatMs(value: number): string {
  return `${value.toFixed(1)} ms`
}

export function CacheStatsPanel() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<CacheStatsResponse | null>(null)

  const loadStats = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchCacheStats()
      setStats(data)
    } catch (err) {
      setStats(null)
      setError(err instanceof Error ? err.message : '无法获取缓存统计')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadStats()
  }, [loadStats])

  if (loading && !stats) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <Spin size="large" tip="正在加载缓存统计…" />
      </div>
    )
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Text type="secondary">进程内累计统计，数据来自 GET /cache/stats</Text>
        <Button icon={<ReloadOutlined />} onClick={() => void loadStats()} loading={loading}>
          刷新
        </Button>
      </div>

      {error && (
        <Alert type="error" showIcon message="加载失败" description={error} />
      )}

      {stats && (
        <>
          <Card size="small">
            <Space wrap>
              <Text strong>缓存状态</Text>
              <Tag color={stats.active ? 'success' : 'default'}>
                {stats.active ? '已启用' : '未启用'}
              </Tag>
              <Text type="secondary">累计请求 {stats.requests}</Text>
            </Space>
          </Card>

          <Card title="命中率">
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="总命中率" value={formatRate(stats.total_hit_rate)} />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="L1 命中率" value={formatRate(stats.l1_hit_rate)} />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="L2 命中率" value={formatRate(stats.l2_hit_rate)} />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="Lookup 命中率" value={formatRate(stats.lookup_hit_rate)} />
              </Col>
            </Row>
          </Card>

          <Card title="延迟">
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="平均延迟" value={formatMs(stats.avg_latency_ms)} />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="命中平均延迟" value={formatMs(stats.avg_hit_latency_ms)} />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="未命中平均延迟" value={formatMs(stats.avg_miss_latency_ms)} />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="平均节省延迟" value={formatMs(stats.avg_latency_saved_ms)} />
              </Col>
            </Row>
          </Card>

          <Card title="资源节省">
            <Row gutter={[16, 16]}>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="跳过向量检索" value={stats.vector_retrievals_saved} />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="减少 LLM 调用" value={stats.llm_calls_saved} />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic
                  title="LLM 调用降幅"
                  value={formatRate(stats.llm_call_reduction_rate)}
                />
              </Col>
              <Col xs={12} sm={8} md={6}>
                <Statistic title="安全拒绝" value={stats.safety_rejects} />
              </Col>
            </Row>
          </Card>

          <Card title="存储与后端">
            <Descriptions bordered size="small" column={{ xs: 1, sm: 2 }}>
              <Descriptions.Item label="Lookup 次数">{stats.lookups}</Descriptions.Item>
              <Descriptions.Item label="L1 命中">{stats.hits_l1}</Descriptions.Item>
              <Descriptions.Item label="L2 命中">{stats.hits_l2}</Descriptions.Item>
              <Descriptions.Item label="未命中">{stats.misses}</Descriptions.Item>
              <Descriptions.Item label="写入次数">{stats.stores}</Descriptions.Item>
              <Descriptions.Item label="L1 条目">{stats.exact_entries}</Descriptions.Item>
              <Descriptions.Item label="L2 条目">{stats.semantic_entries}</Descriptions.Item>
              <Descriptions.Item label="后端状态" span={2}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: 12 }}>
                  {JSON.stringify(stats.backends, null, 2)}
                </pre>
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </>
      )}
    </Space>
  )
}