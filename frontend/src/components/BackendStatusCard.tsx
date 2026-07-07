import { useEffect, useState } from 'react'
import { Alert, Card, Descriptions, Spin, Tag } from 'antd'

import { fetchHealth } from '../api/health'
import type { HealthResponse } from '../api/types'

export function BackendStatusCard() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadHealth() {
      setLoading(true)
      setError(null)
      try {
        const data = await fetchHealth()
        if (!cancelled) {
          setHealth(data)
        }
      } catch (err) {
        if (!cancelled) {
          setHealth(null)
          setError(err instanceof Error ? err.message : '无法连接后端')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadHealth()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <Card title="后端连接状态">
      {loading && <Spin />}
      {!loading && error && (
        <Alert
          type="error"
          showIcon
          message="后端不可用"
          description={error}
        />
      )}
      {!loading && !error && health && (
        <Descriptions column={1} size="small" bordered>
          <Descriptions.Item label="服务状态">
            <Tag color="success">{health.status}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="Pipeline 就绪">
            <Tag color={health.pipeline_ready ? 'success' : 'warning'}>
              {health.pipeline_ready ? '是' : '否（首次请求时懒加载）'}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="模型已加载">
            <Tag color={health.models_loaded ? 'success' : 'default'}>
              {health.models_loaded ? '是' : '否'}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="默认召回路线">
            {String(health.defaults.recall_route ?? '-')}
          </Descriptions.Item>
          <Descriptions.Item label="审计库">
            {health.audit.enabled ? '已启用' : '未启用'}
          </Descriptions.Item>
          {health.cache && Object.keys(health.cache).length > 0 && (
            <>
              <Descriptions.Item label="语义缓存">
                <Tag color={health.cache.active ? 'success' : 'default'}>
                  {health.cache.active ? '已启用' : '未启用'}
                </Tag>
              </Descriptions.Item>
              {typeof health.cache.total_hit_rate === 'number' && (
                <Descriptions.Item label="累计命中率">
                  {(Number(health.cache.total_hit_rate) * 100).toFixed(1)}%
                </Descriptions.Item>
              )}
              {typeof health.cache.avg_latency_ms === 'number' && (
                <Descriptions.Item label="平均延迟">
                  {Number(health.cache.avg_latency_ms).toFixed(1)} ms
                </Descriptions.Item>
              )}
            </>
          )}
        </Descriptions>
      )}
    </Card>
  )
}
