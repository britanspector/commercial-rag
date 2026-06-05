import { Card, Col, Row, Statistic, Typography } from 'antd'

import {
  formatMetricValue,
  type MetricDefinition,
} from './evalMetrics'

const { Text } = Typography

interface EvalMetricsPanelProps {
  title: string
  metrics: Record<string, unknown>
  defs: MetricDefinition[]
}

export function EvalMetricsPanel({
  title,
  metrics,
  defs,
}: EvalMetricsPanelProps) {
  const items = defs
    .filter((def) => def.key in metrics)
    .map((def) => ({ def, value: metrics[def.key] }))

  if (items.length === 0) {
    return (
      <Card title={title}>
        <Text type="secondary">暂无可用指标数据。</Text>
      </Card>
    )
  }

  return (
    <Card title={title}>
      <Row gutter={[16, 16]}>
        {items.map(({ def, value }) => (
          <Col key={def.key} xs={12} sm={8} md={6}>
            <Statistic
              title={def.label}
              value={formatMetricValue(value)}
              valueStyle={{ fontSize: 20 }}
            />
            {def.description && (
              <Text type="secondary" style={{ fontSize: 12 }}>
                {def.description}
              </Text>
            )}
          </Col>
        ))}
      </Row>
    </Card>
  )
}
