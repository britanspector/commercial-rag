import type { ReactNode } from 'react'
import { Alert, Space, Tag, Typography } from 'antd'

const { Paragraph, Title } = Typography

interface PageShellProps {
  title: string
  description: string
  apiEndpoints?: string[]
  plannedFeatures?: string[]
  comingSoon?: boolean
  children?: ReactNode
}

export function PageShell({
  title,
  description,
  apiEndpoints = [],
  plannedFeatures = [],
  comingSoon = false,
  children,
}: PageShellProps) {
  return (
    <div>
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <div>
          <Space align="center">
            <Title level={3} style={{ margin: 0 }}>
              {title}
            </Title>
            {comingSoon && <Tag color="default">预留</Tag>}
          </Space>
          <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
            {description}
          </Paragraph>
        </div>

        {apiEndpoints.length > 0 && (
          <div>
            <Typography.Text type="secondary">对接接口：</Typography.Text>
            <Space wrap style={{ marginTop: 4 }}>
              {apiEndpoints.map((endpoint) => (
                <Tag key={endpoint}>{endpoint}</Tag>
              ))}
            </Space>
          </div>
        )}

        {comingSoon && (
          <Alert
            type="info"
            showIcon
            message="该页面为后续扩展入口，当前仅展示占位内容。"
          />
        )}

        {children ?? (
          plannedFeatures.length > 0 && (
            <div>
              <Title level={5}>待实现功能</Title>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {plannedFeatures.map((feature) => (
                  <li key={feature}>{feature}</li>
                ))}
              </ul>
            </div>
          )
        )}
      </Space>
    </div>
  )
}
