import { Link } from 'react-router-dom'
import { Card, Col, Row, Space, Tag, Typography } from 'antd'

import { BackendStatusCard } from '../components/BackendStatusCard'
import {
  getCoreNavItems,
  getFutureNavItems,
  type NavItem,
} from '../navigation'

const { Paragraph, Title, Text } = Typography

function NavEntryCard({ item }: { item: NavItem }) {
  return (
    <Card
      hoverable
      title={
        <Space>
          {item.icon}
          <span>{item.label}</span>
        </Space>
      }
      extra={
        item.comingSoon ? (
          <Tag>预留</Tag>
        ) : (
          <Link to={item.path}>进入</Link>
        )
      }
    >
      <Paragraph type="secondary" style={{ marginBottom: 8 }}>
        {item.description}
      </Paragraph>
      {item.apiEndpoints.length > 0 && (
        <Space wrap size={[4, 4]}>
          {item.apiEndpoints.map((endpoint) => (
            <Tag key={endpoint}>{endpoint}</Tag>
          ))}
        </Space>
      )}
    </Card>
  )
}

export function HomePage() {
  const coreItems = getCoreNavItems()
  const futureItems = getFutureNavItems()

  return (
    <div>
      <Title level={3}>项目概览</Title>
      <Paragraph>
        金融研报 RAG 管理台。前端通过 HTTP 调用 FastAPI 后端，负责页面展示与交互，
        不包含检索、Rerank 或评测逻辑。
      </Paragraph>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={10}>
          <BackendStatusCard />
        </Col>
        <Col xs={24} lg={14}>
          <Card title="使用说明">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              <li>左侧导航切换各功能页面</li>
              <li>各页面按统一结构预留接口对接区域</li>
              <li>下一步将实现 PDF 上传并调用 POST /upload</li>
            </ul>
          </Card>
        </Col>
      </Row>

      <Title level={4}>RAG 功能</Title>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {coreItems.map((item) => (
          <Col key={item.path} xs={24} md={12} xl={6}>
            <NavEntryCard item={item} />
          </Col>
        ))}
      </Row>

      <Title level={4}>
        后续扩展 <Text type="secondary" style={{ fontSize: 14 }}>（占位）</Text>
      </Title>
      <Row gutter={[16, 16]}>
        {futureItems.map((item) => (
          <Col key={item.path} xs={24} md={12} xl={8}>
            <NavEntryCard item={item} />
          </Col>
        ))}
      </Row>
    </div>
  )
}
