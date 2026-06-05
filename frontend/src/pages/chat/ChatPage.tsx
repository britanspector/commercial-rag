import { useState } from 'react'
import { SendOutlined } from '@ant-design/icons'
import {
  Alert,
  Button,
  Col,
  Empty,
  Form,
  Input,
  Row,
  Select,
  Space,
  Spin,
} from 'antd'

import { postChat } from '../../api/chat'
import type { ChatResponse, QueryType, RecallRoute } from '../../api/types'
import { PageShell } from '../../components/PageShell'
import { getNavItemByPath } from '../../navigation'
import { ChatResultPanel } from './ChatResultPanel'

type ChatPhase = 'idle' | 'loading' | 'success' | 'error'

interface ChatFormValues {
  query: string
  stock_code?: string
  query_type: QueryType
  recall_route: RecallRoute
}

const QUERY_TYPE_OPTIONS = [
  { value: 'factual', label: '事实题 (factual)' },
  { value: 'comparative', label: '对比题 (comparative)' },
  { value: 'summary', label: '总结题 (summary)' },
]

const RECALL_ROUTE_OPTIONS = [
  { value: 'hybrid', label: '混合召回 (hybrid)' },
  { value: 'vector', label: '向量 (vector)' },
  { value: 'bm25', label: 'BM25 (bm25)' },
]

export function ChatPage() {
  const navItem = getNavItemByPath('/chat')!
  const [form] = Form.useForm<ChatFormValues>()
  const [phase, setPhase] = useState<ChatPhase>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [result, setResult] = useState<ChatResponse | null>(null)

  const isLoading = phase === 'loading'

  const handleSubmit = async () => {
    const values = await form.validateFields()
    const query = values.query.trim()
    if (!query) {
      setErrorMessage('请输入问题')
      setPhase('error')
      return
    }

    setPhase('loading')
    setErrorMessage(null)
    setResult(null)

    try {
      const response = await postChat({
        query,
        stock_code: values.stock_code?.trim() ?? '',
        query_type: values.query_type,
        recall_route: values.recall_route,
      })
      setResult(response)
      setPhase('success')
    } catch (error) {
      setPhase('error')
      setErrorMessage(error instanceof Error ? error.message : '问答请求失败')
    }
  }

  const handleReset = () => {
    form.resetFields()
    setPhase('idle')
    setErrorMessage(null)
    setResult(null)
  }

  return (
    <PageShell
      title={navItem.label}
      description={navItem.description}
      apiEndpoints={navItem.apiEndpoints}
    >
      <Row gutter={[24, 24]}>
        <Col xs={24} lg={10}>
          <Form
            form={form}
            layout="vertical"
            initialValues={{
              query_type: 'factual',
              recall_route: 'hybrid',
            }}
            onFinish={() => void handleSubmit()}
          >
            <Form.Item
              label="问题"
              name="query"
              rules={[{ required: true, message: '请输入问题' }]}
            >
              <Input.TextArea
                rows={4}
                placeholder="例如：澜起科技 2026 年 EPS 预测是多少？"
                disabled={isLoading}
                showCount
                maxLength={500}
              />
            </Form.Item>

            <Form.Item label="股票代码（可选）" name="stock_code">
              <Input
                placeholder="例如：688008"
                disabled={isLoading}
                allowClear
              />
            </Form.Item>

            <Form.Item label="问题类型" name="query_type">
              <Select options={QUERY_TYPE_OPTIONS} disabled={isLoading} />
            </Form.Item>

            <Form.Item
              label="召回路线"
              name="recall_route"
              extra="默认 hybrid，与后端评测最优配置一致"
            >
              <Select options={RECALL_ROUTE_OPTIONS} disabled={isLoading} />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<SendOutlined />}
                  loading={isLoading}
                >
                  提问
                </Button>
                <Button onClick={handleReset} disabled={isLoading}>
                  清空
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Col>

        <Col xs={24} lg={14}>
          {phase === 'idle' && (
            <Empty
              description="输入问题后点击「提问」，将调用 POST /chat 获取带引用的回答"
              style={{ marginTop: 48 }}
            />
          )}

          {isLoading && (
            <div style={{ textAlign: 'center', marginTop: 80 }}>
              <Spin size="large" tip="正在检索并生成回答，请稍候…" />
            </div>
          )}

          {phase === 'error' && errorMessage && (
            <Alert
              type="error"
              showIcon
              message="问答失败"
              description={errorMessage}
              style={{ marginTop: 24 }}
            />
          )}

          {phase === 'success' && result && <ChatResultPanel result={result} />}
        </Col>
      </Row>
    </PageShell>
  )
}
