import { useState } from 'react'
import { SearchOutlined } from '@ant-design/icons'
import {
  Alert,
  Button,
  Col,
  Collapse,
  Empty,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Spin,
} from 'antd'

import { postSearch } from '../../api/search'
import type { QueryType, RecallRoute, SearchResponse } from '../../api/types'
import { PageShell } from '../../components/PageShell'
import { getNavItemByPath } from '../../navigation'
import { SearchResultPanel } from './SearchResultPanel'

type SearchPhase = 'idle' | 'loading' | 'success' | 'error'

interface SearchFormValues {
  query: string
  stock_code?: string
  query_type: QueryType
  recall_route: RecallRoute
  recall_top_k?: number
  rerank_top_k?: number
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

export function SearchPage() {
  const navItem = getNavItemByPath('/search')!
  const [form] = Form.useForm<SearchFormValues>()
  const [phase, setPhase] = useState<SearchPhase>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [result, setResult] = useState<SearchResponse | null>(null)

  const isLoading = phase === 'loading'

  const handleSubmit = async () => {
    const values = await form.validateFields()
    const query = values.query.trim()
    if (!query) {
      setErrorMessage('请输入检索问题')
      setPhase('error')
      return
    }

    setPhase('loading')
    setErrorMessage(null)
    setResult(null)

    try {
      const response = await postSearch({
        query,
        stock_code: values.stock_code?.trim() ?? '',
        query_type: values.query_type,
        recall_route: values.recall_route,
        recall_top_k: values.recall_top_k,
        rerank_top_k: values.rerank_top_k,
      })
      setResult(response)
      setPhase('success')
    } catch (error) {
      setPhase('error')
      setErrorMessage(error instanceof Error ? error.message : '检索请求失败')
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
        <Col xs={24} lg={9}>
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
              label="检索问题"
              name="query"
              rules={[{ required: true, message: '请输入检索问题' }]}
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

            <Form.Item label="召回路线" name="recall_route">
              <Select options={RECALL_ROUTE_OPTIONS} disabled={isLoading} />
            </Form.Item>

            <Collapse
              ghost
              items={[
                {
                  key: 'advanced',
                  label: '高级参数',
                  children: (
                    <>
                      <Form.Item
                        label="召回 Top-K"
                        name="recall_top_k"
                        extra="留空使用后端默认值"
                      >
                        <InputNumber
                          min={1}
                          max={200}
                          style={{ width: '100%' }}
                          disabled={isLoading}
                        />
                      </Form.Item>
                      <Form.Item
                        label="Rerank Top-K"
                        name="rerank_top_k"
                        extra="留空使用后端默认值"
                      >
                        <InputNumber
                          min={1}
                          max={50}
                          style={{ width: '100%' }}
                          disabled={isLoading}
                        />
                      </Form.Item>
                    </>
                  ),
                },
              ]}
            />

            <Form.Item>
              <Space>
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<SearchOutlined />}
                  loading={isLoading}
                >
                  检索
                </Button>
                <Button onClick={handleReset} disabled={isLoading}>
                  清空
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Col>

        <Col xs={24} lg={15}>
          {phase === 'idle' && (
            <Empty
              description="输入问题后点击「检索」，将展示 Query Rewrite → Recall → Rerank 全链路结果"
              style={{ marginTop: 48 }}
            />
          )}

          {isLoading && (
            <div style={{ textAlign: 'center', marginTop: 80 }}>
              <Spin size="large" tip="正在执行检索与重排，请稍候…" />
            </div>
          )}

          {phase === 'error' && errorMessage && (
            <Alert
              type="error"
              showIcon
              message="检索失败"
              description={errorMessage}
              style={{ marginTop: 24 }}
            />
          )}

          {phase === 'success' && result && (
            <SearchResultPanel result={result} />
          )}
        </Col>
      </Row>
    </PageShell>
  )
}
