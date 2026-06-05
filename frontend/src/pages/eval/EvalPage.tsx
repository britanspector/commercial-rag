import { useRef, useState } from 'react'
import { ExperimentOutlined } from '@ant-design/icons'
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  InputNumber,
  Progress,
  Row,
  Select,
  Space,
  Switch,
  Typography,
} from 'antd'

import { startEval } from '../../api/eval'
import { pollJobUntilDone } from '../../api/jobs'
import type { EvalJobRequest, EvalType, JobStatusResponse } from '../../api/types'
import { PageShell } from '../../components/PageShell'
import { getNavItemByPath } from '../../navigation'
import { EvalResultPanel } from './EvalResultPanel'

type EvalPhase = 'idle' | 'running' | 'success' | 'error'

interface EvalFormValues {
  eval_type: EvalType
  limit?: number
  skip_ragas: boolean
  save_detail: boolean
  resume: boolean
  compare_routes: boolean
  route: 'vector' | 'bm25' | 'hybrid'
  top_k: number
  legacy_retriever: boolean
  pipeline_stage: 'recall' | 'rerank'
}

const EVAL_TYPE_OPTIONS = [
  {
    value: 'generation',
    label: '生成评测 (generation)',
    desc: '150 题 Pipeline 全链路：引用、拒答、检索命中等',
  },
  {
    value: 'retrieval',
    label: '检索评测 (retrieval)',
    desc: 'Recall@K、MRR、Context Precision',
  },
  {
    value: 'ragas',
    label: 'RAGAS 补跑 (ragas)',
    desc: 'Faithfulness、Answer Relevancy（需已有 generation 明细）',
  },
]

const DEFAULT_QUESTIONS_PATH = 'data/eval/eval_questions.jsonl'

export function EvalPage() {
  const navItem = getNavItemByPath('/eval')!
  const [form] = Form.useForm<EvalFormValues>()
  const evalType = Form.useWatch('eval_type', form) ?? 'generation'

  const [phase, setPhase] = useState<EvalPhase>('idle')
  const [statusText, setStatusText] = useState('')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [jobResult, setJobResult] = useState<JobStatusResponse | null>(null)
  const [activeEvalType, setActiveEvalType] = useState<EvalType>('generation')
  const abortRef = useRef(false)

  const isRunning = phase === 'running'

  const buildRequest = (values: EvalFormValues): EvalJobRequest => {
    const base: EvalJobRequest = {
      eval_type: values.eval_type,
      limit: values.limit,
    }
    if (values.eval_type === 'generation') {
      return {
        ...base,
        skip_ragas: values.skip_ragas,
        save_detail: values.save_detail,
        resume: values.resume,
      }
    }
    if (values.eval_type === 'retrieval') {
      return {
        ...base,
        compare_routes: values.compare_routes,
        route: values.route,
        top_k: values.top_k,
        legacy_retriever: values.legacy_retriever,
        pipeline_stage: values.pipeline_stage,
      }
    }
    return {
      ...base,
      resume: values.resume,
    }
  }

  const handleSubmit = async () => {
    const values = await form.validateFields()
    abortRef.current = false
    setPhase('running')
    setErrorMessage(null)
    setJobResult(null)
    setActiveEvalType(values.eval_type)
    setStatusText('正在提交评测任务…')

    try {
      const started = await startEval(buildRequest(values))
      setStatusText(`任务已提交 (${started.job_id})，等待执行…`)

      const job = await pollJobUntilDone(started.job_id, {
        intervalMs: 3000,
        onUpdate: (current) => {
          setStatusText(
            current.progress
              ? `任务 ${current.job_id}：${current.status}（${current.progress}）`
              : `任务 ${current.job_id}：${current.status}`,
          )
        },
      })

      if (abortRef.current) {
        return
      }

      if (job.status === 'failed') {
        throw new Error(job.error || '评测任务失败')
      }

      setJobResult(job)
      setPhase('success')
      setStatusText('评测完成')
    } catch (error) {
      if (abortRef.current) {
        return
      }
      setPhase('error')
      setErrorMessage(error instanceof Error ? error.message : '评测请求失败')
      setStatusText('')
    }
  }

  const handleReset = () => {
    abortRef.current = true
    setPhase('idle')
    setErrorMessage(null)
    setJobResult(null)
    setStatusText('')
  }

  return (
    <PageShell
      title={navItem.label}
      description={navItem.description}
      apiEndpoints={navItem.apiEndpoints}
    >
      <Row gutter={[24, 24]}>
        <Col xs={24} lg={10}>
          <Card title="评测配置" size="small">
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
              message="评测集"
              description={
                <>
                  后端使用内置评测集 <Typography.Text code>{DEFAULT_QUESTIONS_PATH}</Typography.Text>
                  （150 题）。当前 API 不支持上传自定义评测集。
                </>
              }
            />

            <Form
              form={form}
              layout="vertical"
              initialValues={{
                eval_type: 'generation',
                skip_ragas: true,
                save_detail: true,
                resume: false,
                compare_routes: false,
                route: 'hybrid',
                top_k: 10,
                legacy_retriever: false,
                pipeline_stage: 'rerank',
              }}
              onFinish={() => void handleSubmit()}
            >
              <Form.Item
                label="评测类型"
                name="eval_type"
                rules={[{ required: true }]}
              >
                <Select
                  disabled={isRunning}
                  options={EVAL_TYPE_OPTIONS.map((item) => ({
                    value: item.value,
                    label: item.label,
                  }))}
                />
              </Form.Item>

              <Typography.Paragraph type="secondary" style={{ marginTop: -12 }}>
                {EVAL_TYPE_OPTIONS.find((item) => item.value === evalType)?.desc}
              </Typography.Paragraph>

              <Form.Item
                label="题目上限 (limit)"
                name="limit"
                extra="留空表示全量 150 题；调试可设 5"
              >
                <InputNumber
                  min={1}
                  max={500}
                  style={{ width: '100%' }}
                  disabled={isRunning}
                />
              </Form.Item>

              {evalType === 'generation' && (
                <>
                  <Form.Item
                    label="跳过内嵌 RAGAS"
                    name="skip_ragas"
                    valuePropName="checked"
                  >
                    <Switch disabled={isRunning} />
                  </Form.Item>
                  <Form.Item
                    label="保存逐题明细"
                    name="save_detail"
                    valuePropName="checked"
                  >
                    <Switch disabled={isRunning} />
                  </Form.Item>
                  <Form.Item label="断点续跑" name="resume" valuePropName="checked">
                    <Switch disabled={isRunning} />
                  </Form.Item>
                </>
              )}

              {evalType === 'retrieval' && (
                <>
                  <Form.Item label="三路对比" name="compare_routes" valuePropName="checked">
                    <Switch disabled={isRunning} />
                  </Form.Item>
                  <Form.Item label="召回路线" name="route">
                    <Select
                      disabled={isRunning}
                      options={[
                        { value: 'hybrid', label: 'hybrid' },
                        { value: 'vector', label: 'vector' },
                        { value: 'bm25', label: 'bm25' },
                      ]}
                    />
                  </Form.Item>
                  <Form.Item label="Top-K" name="top_k">
                    <InputNumber
                      min={1}
                      max={50}
                      style={{ width: '100%' }}
                      disabled={isRunning}
                    />
                  </Form.Item>
                  <Form.Item
                    label="Pipeline 阶段"
                    name="pipeline_stage"
                  >
                    <Select
                      disabled={isRunning}
                      options={[
                        { value: 'rerank', label: 'rerank' },
                        { value: 'recall', label: 'recall' },
                      ]}
                    />
                  </Form.Item>
                  <Form.Item
                    label="旧版 Retriever"
                    name="legacy_retriever"
                    valuePropName="checked"
                  >
                    <Switch disabled={isRunning} />
                  </Form.Item>
                </>
              )}

              {evalType === 'ragas' && (
                <Form.Item label="断点续跑" name="resume" valuePropName="checked">
                  <Switch disabled={isRunning} />
                </Form.Item>
              )}

              <Form.Item>
                <Space>
                  <Button
                    type="primary"
                    htmlType="submit"
                    icon={<ExperimentOutlined />}
                    loading={isRunning}
                  >
                    启动评测
                  </Button>
                  <Button onClick={handleReset} disabled={isRunning}>
                    重置
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        <Col xs={24} lg={14}>
          {phase === 'idle' && (
            <Card>
              <Typography.Paragraph type="secondary" style={{ margin: 0 }}>
                选择评测类型并点击「启动评测」。任务异步执行，通过 GET /jobs/{'{job_id}'} 轮询进度。
                建议先用 limit=5 做冒烟验证。
              </Typography.Paragraph>
            </Card>
          )}

          {isRunning && (
            <Card>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Typography.Text>{statusText}</Typography.Text>
                <Progress percent={100} status="active" showInfo={false} />
                <Typography.Text type="secondary">
                  全量评测可能耗时较长（检索/生成/RAGAS），请勿关闭页面。
                </Typography.Text>
              </Space>
            </Card>
          )}

          {phase === 'error' && errorMessage && (
            <Alert
              type="error"
              showIcon
              message="评测失败"
              description={errorMessage}
            />
          )}

          {phase === 'success' && jobResult && (
            <EvalResultPanel job={jobResult} evalType={activeEvalType} />
          )}
        </Col>
      </Row>
    </PageShell>
  )
}
