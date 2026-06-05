import { useRef, useState } from 'react'
import { InboxOutlined } from '@ant-design/icons'
import {
  Alert,
  Button,
  Form,
  Progress,
  Select,
  Space,
  Switch,
  Typography,
  Upload,
} from 'antd'
import type { UploadFile } from 'antd/es/upload/interface'

import { pollJobUntilDone } from '../../api/jobs'
import { uploadPdf } from '../../api/upload'
import type { JobStatusResponse, UploadResponse } from '../../api/types'
import { PageShell } from '../../components/PageShell'
import { getIndustryLabel, INDUSTRY_OPTIONS } from '../../constants/industries'
import { getNavItemByPath } from '../../navigation'
import { AsyncJobResultPanel, UploadResultPanel } from './UploadResultPanel'

type UploadPhase = 'idle' | 'uploading' | 'processing' | 'success' | 'error'

interface UploadFormValues {
  industry: string
  replace_existing: boolean
  background: boolean
}

const { Dragger } = Upload
const { Text } = Typography

export function UploadPage() {
  const navItem = getNavItemByPath('/upload')!
  const [form] = Form.useForm<UploadFormValues>()
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [phase, setPhase] = useState<UploadPhase>('idle')
  const [uploadPercent, setUploadPercent] = useState(0)
  const [statusText, setStatusText] = useState('')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)
  const [jobResult, setJobResult] = useState<JobStatusResponse | null>(null)
  const abortRef = useRef(false)

  const selectedFile = fileList[0]?.originFileObj
  const isBusy = phase === 'uploading' || phase === 'processing'

  const resetResult = () => {
    setErrorMessage(null)
    setUploadResult(null)
    setJobResult(null)
    setUploadPercent(0)
    setStatusText('')
  }

  const handleSubmit = async () => {
    if (!selectedFile) {
      setErrorMessage('请先选择 PDF 文件')
      setPhase('error')
      return
    }

    if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
      setErrorMessage('仅支持 PDF 文件')
      setPhase('error')
      return
    }

    const values = await form.validateFields()
    abortRef.current = false
    resetResult()
    setPhase('uploading')
    setStatusText('正在上传文件…')

    try {
      const response = await uploadPdf({
        file: selectedFile,
        industry: values.industry === 'uploads' ? '' : values.industry,
        industry_label:
          values.industry === 'uploads' ? '' : getIndustryLabel(values.industry),
        replace_existing: values.replace_existing,
        background: values.background,
        onUploadProgress: (percent) => {
          setUploadPercent(percent)
          if (percent >= 100) {
            setStatusText(values.background ? '上传完成，任务已提交…' : '上传完成，正在入库…')
          }
        },
      })

      if (abortRef.current) {
        return
      }

      if (response.async_mode && response.job_id) {
        setPhase('processing')
        setStatusText('后台任务处理中…')
        const job = await pollJobUntilDone(response.job_id, {
          onUpdate: (current) => {
            setStatusText(
              current.progress
                ? `任务状态：${current.status}（${current.progress}）`
                : `任务状态：${current.status}`,
            )
          },
        })

        if (abortRef.current) {
          return
        }

        if (job.status === 'failed') {
          throw new Error(job.error || '后台入库失败')
        }

        setJobResult(job)
        setPhase('success')
        setStatusText('入库完成')
        return
      }

      setUploadResult(response)
      setPhase('success')
      setStatusText('入库完成')
    } catch (error) {
      if (abortRef.current) {
        return
      }
      setPhase('error')
      setErrorMessage(error instanceof Error ? error.message : '上传失败')
      setStatusText('')
    }
  }

  const handleReset = () => {
    abortRef.current = true
    setFileList([])
    setPhase('idle')
    resetResult()
    form.resetFields()
  }

  return (
    <PageShell
      title={navItem.label}
      description={navItem.description}
      apiEndpoints={navItem.apiEndpoints}
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          industry: 'uploads',
          replace_existing: true,
          background: false,
        }}
        style={{ maxWidth: 720 }}
      >
        <Form.Item
          label="行业分类"
          name="industry"
          rules={[{ required: true, message: '请选择行业' }]}
        >
          <Select options={INDUSTRY_OPTIONS} disabled={isBusy} />
        </Form.Item>

        <Form.Item label="PDF 文件" required>
          <Dragger
            accept=".pdf,application/pdf"
            maxCount={1}
            disabled={isBusy}
            fileList={fileList}
            beforeUpload={() => false}
            onChange={({ fileList: nextList }) => {
              setFileList(nextList.slice(-1))
              if (phase === 'error') {
                setPhase('idle')
                setErrorMessage(null)
              }
            }}
            onRemove={() => {
              setFileList([])
              return true
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">点击或拖拽 PDF 到此处</p>
            <p className="ant-upload-hint">仅支持 .pdf，单文件上限 100MB</p>
          </Dragger>
        </Form.Item>

        <Form.Item
          label="覆盖同 doc_id 旧数据"
          name="replace_existing"
          valuePropName="checked"
        >
          <Switch disabled={isBusy} />
        </Form.Item>

        <Form.Item
          label="后台异步入库"
          name="background"
          valuePropName="checked"
          extra="大文件建议开启，提交后通过任务轮询查看进度"
        >
          <Switch disabled={isBusy} />
        </Form.Item>

        <Form.Item>
          <Space>
            <Button type="primary" onClick={() => void handleSubmit()} loading={isBusy}>
              开始上传
            </Button>
            <Button onClick={handleReset} disabled={isBusy}>
              重置
            </Button>
          </Space>
        </Form.Item>
      </Form>

      {isBusy && (
        <div style={{ maxWidth: 720, marginBottom: 16 }}>
          <Text type="secondary">{statusText}</Text>
          {phase === 'uploading' && (
            <Progress
              percent={uploadPercent}
              status={uploadPercent >= 100 ? 'success' : 'active'}
              style={{ marginTop: 8 }}
            />
          )}
          {phase === 'processing' && (
            <Progress percent={100} status="active" showInfo={false} style={{ marginTop: 8 }} />
          )}
        </div>
      )}

      {phase === 'error' && errorMessage && (
        <Alert
          type="error"
          showIcon
          message="上传失败"
          description={errorMessage}
          style={{ maxWidth: 720, marginBottom: 16 }}
        />
      )}

      {phase === 'success' && uploadResult && (
        <UploadResultPanel result={uploadResult} />
      )}

      {phase === 'success' && jobResult && (
        <AsyncJobResultPanel job={jobResult} />
      )}
    </PageShell>
  )
}
