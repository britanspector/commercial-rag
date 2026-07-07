import type { ReactNode } from 'react'
import {
  ApartmentOutlined,
  ApiOutlined,
  AuditOutlined,
  CloudUploadOutlined,
  CommentOutlined,
  DashboardOutlined,
  ExperimentOutlined,
  SearchOutlined,
} from '@ant-design/icons'

export type NavGroup = 'overview' | 'core' | 'future'

export interface NavItem {
  key: string
  path: string
  label: string
  icon: ReactNode
  group: NavGroup
  description: string
  apiEndpoints: string[]
  plannedFeatures: string[]
  comingSoon?: boolean
}

export const NAV_GROUP_LABELS: Record<NavGroup, string> = {
  overview: '概览',
  core: 'RAG 功能',
  future: '后续扩展',
}

export const NAV_ITEMS: NavItem[] = [
  {
    key: '/',
    path: '/',
    label: '概览',
    icon: <DashboardOutlined />,
    group: 'overview',
    description: '查看系统状态与各功能入口。',
    apiEndpoints: ['GET /health'],
    plannedFeatures: [],
  },
  {
    key: '/upload',
    path: '/upload',
    label: 'PDF 上传 / 知识库构建',
    icon: <CloudUploadOutlined />,
    group: 'core',
    description:
      '上传研报 PDF，触发解析、分块、向量与 BM25 索引构建，完成知识库入库。',
    apiEndpoints: ['POST /upload', 'GET /jobs/{job_id}'],
    plannedFeatures: [
      '行业选择与 PDF 文件上传',
      '同步 / 异步入库模式',
      '解析、分块、向量化阶段进度展示',
    ],
  },
  {
    key: '/chat',
    path: '/chat',
    label: 'RAG 问答',
    icon: <CommentOutlined />,
    group: 'core',
    description: '输入金融研报相关问题，展示答案、引用来源与拒答原因。',
    apiEndpoints: ['POST /chat'],
    plannedFeatures: [
      '问题输入与可选股票代码 / 题型',
      '答案与引用列表展示',
      'Rerank 命中详情',
      '响应 cache 字段：本次命中/延迟',
    ],
  },
  {
    key: '/search',
    path: '/search',
    label: '检索调试',
    icon: <SearchOutlined />,
    group: 'core',
    description: '查看召回与重排结果，用于调试检索质量（不生成答案）。',
    apiEndpoints: ['POST /search'],
    plannedFeatures: [
      'Query Rewrite 信息',
      'Recall / Rerank 分阶段命中列表',
      '召回路线与 Top-K 参数调节',
      '响应 cache 字段：本次命中/延迟',
    ],
  },
  {
    key: '/eval',
    path: '/eval',
    label: '自动化评测',
    icon: <ExperimentOutlined />,
    group: 'core',
    description: '触发批量评测任务并查看进度与结果摘要。',
    apiEndpoints: ['POST /eval', 'GET /jobs/{job_id}'],
    plannedFeatures: [
      'generation / retrieval / ragas 评测类型',
      '异步任务状态轮询',
      '评测参数配置',
    ],
  },
  {
    key: '/cache',
    path: '/cache',
    label: '缓存监控',
    icon: <ApiOutlined />,
    group: 'core',
    description: '展示语义缓存累计命中率、延迟与后端状态，对接 GET /cache/stats。',
    apiEndpoints: ['GET /cache/stats', 'GET /health'],
    plannedFeatures: [
      '累计 L1/L2 命中率与延迟统计',
      '向量检索 / LLM 调用节省量',
      'L1/L2 后端状态',
    ],
  },
  {
    key: '/trace',
    path: '/trace',
    label: 'Agent Trace',
    icon: <AuditOutlined />,
    group: 'future',
    description: '展示 Agent 执行轨迹与 Tool 调用（后续接入 LangGraph）。',
    apiEndpoints: [],
    plannedFeatures: ['节点轨迹时间线', 'Tool 输入输出'],
    comingSoon: true,
  },
  {
    key: '/workflow',
    path: '/workflow',
    label: 'Multi-Agent 工作流',
    icon: <ApartmentOutlined />,
    group: 'future',
    description: '可视化 Multi-Agent 编排与状态图（后续接入）。',
    apiEndpoints: [],
    plannedFeatures: ['工作流状态图', '断点续跑与人工审核'],
    comingSoon: true,
  },
]

export function getNavItemByPath(pathname: string): NavItem | undefined {
  return NAV_ITEMS.find((item) => item.path === pathname)
}

export function getCoreNavItems(): NavItem[] {
  return NAV_ITEMS.filter((item) => item.group === 'core')
}

export function getFutureNavItems(): NavItem[] {
  return NAV_ITEMS.filter((item) => item.group === 'future')
}
