import { Card, Collapse, Space, Tag, Typography } from 'antd'

import type { CitationResponse, RetrievedChunkResponse } from '../../api/types'

const { Paragraph, Text } = Typography

interface CitationListProps {
  citations: CitationResponse[]
  rerankHits: RetrievedChunkResponse[]
}

function formatPage(citation: CitationResponse): string {
  if (citation.page_label) {
    return citation.page_label
  }
  if (citation.page_start === citation.page_end) {
    return `第 ${citation.page_start} 页`
  }
  return `第 ${citation.page_start}-${citation.page_end} 页`
}

function findChunkText(
  chunkId: string,
  rerankHits: RetrievedChunkResponse[],
): string | undefined {
  return rerankHits.find((hit) => hit.chunk_id === chunkId)?.text
}

export function CitationList({ citations, rerankHits }: CitationListProps) {
  if (citations.length === 0) {
    return (
      <Text type="secondary">本次回答未附带引用。</Text>
    )
  }

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      {citations.map((citation) => {
        const chunkText = findChunkText(citation.chunk_id, rerankHits)
        const source =
          citation.source_document ||
          citation.display_name ||
          citation.filename ||
          citation.doc_id ||
          '未知文档'

        return (
          <Card
            key={`${citation.index}-${citation.chunk_id}`}
            size="small"
            title={
              <Space wrap>
                <Tag color="blue">[{citation.index}]</Tag>
                <span>{citation.company_name || source}</span>
              </Space>
            }
            extra={
              <Tag color="purple">
                rerank {(citation.score_rerank ?? 0).toFixed(3)}
              </Tag>
            }
          >
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <Text type="secondary">
                {source} · {formatPage(citation)}
                {citation.section_title ? ` · ${citation.section_title}` : ''}
              </Text>
              {chunkText ? (
                <Collapse
                  ghost
                  size="small"
                  items={[
                    {
                      key: 'text',
                      label: '查看 chunk 原文',
                      children: (
                        <Paragraph
                          style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}
                        >
                          {chunkText}
                        </Paragraph>
                      ),
                    },
                  ]}
                />
              ) : (
                <Text type="secondary">（无 chunk 原文）</Text>
              )}
            </Space>
          </Card>
        )
      })}
    </Space>
  )
}
