# 分块（Chunk）说明

## 运行

```bash
conda activate commercial-rag
pip install -r requirements.txt
python src/chunk_mineru.py
python src/check_chunks.py
```

## 策略（mineru_paragraph_v3）

### 正文 chunk
- 基于 MinerU v2 段落单元合并
- 优先在句号/分号边界切分，避免句中断开
- 单段超长时使用少量 overlap（20 tokens）
- `embedding_text` 上限 **512 tokens**

### 表格 chunk
- 长表按行组拆分，每块 **≤440 tokens**
- 每块重复：公司、章节、表格标题、单位、列头、页码
- `table_raw` 保留原始表格文本
- `embedding_text` 含自然语言描述（如「2026E 营业收入为xx百万元」）
- 同表子块共享 `table_id`

### 噪声处理
- 免责声明、分析师联系方式、研报链接等 → `content_type=noise`, `is_retrievable=false`
- 「风险提示」正文保留且可检索

## 主要字段

| 字段 | 说明 |
|------|------|
| `embedding_text` | 送入 embedding 模型的文本 |
| `table_raw` | 表格原始文本 |
| `table_id` | 同表子块 ID |
| `content_type` | text / table / noise |
| `is_retrievable` | 是否进入向量库 |
| `company_name`, `stock_code`, `broker`, `report_title`, `report_date`, `rating` | 文档元数据 |
