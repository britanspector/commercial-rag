# 方案 B：MinerU PDF 解析

在 `feature-pdf-mineru` 分支上运行。

## 安装

```bash
conda activate commercial-rag
pip install -r requirements-mineru.txt
```

首次运行会自动下载 MinerU 模型。若 HuggingFace 访问受限，可设置：

```bash
set MINERU_MODEL_SOURCE=modelscope
```

## 运行

```bash
python src/parse_pdf_mineru.py
python src/check_parser_mineru.py
```

## 配置

在 `src/parse_pdf_mineru.py` 顶部可调整：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MINERU_BACKEND` | `pipeline` | CPU 友好；有 GPU 可尝试 `hybrid-auto-engine` |
| `MINERU_METHOD` | `auto` | 自动判断是否 OCR |
| `MINERU_DEVICE` | `cpu` | 有 CUDA 可改为 `cuda` |

## 输出

| 文件 | 说明 |
|------|------|
| `data/parsed/documents.jsonl` | 每份 PDF 一行，含完整 Markdown |
| `data/parsed/mineru/<doc_id>/` | MinerU 原始输出（含 .md、图片等） |
| `data/parsed/parse_summary.csv` | 解析统计 |

## 对比分支

方案 A（启发式 PyMuPDF）见 `feature-pdf-2` 分支，运行 `python src/parse_pdf_pages.py`。

## 分支切换

```bash
git switch feature-pdf-2        # 方案 A
git switch feature-pdf-mineru   # 方案 B
```
