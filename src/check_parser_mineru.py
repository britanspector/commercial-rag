import json
import random
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
INPUT_DOCUMENTS_JSONL = PROJECT_ROOT / "data" / "parsed" / "documents.jsonl"

SAMPLE_COUNT = 3
PREVIEW_CHARS = 1500


def load_document_records() -> list[dict]:
    records = []

    with open(INPUT_DOCUMENTS_JSONL, "r", encoding="utf-8") as input_file:
        for line in input_file:
            records.append(json.loads(line))

    return records


def print_document_sample(record: dict) -> None:
    print("\n" + "=" * 90)
    print(f"文件名：{record['filename']}")
    print(f"解析方法：{record.get('parse_method', 'unknown')}")
    print(f"Markdown 路径：{record.get('markdown_path', '')}")
    print(f"正文字符数：{record['text_char_count']}")
    print(f"MinerU 配置：{record.get('mineru_backend')} / {record.get('mineru_method')}")
    if record.get("metadata"):
        print(f"文档 metadata：{record['metadata']}")

    print("\n【Markdown 预览】")
    print(record["text"][:PREVIEW_CHARS])


def main() -> None:
    records = load_document_records()

    if not records:
        raise ValueError("解析结果为空，请先运行 src/parse_pdf_mineru.py。")

    sample_count = min(SAMPLE_COUNT, len(records))
    samples = random.sample(records, sample_count)

    print(f"当前文档数：{len(records)}")
    print(f"本次随机抽查文档数：{sample_count}")

    for record in samples:
        print_document_sample(record)


if __name__ == "__main__":
    main()
