import json
import random
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
INPUT_CHUNKS_JSONL = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"

SAMPLE_COUNT = 8
PREVIEW_CHARS = 600


def load_chunk_records() -> list[dict]:
    records = []

    with open(INPUT_CHUNKS_JSONL, "r", encoding="utf-8") as input_file:
        for line in input_file:
            records.append(json.loads(line))

    return records


def print_chunk_sample(record: dict) -> None:
    print("\n" + "=" * 90)
    print(f"chunk_id：{record['chunk_id']}")
    print(f"文档：{record['filename']}")
    print(f"章节：{record.get('section_title', '')}")
    print(f"页码范围：{record.get('page_start')} - {record.get('page_end')}")
    print(f"字符数：{record['char_count']}")
    if record.get("metadata"):
        print(f"metadata：{record['metadata']}")

    print("\n【正文预览】")
    print(record["text"][:PREVIEW_CHARS])


def main() -> None:
    records = load_chunk_records()

    if not records:
        raise ValueError("chunks.jsonl 为空，请先运行 src/chunk_mineru.py。")

    sample_count = min(SAMPLE_COUNT, len(records))
    samples = random.sample(records, sample_count)

    print(f"当前总 chunk 数：{len(records)}")
    print(f"本次随机抽查 chunk 数：{sample_count}")

    for record in samples:
        print_chunk_sample(record)


if __name__ == "__main__":
    main()
