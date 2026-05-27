import json
import random
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
INPUT_CHUNKS_JSONL = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"

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
    print(f"content_type：{record.get('content_type')}")
    print(f"is_retrievable：{record.get('is_retrievable')}")
    print(f"文档：{record['filename']}")
    print(f"公司：{record.get('company_name')} ({record.get('stock_code')})")
    print(f"章节：{record.get('section_title', '')}")
    print(f"页码：{record.get('page_start')} - {record.get('page_end')}")
    print(f"embedding tokens：{record.get('embedding_token_count', 'N/A')}")
    if record.get("table_id"):
        print(
            f"table_id：{record['table_id']} "
            f"part {record.get('table_part_index')}/{record.get('table_part_count')}"
        )

    print("\n【embedding_text 预览】")
    print((record.get("embedding_text") or record["text"])[:PREVIEW_CHARS])


def main() -> None:
    records = load_chunk_records()
    if not records:
        raise ValueError("chunks.jsonl 为空，请先运行 src/chunk_mineru.py。")

    embed_tokens = [record.get("embedding_token_count", 0) for record in records]
    text_count = sum(1 for record in records if record.get("content_type") == "text")
    table_count = sum(1 for record in records if record.get("content_type") == "table")
    noise_count = sum(1 for record in records if record.get("content_type") == "noise")
    over_512 = sum(1 for value in embed_tokens if value > 512)

    print(f"总 chunk 数：{len(records)}")
    print(f"正文：{text_count}  表格：{table_count}  噪声：{noise_count}")
    print(f"可检索：{sum(1 for r in records if r.get('is_retrievable'))}")
    print(f"max embedding tokens：{max(embed_tokens) if embed_tokens else 0}")
    print(f">512 tokens：{over_512}")

    text_samples = random.sample(
        [record for record in records if record.get("content_type") == "text"],
        min(3, text_count),
    )
    table_samples = random.sample(
        [record for record in records if record.get("content_type") == "table"],
        min(3, table_count),
    )

    for record in text_samples + table_samples:
        print_chunk_sample(record)


if __name__ == "__main__":
    main()
