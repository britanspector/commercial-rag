import json
import random
from pathlib import Path


# ============================================================
# 路径配置
# ============================================================

# 获取当前文件所在目录 (src/)
CURRENT_DIR = Path(__file__).parent
# 向上取父目录，得到项目根目录
PROJECT_ROOT = CURRENT_DIR.parent
INPUT_PAGE_JSONL = PROJECT_ROOT / "data" / "parsed" / "pages.jsonl"

# 随机查看多少页
SAMPLE_COUNT = 10


def load_page_records() -> list[dict]:
    records = []

    with open(INPUT_PAGE_JSONL, "r", encoding="utf-8") as input_file:
        for line in input_file:
            records.append(json.loads(line))

    return records


def print_page_sample(record: dict) -> None:
    print("\n" + "=" * 90)
    print(f"文件名：{record['filename']}")
    print(f"PDF 页码：{record['pdf_page_number']}")
    print(f"正文字符数：{record['text_char_count']}")
    print(f"表格数量：{record['table_count']}")
    print(f"是否需要人工检查：{record['needs_manual_check']}")

    print("\n【正文预览】")
    print(record["text"][:1000])

    if record["tables"]:
        print("\n【表格预览】")
        for index, table in enumerate(record["tables"], start=1):
            print(f"\n表格 {index}：")
            print(table[:1000])


def main() -> None:
    records = load_page_records()

    if not records:
        raise ValueError("解析结果为空，请先运行 PDF 解析脚本。")

    sample_count = min(SAMPLE_COUNT, len(records))
    samples = random.sample(records, sample_count)

    print(f"当前总页面数：{len(records)}")
    print(f"本次随机抽查页面数：{sample_count}")

    for record in samples:
        print_page_sample(record)


if __name__ == "__main__":
    main()