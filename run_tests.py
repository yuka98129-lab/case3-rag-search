"""
案件3 RAGパイプライン - 精度検証スクリプト
case3-test-questions.csv の12問を実行し、結果をレポートにまとめる
"""
import csv
import json
import time

from search_and_answer import answer_question

CSV_PATH = "/mnt/user-data/uploads/Downloads/case3-test-questions.csv"


def main():
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = []
    for row in rows:
        print(f"Q{row['番号']}: {row['質問']}")
        result = answer_question(row["質問"])
        result["expected"] = row["期待される回答"]
        result["expected_source"] = row["出典文書"]
        result["type"] = row["種別"]
        results.append(result)
        print(f"  -> {result['answer'][:80]}...")
        time.sleep(0.5)

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n完了。test_results.json に保存しました。")


if __name__ == "__main__":
    main()
