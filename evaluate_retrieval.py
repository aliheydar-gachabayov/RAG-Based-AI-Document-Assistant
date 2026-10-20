
import json
from pathlib import Path

from search import search_documents

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "evaluation_dataset.json"


def evaluate():
    with open(DATASET_PATH, encoding="utf-8") as file:
        test_cases = json.load(file)

    hits = []
    reciprocal_ranks = []

    for case in test_cases:
        question = case["question"]
        relevant_pages = set(case["relevant_pages"])

        if not relevant_pages:
            raise ValueError(
                f"Missing ground truth for: {question}"
            )

        results = search_documents(question, top_k=3)

        first_relevant_rank = None

        for rank, result in enumerate(results, start=1):
            if result["page"] in relevant_pages:
                first_relevant_rank = rank
                break

        hit = 1 if first_relevant_rank else 0
        rr = 1 / first_relevant_rank if first_relevant_rank else 0

        hits.append(hit)
        reciprocal_ranks.append(rr)

        print(f"\nQuestion: {question}")
        print(f"Relevant pages: {sorted(relevant_pages)}")
        print(f"Retrieved pages: {[r['page'] for r in results]}")
        print(f"Hit@3: {hit}")
        print(f"Reciprocal rank: {rr:.3f}")

    print("\n=== FINAL EVALUATION ===")
    print(f"Questions: {len(test_cases)}")
    print(f"Hit Rate@3: {sum(hits) / len(hits):.2%}")
    print(
        f"MRR@3: "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
    )


if __name__ == "__main__":
    evaluate()
