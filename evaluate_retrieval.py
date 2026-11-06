
import json
from pathlib import Path

from search import search_documents, search_documents_mmr

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "evaluation_dataset.json"


def evaluate(search_function, method_name):
    with open(DATASET_PATH, encoding="utf-8") as file:
        test_cases = json.load(file)

    for k in [3, 10]:
        hits = []
        reciprocal_ranks = []

        print(f"\n{'=' * 60}")
        print(f"{method_name}: TOP {k}")
        print(f"{'=' * 60}")

        for case in test_cases:
            question = case["question"]
            relevant_pages = set(case["relevant_pages"])

            results = search_function(question, top_k=k)

            first_relevant_rank = None

            for rank, result in enumerate(results, start=1):
                if result["page"] in relevant_pages:
                    first_relevant_rank = rank
                    break

            hit = 1 if first_relevant_rank else 0

            reciprocal_rank = (
                1 / first_relevant_rank
                if first_relevant_rank
                else 0
            )

            hits.append(hit)
            reciprocal_ranks.append(reciprocal_rank)

            print(f"\nQuestion: {question}")
            print(f"First relevant rank: {first_relevant_rank}")
            print(f"Hit@{k}: {hit}")
            print(f"Reciprocal rank: {reciprocal_rank:.3f}")

        print(f"\nHit Rate@{k}: {sum(hits) / len(hits):.3f}")
        print(
            f"MRR@{k}: "
            f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
        )


if __name__ == "__main__":
    evaluate(search_documents, "STANDARD RETRIEVAL")
    evaluate(search_documents_mmr, "MMR RETRIEVAL")
