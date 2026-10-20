
from search import search_documents

test_cases = [
    {
        "question": "What is multi-head attention?",
        "keywords": ["multi-head", "attention"]
    },
    {
        "question": "What is positional encoding?",
        "keywords": ["positional", "encoding"]
    },
    {
        "question": "What is scaled dot-product attention?",
        "keywords": ["scaled", "dot-product"]
    },
    {
        "question": "What are the encoder and decoder?",
        "keywords": ["encoder", "decoder"]
    },
    {
        "question": "Why does the Transformer use attention?",
        "keywords": ["attention", "recurrent"]
    }
]


def evaluate_retrieval():
    passed = 0

    for case in test_cases:
        results = search_documents(
            case["question"],
            top_k=3
        )

        # Simple keyword-based relevance proxy
        relevant = any(
            all(
                keyword in result["text"].lower()
                for keyword in case["keywords"]
            )
            for result in results
        )

        if relevant:
            passed += 1

        status = "PASS" if relevant else "FAIL"

        print(f"{status}: {case['question']}")

    score = passed / len(test_cases)

    print("\n-------------------------------")
    print(f"Keyword-match proxy@3: {score:.1%}")
    print(f"Passed: {passed}/{len(test_cases)}")


if __name__ == "__main__":
    evaluate_retrieval()
