from typing import List, Dict
import math
from rouge_score import rouge_scorer
from rouge_score import rouge_scorer



def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """
    How many retrieved items are relevant?
    """
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)

    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """
    How many relevant items were retrieved?
    """
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)

    return hits / len(relevant) if relevant else 0


def rouge_l(hypothesis: str, reference: str) -> float:
    """
    Simple ROUGE-L metric based on longest common subsequence.
    """

    def lcs(x, y):
        table = [[0] * (len(y)+1) for _ in range(len(x)+1)]
        for i in range(1, len(x)+1):
            for j in range(1, len(y)+1):
                if x[i-1] == y[j-1]:
                    table[i][j] = table[i-1][j-1] + 1
                else:
                    table[i][j] = max(table[i-1][j], table[i][j-1])
        return table[-1][-1]

    l = lcs(hypothesis, reference)
    return l / max(len(reference), 1)


def evaluate_retrieval(results: List[Dict], ground_truth: str) -> Dict:
    """
    Evaluate a RAG retrieval run.
    `results` is list of chunks returned by Qdrant
    """

    retrieved_texts = [r["text"] for r in results]

    rouge_scores = [rouge_l(text, ground_truth) for text in retrieved_texts]

    return {
        "rouge_mean": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0,
        "precision@5": precision_at_k(retrieved_texts, [ground_truth], k=5),
        "recall@5": recall_at_k(retrieved_texts, [ground_truth], k=5)
    }






def compute_similarity_stats(results: List[Dict]):
    """
    Compute min/avg/max similarity score from vector DB results.
    """
    if not results:
        return {"min": 0, "avg": 0, "max": 0}

    scores = [r.get("score", 0) for r in results]

    return {
        "min": round(min(scores), 4),
        "avg": round(sum(scores) / len(scores), 4),
        "max": round(max(scores), 4)
    }


def compute_rouge_relevance(answer: str, retrieved_texts: List[str]):
    """
    Measures how much retrieved context relates to final answer.
    Uses simple ROUGE-L (assignment allows lightweight metrics).
    """

    if not answer or not retrieved_texts:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    scores = []
    for chunk in retrieved_texts:
        score = scorer.score(answer, chunk)["rougeL"].fmeasure
        scores.append(score)

    avg_rouge = sum(scores) / len(scores)
    return round(avg_rouge, 4)


def compute_hit_rate(results: List[Dict], true_doc_id: str | None):
    """
    Measures how many retrieved chunks belong to the expected document.
    Only useful when doc_id is provided in QueryRequest.
    """

    if not true_doc_id:
        return 0.0

    if not results:
        return 0.0

    hits = 0
    for r in results:
        if r.get("doc_id") == true_doc_id:
            hits += 1

    return round(hits / len(results), 4)
