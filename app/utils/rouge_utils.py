
from rouge_score import rouge_scorer


scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l(query: str, text: str) -> float:
    """
    Returns ROUGE-L F1 score between 0 and 1.
    """
    if not text:
        return 0.0

    score = scorer.score(query, text)
    return float(score["rougeL"].fmeasure)
