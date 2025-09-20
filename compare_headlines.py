import argparse
import json
from typing import List, Dict

from collectors.finviz import fetch_finviz_headlines


def analyze_vader(texts: List[str]) -> List[Dict[str, float]]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        sc = an.polarity_scores(t)
        out.append({"compound": float(sc.get("compound", 0.0))})
    return out


def analyze_model(model_path: str, texts: List[str]) -> List[Dict[str, float]]:
    import joblib
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
    res: List[Dict[str, float]] = []
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)
        classes = list(getattr(pipeline, "classes_", []))
        if not classes and hasattr(pipeline, "named_steps"):
            last = list(pipeline.named_steps.values())[-1]
            classes = list(getattr(last, "classes_", []))
        idx = {c: i for i, c in enumerate(classes)}
        for i, _ in enumerate(texts):
            bull = float(proba[i][idx.get("bullish", 0)]) if "bullish" in idx else 0.0
            bear = float(proba[i][idx.get("bearish", 0)]) if "bearish" in idx else 0.0
            res.append({"score": bull - bear})
    else:
        preds = pipeline.predict(texts)
        mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        for y in preds:
            res.append({"score": float(mapping.get(str(y), 0.0))})
    return res


def summarize(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"mean": 0.0, "n": 0}
    mean = sum(scores) / len(scores)
    return {"mean": mean, "n": len(scores)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare VADER vs models on FinViz headlines")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--finance-model", default="models/finance_current.joblib")
    ap.add_argument("--signals-model", default="models/finance_signals_current.joblib")
    args = ap.parse_args()

    headlines = fetch_finviz_headlines(args.ticker)
    vader = analyze_vader(headlines)
    fin = analyze_model(args.finance_model, headlines)
    sig = analyze_model(args.signals_model, headlines)

    out = {
        "ticker": args.ticker,
        "num_headlines": len(headlines),
        "vader": summarize([x["compound"] for x in vader]),
        "finance_model": summarize([x["score"] for x in fin]),
        "signals_model": summarize([x["score"] for x in sig]),
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


