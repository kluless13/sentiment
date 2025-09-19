import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def try_import_vader() -> "SentimentIntensityAnalyzer":
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except Exception as exc:  # pragma: no cover
        logging.error(
            "vaderSentiment is required. Install with: pip install vaderSentiment. Error: %s",
            exc,
        )
        raise
    return SentimentIntensityAnalyzer


def normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collect_from_stdin(text_arg: Optional[str], limit: Optional[int]) -> List[str]:
    texts: List[str] = []
    if text_arg:
        texts = [text_arg]
    elif not sys.stdin.isatty():
        data = sys.stdin.read()
        lines = [ln for ln in data.splitlines() if ln.strip()]
        texts = lines
    else:
        raise ValueError("No input text provided. Use --text or pipe input to stdin.")
    if limit is not None:
        texts = texts[: limit]
    return [normalize_text(t) for t in texts if t.strip()]


def guess_text_column(headers: List[str]) -> Optional[str]:
    candidates = [
        "text",
        "tweet",
        "content",
        "body",
        "message",
        "headline",
        "title",
        "comment",
        "review",
        "description",
    ]
    lowered = {h.lower(): h for h in headers}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def collect_from_csv(csv_path: str, text_col: Optional[str], limit: Optional[int]) -> List[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    texts: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header row.")
        chosen_col = text_col or guess_text_column(reader.fieldnames)
        if not chosen_col:
            raise ValueError(
                f"Could not infer text column. Available columns: {reader.fieldnames}. Use --text-col."
            )
        for row in reader:
            val = row.get(chosen_col, "")
            if val and val.strip():
                texts.append(normalize_text(val))
            if limit is not None and len(texts) >= limit:
                break
    return texts


def collect_from_kaggle(dataset: str, text_col: Optional[str], limit: Optional[int]) -> List[str]:
    try:
        import kagglehub  # type: ignore
    except Exception as exc:  # pragma: no cover
        logging.error(
            "kagglehub is required for --source kaggle. Install with: pip install kagglehub. Error: %s",
            exc,
        )
        raise
    base_path = kagglehub.dataset_download(dataset)
    logging.info("Downloaded Kaggle dataset to: %s", base_path)
    # Find the first CSV file
    candidate_csv: Optional[str] = None
    for root, _dirs, files in os.walk(base_path):
        for name in files:
            if name.lower().endswith(".csv"):
                candidate_csv = os.path.join(root, name)
                break
        if candidate_csv:
            break
    if not candidate_csv:
        raise FileNotFoundError("No CSV files found in the Kaggle dataset directory.")
    logging.info("Using CSV: %s", candidate_csv)
    return collect_from_csv(candidate_csv, text_col=text_col, limit=limit)


def analyze_vader(texts: List[str]) -> List[Dict[str, object]]:
    SentimentIntensityAnalyzer = try_import_vader()
    analyzer = SentimentIntensityAnalyzer()
    results: List[Dict[str, object]] = []
    for t in texts:
        scores = analyzer.polarity_scores(t)
        compound = float(scores.get("compound", 0.0))
        if compound > 0.05:
            label = "bullish"
        elif compound < -0.05:
            label = "bearish"
        else:
            label = "neutral"
        results.append(
            {
                "text": t,
                "scores": scores,
                "compound": compound,
                "label": label,
            }
        )
    return results


def aggregate_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(results)
    if n == 0:
        raise ValueError("No documents to aggregate.")
    compounds = [float(r["compound"]) for r in results]
    mean = sum(compounds) / n
    variance = sum((c - mean) ** 2 for c in compounds) / n if n > 1 else 0.0
    stdev = math.sqrt(variance)
    stderr = stdev / math.sqrt(n) if n > 0 else 0.0
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for r in results:
        counts[str(r["label"])]+=1
    if mean > 0.05:
        final_label = "bullish"
    elif mean < -0.05:
        final_label = "bearish"
    else:
        final_label = "neutral"
    return {
        "documents": n,
        "mean_compound": mean,
        "stdev_compound": stdev,
        "stderr_compound": stderr,
        "label_counts": counts,
        "final_label": final_label,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Universal sentiment analysis CLI (VADER baseline)")
    p.add_argument("--source", choices=["stdin", "csv", "kaggle"], default="stdin")
    p.add_argument("--text", help="Single text input when using --source stdin", default=None)
    p.add_argument("--csv-path", help="Path to CSV file when --source csv", default=None)
    p.add_argument("--text-col", help="Text column name for CSVs (will be inferred if omitted)", default=None)
    p.add_argument("--kaggle-dataset", help="Kaggle dataset slug for --source kaggle", default="equinxx/stock-tweets-for-sentiment-analysis-and-prediction")
    p.add_argument("--limit", type=int, default=None, help="Limit number of documents to analyze")
    p.add_argument("--query", default=None, help="Topic/entity descriptor for metadata")
    p.add_argument("--ticker", default=None, help="Symbol like AAPL, NVDA, BTC (metadata)")
    p.add_argument("--json-out", default=None, help="Path to write detailed JSON output")
    p.add_argument("--sk-model", default=None, help="Path to a trained scikit-learn joblib model to use instead of VADER")
    p.add_argument("-v", action="count", default=0, help="Increase verbosity (up to -vv)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.v)

    try:
        if args.source == "stdin":
            texts = collect_from_stdin(args.text, args.limit)
        elif args.source == "csv":
            if not args.csv_path:
                raise ValueError("--csv-path is required for --source csv")
            texts = collect_from_csv(args.csv_path, args.text_col, args.limit)
        else:  # kaggle
            texts = collect_from_kaggle(args.kaggle_dataset, args.text_col, args.limit)

        if not texts:
            logging.error("No texts available for analysis.")
            return 2

        if args.sk_model:
            try:
                import joblib  # type: ignore
            except Exception as exc:
                logging.error("joblib is required for --sk-model. Install with: pip install joblib. Error: %s", exc)
                return 1
            try:
                payload = joblib.load(args.sk_model)
                pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
                proba = None
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(texts)
                preds = pipeline.predict(texts)
                results: List[Dict[str, object]] = []
                # Map to pseudo-compound by class for downstream aggregation
                mapping = {"bullish": 0.7, "neutral": 0.0, "bearish": -0.7}
                for i, t in enumerate(texts):
                    label = str(preds[i])
                    comp = mapping.get(label, 0.0)
                    item: Dict[str, object] = {"text": t, "compound": float(comp), "label": label}
                    if proba is not None:
                        try:
                            classes = list(getattr(pipeline, "classes_", []))
                            # classes_ on the last estimator in Pipeline
                            if not classes and hasattr(pipeline, "named_steps"):
                                last = list(pipeline.named_steps.values())[-1]
                                classes = list(getattr(last, "classes_", []))
                            if classes:
                                class_to_idx = {c: idx for idx, c in enumerate(classes)}
                                probs = {str(c): float(proba[i][idx]) for c, idx in class_to_idx.items()}
                                item["probabilities"] = probs
                        except Exception:
                            pass
                    results.append(item)
            except Exception as exc:
                logging.exception("Failed to load/use scikit model: %s", exc)
                return 1
        else:
            results = analyze_vader(texts)
        summary = aggregate_results(results)

        concise = {
            "final_label": summary["final_label"],
            "mean_compound": round(float(summary["mean_compound"]), 4),
            "documents": int(summary["documents"]),
            "label_counts": summary["label_counts"],
        }

        meta: Dict[str, Optional[str]] = {"query": args.query, "ticker": args.ticker}
        print(json.dumps({"summary": concise, "meta": meta}, ensure_ascii=False))

        if args.json_out:
            payload = {"meta": meta, "summary": summary, "results": results}
            with open(args.json_out, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
            logging.info("Wrote JSON output to %s", args.json_out)

        return 0
    except Exception as exc:  # pragma: no cover
        logging.exception("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
