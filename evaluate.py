import argparse
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()


def load_labeled(csv_path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in labeled CSV. Available: {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text"}).copy()
    df["text"] = df["text"].astype(str).map(normalize_text)
    df = df.dropna(subset=["text", label_col])
    df = df[df["text"].str.len() > 0]
    return df


def label_from_prices(
    tweets_csv: str,
    prices_csv: str,
    threshold: float,
    tweet_text_col: str = "Tweet",
    tweet_date_col: str = "Date",
    tweet_symbol_col: str = "Stock Name",
    price_date_col: str = "Date",
    price_close_col: str = "Close",
    price_symbol_col: str = "Stock Name",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    # Load tweets
    t = pd.read_csv(tweets_csv)
    for c in [tweet_text_col, tweet_date_col, tweet_symbol_col]:
        if c not in t.columns:
            raise ValueError(f"Tweets CSV missing column '{c}'. Columns: {list(t.columns)}")
    t = t[[tweet_date_col, tweet_symbol_col, tweet_text_col]].copy()
    t[tweet_date_col] = pd.to_datetime(t[tweet_date_col], utc=True, errors="coerce").dt.tz_convert(None)
    t[tweet_symbol_col] = t[tweet_symbol_col].astype(str).str.upper()
    t[tweet_text_col] = t[tweet_text_col].astype(str).map(normalize_text)
    t = t.dropna(subset=[tweet_date_col, tweet_text_col])
    if limit is not None:
        t = t.iloc[:limit].copy()

    # Load prices
    p = pd.read_csv(prices_csv)
    for c in [price_date_col, price_close_col, price_symbol_col]:
        if c not in p.columns:
            raise ValueError(f"Prices CSV missing column '{c}'. Columns: {list(p.columns)}")
    p = p[[price_date_col, price_symbol_col, price_close_col]].copy()
    p[price_date_col] = pd.to_datetime(p[price_date_col], utc=False, errors="coerce")
    p[price_symbol_col] = p[price_symbol_col].astype(str).str.upper()
    p = p.dropna(subset=[price_date_col, price_close_col])

    # Per-symbol asof merge
    symbols = sorted(set(t[tweet_symbol_col]) & set(p[price_symbol_col]))
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        t_sym = t[t[tweet_symbol_col] == sym].sort_values(tweet_date_col)
        p_sym = p[p[price_symbol_col] == sym].sort_values(price_date_col)
        if t_sym.empty or p_sym.empty:
            continue
        prev = pd.merge_asof(
            t_sym,
            p_sym[[price_date_col, price_close_col]].rename(columns={price_date_col: tweet_date_col}),
            on=tweet_date_col,
            direction="backward",
        ).rename(columns={price_close_col: "prev_close"})
        nxt = pd.merge_asof(
            t_sym,
            p_sym[[price_date_col, price_close_col]].rename(columns={price_date_col: tweet_date_col}),
            on=tweet_date_col,
            direction="forward",
        ).rename(columns={price_close_col: "next_close"})
        df_sym = prev[[tweet_date_col, tweet_symbol_col, tweet_text_col]].copy()
        df_sym["prev_close"] = prev["prev_close"].values
        df_sym["next_close"] = nxt["next_close"].values
        frames.append(df_sym)
    if not frames:
        raise ValueError("No aligned data for labeling.")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["prev_close", "next_close"])  # drop entries lacking adjacent price
    ret = (df["next_close"].astype(float) / df["prev_close"].astype(float)) - 1.0
    df["label"] = np.where(ret > threshold, "bullish", np.where(ret < -threshold, "bearish", "neutral"))
    df = df.rename(columns={tweet_text_col: "text"})
    return df[["text", "label"]]


def vader_predict(texts: List[str]) -> List[str]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except Exception as exc:
        raise RuntimeError("vaderSentiment is required: pip install vaderSentiment") from exc
    analyzer = SentimentIntensityAnalyzer()
    labels: List[str] = []
    for t in texts:
        c = float(analyzer.polarity_scores(t).get("compound", 0.0))
        labels.append("bullish" if c > 0.05 else ("bearish" if c < -0.05 else "neutral"))
    return labels


def model_predict(model_path: str, texts: List[str]) -> List[str]:
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload
    return [str(x) for x in pipeline.predict(texts)]


def evaluate(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, str]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, digits=4)
    return acc, f1, report


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate senti1 (scikit) vs VADER")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels-csv", help="CSV with columns: text,label")
    src.add_argument("--tweets-csv", help="Tweets CSV to derive labels from prices")
    p.add_argument("--prices-csv", help="Prices CSV for labeling (required if --tweets-csv used)")
    p.add_argument("--threshold", type=float, default=0.01, help="Return threshold for labeling when deriving")
    p.add_argument("--model", required=True, help="Path to trained scikit model (e.g., models/senti1.joblib)")
    p.add_argument("--limit", type=int, default=None, help="Limit examples for quick evaluation")
    p.add_argument("-v", action="count", default=0, help="Increase verbosity")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.v)
    try:
        if args.labels_csv:
            df = load_labeled(args.labels_csv)
        else:
            if not args.prices_csv:
                raise ValueError("--prices-csv is required when using --tweets-csv")
            df = label_from_prices(
                tweets_csv=args.tweets_csv,
                prices_csv=args.prices_csv,
                threshold=args.threshold,
                limit=args.limit,
            )

        if args.limit is not None:
            df = df.iloc[: args.limit].copy()

        texts = df["text"].astype(str).tolist()
        y_true = df["label"].astype(str).tolist()

        y_vader = vader_predict(texts)
        acc_v, f1_v, rep_v = evaluate(y_true, y_vader)

        y_model = model_predict(args.model, texts)
        acc_m, f1_m, rep_m = evaluate(y_true, y_model)

        print(json.dumps(
            {
                "vader": {"accuracy": acc_v, "f1_weighted": f1_v, "report": rep_v},
                "model": {"accuracy": acc_m, "f1_weighted": f1_m, "report": rep_m},
                "n": len(texts),
            },
            ensure_ascii=False,
        ))
        return 0
    except Exception as exc:
        logging.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


