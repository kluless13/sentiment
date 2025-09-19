import argparse
import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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
    txt = text.replace("\r", " ").replace("\n", " ")
    return " ".join(txt.split()).strip()


def load_tweets(csv_path: str, text_col: str = "Tweet", date_col: str = "Date", stock_col: str = "Stock Name") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tweets CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in [text_col, date_col, stock_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in tweets CSV. Available: {list(df.columns)}")
    df = df[[date_col, stock_col, text_col]].copy()
    df[text_col] = df[text_col].astype(str).map(normalize_text)
    # Parse timezone-aware timestamps and convert to naive UTC
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_convert(None)
    df[stock_col] = df[stock_col].astype(str).str.upper()
    df = df.dropna(subset=[date_col, text_col])
    return df


def load_prices(csv_path: str, date_col: str = "Date", close_col: str = "Close", stock_col: str = "Stock Name") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Prices CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in [date_col, close_col, stock_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in prices CSV. Available: {list(df.columns)}")
    df = df[[date_col, stock_col, close_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df[stock_col] = df[stock_col].astype(str).str.upper()
    df = df.dropna(subset=[date_col, close_col])
    df = df.sort_values([stock_col, date_col]).reset_index(drop=True)
    return df


def label_by_forward_return(
    tweets: pd.DataFrame,
    prices: pd.DataFrame,
    threshold: float = 0.01,
    date_col: str = "Date",
    stock_col: str = "Stock Name",
    close_col: str = "Close",
) -> pd.DataFrame:
    # Ensure dtype alignment and strict sorting required by merge_asof
    tweets = tweets.copy()
    prices = prices.copy()
    tweets[date_col] = pd.to_datetime(tweets[date_col], utc=False, errors="coerce")
    prices[date_col] = pd.to_datetime(prices[date_col], utc=False, errors="coerce")
    tweets = tweets.dropna(subset=[date_col, stock_col]).copy()
    prices = prices.dropna(subset=[date_col, stock_col, close_col]).copy()

    symbols = sorted(set(tweets[stock_col].astype(str)) & set(prices[stock_col].astype(str)))
    if not symbols:
        raise ValueError("No overlapping symbols between tweets and prices.")

    per_symbol_frames = []
    for sym in symbols:
        t_sym = tweets[tweets[stock_col].astype(str) == sym].sort_values(date_col)
        r_sym = prices[prices[stock_col].astype(str) == sym].sort_values(date_col)
        if t_sym.empty or r_sym.empty:
            continue
        prev = pd.merge_asof(
            t_sym,
            r_sym[[date_col, close_col]],
            on=date_col,
            direction="backward",
        ).rename(columns={close_col: "prev_close"})
        nxt = pd.merge_asof(
            t_sym,
            r_sym[[date_col, close_col]],
            on=date_col,
            direction="forward",
        ).rename(columns={close_col: "next_close"})
        df_sym = prev[[date_col, stock_col]].copy()
        df_sym["prev_close"] = prev["prev_close"].values
        df_sym["next_close"] = nxt["next_close"].values
        df_sym = df_sym.join(t_sym.drop(columns=[date_col, stock_col]).reset_index(drop=True))
        per_symbol_frames.append(df_sym)

    if not per_symbol_frames:
        raise ValueError("No data after per-symbol alignment.")

    df = pd.concat(per_symbol_frames, ignore_index=True)

    df = df.dropna(subset=["prev_close", "next_close"]).reset_index(drop=True)
    ret = (df["next_close"].astype(float) / df["prev_close"].astype(float)) - 1.0
    df["forward_return"] = ret
    cond_pos = df["forward_return"] > threshold
    cond_neg = df["forward_return"] < -threshold
    df["label"] = np.where(cond_pos, "bullish", np.where(cond_neg, "bearish", "neutral"))
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    solver="liblinear",
                    multi_class="auto",
                    class_weight=None,
                    n_jobs=None,
                ),
            ),
        ]
    )


def train_model(
    tweets_csv: str,
    prices_csv: Optional[str],
    output_path: str,
    threshold: float,
    test_size: float,
    random_state: int,
    limit: Optional[int],
) -> None:
    tweets = load_tweets(tweets_csv)
    if limit is not None and limit > 0:
        tweets = tweets.iloc[:limit].copy()
    if prices_csv:
        prices = load_prices(prices_csv)
        labeled = label_by_forward_return(tweets, prices, threshold=threshold)
    else:
        # Expect a pre-labeled CSV with 'label' column
        if "label" not in tweets.columns:
            raise ValueError("When prices_csv is not provided, tweets CSV must include a 'label' column.")
        labeled = tweets.rename(columns={"Tweet": "text"}).copy()

    text_col = "Tweet" if "Tweet" in labeled.columns else ("text" if "text" in labeled.columns else None)
    if text_col is None:
        raise ValueError("Could not find text column ('Tweet' or 'text').")

    labeled = labeled[[text_col, "label"]].dropna().rename(columns={text_col: "text"})
    # Drop empty strings
    labeled = labeled[labeled["text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        labeled["text"].values,
        labeled["label"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=labeled["label"].values,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, digits=4)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump({"pipeline": pipeline, "labels": list(sorted(set(y_train)))}, output_path)

    logging.info("Saved model to %s", output_path)
    print(json.dumps({"f1_weighted": f1, "report": report}, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a TF-IDF + LogisticRegression sentiment model")
    p.add_argument("--tweets-csv", required=True, help="Path to tweets CSV (must have Tweet, Date, Stock Name)")
    p.add_argument("--prices-csv", default=None, help="Path to prices CSV to derive labels (optional)")
    p.add_argument("--output", default="models/sentiment_pipeline.joblib", help="Output path for model")
    p.add_argument("--threshold", type=float, default=0.01, help="Return threshold for labels (e.g., 0.01 = 1%)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--limit", type=int, default=None, help="Limit number of tweets for quick training")
    p.add_argument("-v", action="count", default=0, help="Increase verbosity (-v, -vv)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.v)
    try:
        train_model(
            tweets_csv=args.tweets_csv,
            prices_csv=args.prices_csv,
            output_path=args.output,
            threshold=args.threshold,
            test_size=args.test_size,
            random_state=args.random_state,
            limit=args.limit,
        )
        return 0
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


